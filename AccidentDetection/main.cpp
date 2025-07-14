#define _CRT_SECURE_NO_WARNINGS

#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/background_segm.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <time.h>

#include "include/yolo_utils.h"
#include "include/detector.h"
#include "include/Intersection.h"
#include "include/MultiCams.h"
#include "include/Multitracker.h"
#include "include/Display.h"
#include "include/SpeedEstimator.h"

using namespace cv;
using namespace std;
string getFileName(const string &filePath);
Point2f Point_warp(const Mat &Trans, const Point &p);
string getTimeString(int numFrame, int frameRate, int h = 0, int m = 0, int s = 0);
float overlapRate(const Rect &box1, const Rect &box2);

int main(int argc, char **argv)
{
    Mat frame;
    vector<Mat> icons;
    string timestamp;
    vector<Mat> rois;
    vector<int> car_size;
    int ave_car = 0;
    Scalar Colors[9] = {Scalar(255, 0, 0), Scalar(0, 0, 255), Scalar(0, 255, 0), Scalar(255, 255, 0),
                        Scalar(0, 255, 255), Scalar(255, 0, 255), Scalar(255, 127, 255), Scalar(127, 0, 255), Scalar(127, 0, 127)};

    double a[3][3] = {{1169.1, 0, 959.5}, {0, 1168.66, 539.5}, {0, 0, 1}};
    Mat cameraMatrix(3, 3, CV_64F, a);
    double b[5] = {-0.396267, 0.202849, 0.00852173, -0.00236845, -0.0458549};
    Mat distCoeffs(1, 5, CV_64F, b);

    // configs
    const float confThreshold = 0.3f;
    const float iouThreshold = 0.4f;
    bool isGPU = true;
    string resultPath = "./results/";
    string iconPath = "./icons/";
    string classNamesPath = "./data/traffic.names";
    string GISPath = "GIS_stmarc.xml";
    string modelPath = "models/yolov5s.onnx";
    string cameras_file = "cameras.xml";

    if (argc > 1)
        GISPath = argv[1];
    if (argc > 2)
        modelPath = argv[2];
    if (argc > 3)
        cameras_file = argv[3];

    // load class names
    auto classNames = yolo_utils::loadNames(classNamesPath);
    if (classNames.empty())
    {
        cerr << "Error: Empty class names file." << endl;
        return -1;
    }

    // object detector
    YOLODetector detector{nullptr};
    try
    {
        detector = YOLODetector(modelPath, isGPU, Size(1280, 1280));
        cout << "Model was initialized." << endl;
    }
    catch (const exception &e)
    {
        cerr << e.what() << endl;
        return -1;
    }

    // read video files and transmission matrices
    MultiCams *cameras = new MultiCams(cameras_file);
    vector<VideoCapture> cams;
    for (int i = 0; i < (int)cameras->video_addresses.size(); i++)
    {
        VideoCapture cam(cameras->video_addresses[i]);
        if (!cam.isOpened())
        {
            cerr << "Error opening video stream or file " << i << endl;
            return -1;
        }
        cams.push_back(move(cam));
    }

    // load intersection information
    Intersection GIS_map;
    GIS_map.loadConfig(GISPath);

    // create tracks
    Multitracker *multiTracker = new Multitracker;

    // speed estimation
    SpeedEstimator *speedEstimator = new SpeedEstimator;

    // display windows
    namedWindow("output", WINDOW_NORMAL);
    namedWindow("display", WINDOW_NORMAL);

    // load icons
    for (int i = 0; i < (int)classNames.size(); i++)
    {
        Mat icon = imread(iconPath + classNames[i] + ".jpg");
        resize(icon, icon, Size(100, 100));
        icons.push_back(icon);
    }

    // output video writers and files
    string videoName = getFileName(cameras->video_addresses[0]);
    int frameRate = (int)cams[0].get(CAP_PROP_FPS);
    int width = (int)cams[0].get(CAP_PROP_FRAME_WIDTH);
    int height = (int)cams[0].get(CAP_PROP_FRAME_HEIGHT);
    VideoWriter outputVideo(resultPath + videoName + "_result.mp4",
                            VideoWriter::fourcc('A', 'V', 'C', '1'), frameRate, Size(width, height), true);
    VideoWriter outputVideo2(resultPath + videoName + "_map.mp4",
                             VideoWriter::fourcc('A', 'V', 'C', '1'), frameRate,
                             Size(GIS_map.lane_map.cols, GIS_map.lane_map.rows), true);
    ofstream out(resultPath + videoName + ".txt");
    ofstream lane_change_file(resultPath + videoName + "_lane_change.txt");
    ofstream counting_file(resultPath + videoName + "_counting.txt");
    ofstream conflict_file(resultPath + videoName + "_conflict.txt");

    // set ROI masks
    for (int i = 0; i < (int)cameras->video_addresses.size(); i++)
    {
        Mat mask(Size(width, height), CV_8UC1, Scalar::all(255));
        if (cameras->rois[i].size() >= 3)
        {
            mask = Mat(Size(width, height), CV_8UC1, Scalar::all(0));
            const Point *pts = (const Point *)Mat(cameras->rois[i]).data;
            int npts = Mat(cameras->rois[i]).rows;
            fillConvexPoly(mask, pts, npts, 255, LINE_8);
        }
        rois.push_back(mask);
    }

    int n_frame = 0;
    bool bSuccess = true;
    int cam_n = (int)cams.size();

    while ((char)waitKey(1) != 'q' && bSuccess)
    {
        n_frame++;
        timestamp = getTimeString(n_frame, frameRate,
                                  cameras->hours[0], cameras->minutes[0], cameras->seconds[0]);

        vector<Detection> traffic_detections;
        Mat display = GIS_map.bg.clone();

        for (int c = 0; c < cam_n; ++c)
        {
            Mat original;
            if (!cams[c].read(original))
            {
                cerr << "[Error] failed to read frame " << n_frame
                     << " from camera " << c << endl;
                bSuccess = false;
                break;
            }

            undistort(original, frame, cameraMatrix, distCoeffs);
            vector<Detection> result = detector.detect(frame, confThreshold, iouThreshold);
            polylines(frame, cameras->rois[c], true, Scalar(0, 0, 255), 3);

            bool skipCurrent = false;
            for (auto &det : result)
            {
                Point2f center(det.box.x + det.box.width / 2,
                               det.box.y + det.box.height / 2);
                if (rois[c].at<uchar>(center.y, center.x) == 0)
                    continue;

                Detection d = det;
                d.front_point = cameras->front_point[c];
                if (d.orig_box.x < 10 || d.orig_box.y < 10 ||
                    d.orig_box.x + d.orig_box.width > frame.cols - 10 ||
                    d.orig_box.y + d.orig_box.height > frame.rows - 10)
                    continue;

                int x = INT_MAX, y = INT_MAX;
                for (int k = 0; k < 4; ++k)
                {
                    Point p;
                    switch (k)
                    {
                    case 0:
                        p = det.box.tl();
                        break;
                    case 1:
                        p = Point(det.box.x, det.box.y + det.box.height);
                        break;
                    case 2:
                        p = det.box.br();
                        break;
                    default:
                        p = Point(det.box.x + det.box.width, det.box.y);
                        break;
                    }
                    Point warpPt = Point_warp(cameras->tMatrix[c], p);
                    if (warpPt.x <= 0 || warpPt.y <= 0 ||
                        warpPt.x >= display.cols || warpPt.y >= display.rows)
                    {
                        skipCurrent = true;
                        break;
                    }
                    x = min(x, warpPt.x);
                    y = min(y, warpPt.y);
                    d.outline.push_back(warpPt);
                }
                if (skipCurrent)
                    continue;

                Point2f warpedCenter = Point_warp(cameras->tMatrix[c], center);
                if (warpedCenter.x <= 0 || warpedCenter.y <= 0 ||
                    warpedCenter.x >= display.cols || warpedCenter.y >= display.rows)
                    continue;
                d.centroid = warpedCenter;
                d.box = Rect(x, y, abs(2 * (warpedCenter.x - x)), abs(2 * (warpedCenter.y - y)));

                bool found = false;
                for (auto &td : traffic_detections)
                {
                    if (abs(td.centroid.x - warpedCenter.x) < 20 &&
                        abs(td.centroid.y - warpedCenter.y) < 20)
                    {
                        found = true;
                        break;
                    }
                }
                if (!found)
                    traffic_detections.push_back(d);
            }
        }

        if (!bSuccess)
            break;

        if (!traffic_detections.empty())
        {
            multiTracker->update_tracks(traffic_detections, GIS_map, 1);
            speedEstimator->detectSpeed(multiTracker->tracks, frameRate, GIS_map.ftpp);
            multiTracker->lane_change(timestamp, lane_change_file);
            multiTracker->countV(timestamp, frame, counting_file, GIS_map);
            multiTracker->conflict_detection(timestamp, classNames, frame, conflict_file, display);
        }

        out << "======================= frame: " << n_frame << endl;
        out << "Time: " << timestamp << endl;
        putText(frame, cameras->cam_info[0], Point(30, 30), FONT_ITALIC, 1, Scalar(255, 255, 255), 2);
        putText(frame, to_string(n_frame), Point(frame.cols - 100, frame.rows - 30), FONT_ITALIC, 0.8, Scalar(255, 255, 255), 2);
        putText(display, to_string(n_frame), Point(display.cols - 100, display.rows - 30), FONT_ITALIC, 0.8, Scalar(255, 255, 255), 2);
        putText(frame, timestamp, Point(frame.cols - 300, 30), FONT_ITALIC, 0.8, Scalar(0, 0, 255), 2);
        putText(display, timestamp, Point(display.cols - 300, 30), FONT_ITALIC, 0.8, Scalar(0, 0, 0), 2);

        imshow("display", display);
        imshow("output", frame);
        outputVideo << frame;
        outputVideo2 << display;
        waitKey(1);
    }

    outputVideo.release();
    outputVideo2.release();
    return 0;
}

std::string
getFileName(std::string filePath)
{
	std::string rawname, seperator;
	if (filePath.find_last_of('\\') == string::npos)
		seperator = '/';
	else
		seperator = '\\';

	std::size_t dotPos = filePath.rfind('.');
	std::size_t sepPos = filePath.rfind(seperator);

	rawname = filePath.substr(sepPos + 1, dotPos - sepPos - 1);

	return rawname;
}

Point2f Point_warp(Mat Trans, Point p)

{

	cv::Mat_<double> src(3, 1);
	src(0, 0) = p.x;
	src(1, 0) = p.y;
	src(2, 0) = 1.0;
	Mat_<double> dst = Trans * src;
	for (int i = 0; i < dst.cols; i++) {
		dst(0, i) = dst(0, i) / dst(2, i);
		dst(1, i) = dst(1, i) / dst(2, i);
	}
	//if (dst(0, 0) < 0)
	//	dst(0, 0) = 0;
	//if (dst(1, 0) < 0)
	//	dst(1, 0) = 0;
	return Point2f(dst(0, 0), dst(1, 0));
}

string getTimeString(int numFrame, int frameRate, int h, int m, int s) {
	// Add result to file
	int totalSecs = (numFrame / frameRate);
	int hours = (totalSecs / 3600 + h) % 24;
	int minutes = (totalSecs % 3600) / 60 + m;
	int seconds = totalSecs % 60 + s;
	if (seconds >= 60) {
		minutes++;
		seconds -= 60;
	}
	if (minutes >= 60) {
		hours++;
		minutes -= 60;
	}
	int miliseconds = numFrame % frameRate * 1000 / frameRate;
	string timeString = format("%02d:%02d:%02d.%03d", hours, minutes, seconds, miliseconds);
	return timeString;
}

float overlapRate(cv::Rect box1, cv::Rect box2)
{
	int x1 = box1.x, y1 = box1.y, w1 = box1.width, h1 = box1.height;
	int x2 = box2.x, y2 = box2.y, w2 = box2.width, h2 = box2.height;

	int endx = max(x1 + w1, x2 + w2);
	int startx = min(x1, x2);
	int width = w1 + w2 - (endx - startx);  
	int endy = max(y1 + h1, y2 + h2);
	int starty = min(y1, y2);
	int height = h1 + h2 - (endy - starty);  
	if (width > 0 && height > 0) {
		int area = width * height;  
		int area1 = w1 * h1;
		int area2 = w2 * h2;
		float ratio = (float) max(area/(0.1+area1),area/ (0.1 + area2));
		return ratio;
	}
	else 
		return 0.0;

}