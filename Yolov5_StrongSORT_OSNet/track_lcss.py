import argparse
import collections
from shapely.geometry import Point, Polygon
import os
from tslearn.metrics import lcss
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, print_args, check_file)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT

# remove duplicated stream handler to avoid duplicated logging
logging.getLogger().removeHandler(logging.getLogger().handlers[0])

@torch.no_grad()
def run(
        source='0',
        yolo_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
        strong_sort_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        config_strongsort=ROOT / 'strong_sort/configs/strong_sort.yaml',
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_vid=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = yolo_weights.stem
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(yolo_weights[0]).stem
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + strong_sort_weights.stem
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_weights, device=device, dnn=dnn, data=None, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # initialize StrongSORT
    cfg = get_config()
    cfg.merge_from_file(opt.config_strongsort)

    # Create as many strong sort instances as there are video sources
    strongsort_list = []
    for i in range(nr_sources):
        strongsort_list.append(
            StrongSORT(
                strong_sort_weights,
                device,
                half,
                max_dist=cfg.STRONGSORT.MAX_DIST,
                max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                max_age=cfg.STRONGSORT.MAX_AGE,
                n_init=cfg.STRONGSORT.N_INIT,
                nn_budget=cfg.STRONGSORT.NN_BUDGET,
                mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                ema_alpha=cfg.STRONGSORT.EMA_ALPHA,

            )
        )
        strongsort_list[i].model.warmup()
    outputs = [None] * nr_sources

    # Run tracking
    model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0

    # Added for tracking first and last coordinates of each track
    coords = {}      # id → list of coordinates
    previous_ids = set()        # ids seen in the prior frame
    # west_box = Polygon([(376, 1224), (0, 945), (0, 304), (850, 683)]) #car lane only box
    # orig_west = [(376, 1224), (0, 945), (0, 55), (1020, 479)]
    # orig_east = [(1911, 845), (1533, 1450), (2561, 1450), (2561, 1210)]
    # cap = cv2.VideoCapture('trimmed_stmarc.mp4')
    # if not cap.isOpened():
    #     print(f"Error: could not open {'trimmed_stmarc.mp4'}")
    #     return
    # ret, frame = cap.read()
    # h, w = frame.shape[:2]

    # # If polygons were defined on a different resolution,
    # scale_x = w  / 2561
    # scale_y = h  / 1450

    # west_points = [(int(x * scale_x), int(y * scale_y)) for x, y in orig_west]
    # east_points = [(int(x * scale_x), int(y * scale_y)) for x, y in orig_east]

    # west_box = Polygon(west_points)
    # east_box = Polygon(east_points)

    west_box = Polygon([(376, 1224), (0, 945), (0, 55), (1020, 479)])
    east_box = Polygon([(1911, 845), (1533, 1431), (2561, 1450), (2561, 1210)])
    north_box = Polygon([(1050, 460), (1443, 0), (2400, 0), (1900, 845)])
    south_box = Polygon([(707, 1440), (355, 1440), (259, 1370), (382, 1234)])

    # names = model.module.names if hasattr(model, 'module') else model.names
    names = model.names
    id_to_class = {}      # will map each track ID to its class label

    # count how many objects crossed from west to east
    count = 0
    we = [] # west to east
    ne = [] # north to east
    ns = [] # north to south

    cross_display = [] # messages to display on video

    # base path for objects that cross from west to east, based on car ID:10
    we_path = [(39.5, 229.0), (45.5, 232.0), (52.0, 236.5), (59.5, 242.5), (66.5, 250.0), (73.5, 256.0), (83.5, 264.5), (98.0, 271.5), (111.5, 280.5), (126.5, 287.5), (140.0, 293.5), (155.0, 301.5), (170.0, 308.0), (183.5, 316.0), (200.0, 324.5), (213.5, 332.5), (230.5, 340.0), (246.0, 349.5), (262.0, 357.0), (277.0, 364.5), (293.0, 372.5), (307.0, 381.0), (325.0, 390.0), (342.0, 399.0), (358.0, 407.5), (375.5, 416.0), (392.0, 425.0), (408.5, 432.0), (426.5, 441.5), (441.5, 450.5), (460.5, 459.0), (478.0, 467.5), (496.0, 476.5), (513.5, 486.5), (532.0, 495.0), (550.0, 505.0), (568.0, 515.0), (587.5, 523.5), (606.0, 535.0), (625.5, 544.5), (644.5, 555.0), (664.5, 565.5), (684.0, 575.0), (704.0, 586.0), (724.5, 596.0), (745.5, 608.5), (766.0, 619.0), (787.0, 629.5), (808.5, 641.0), (831.5, 646.5), (853.0, 651.0), (874.5, 656.0), (897.5, 663.0), (918.5, 669.0), (923.5, 675.0), (948.0, 680.5), (962.0, 686.5), (970.5, 692.5), (986.5, 698.0), (1002.0, 705.0), (1019.5, 711.5)]
    ns_path = [(981.5, 30.0), (976.5, 37.0), (973.0, 43.5), (963.0, 51.0), (955.0, 57.0), (937.5, 71.5), (927.5, 84.5), (917.0, 99.5), (899.0, 127.5), (888.5, 142.5), (878.0, 155.0), (858.0, 184.5), (836.0, 214.5), (816.0, 244.5), (806.0, 259.5), (782.0, 290.5), (771.0, 307.5), (758.5, 323.5), (700.0, 406.5), (662.5, 459.5), (637.0, 496.0), (625.0, 513.5), (596.5, 551.5), (583.5, 570.5), (569.5, 590.5), (555.5, 610.5), (540.5, 623.0), (527.5, 633.0), (515.0, 642.0), (508.0, 652.0), (498.5, 662.5), (491.5, 672.0), (484.5, 683.0), (476.0, 693.0), (465.5, 703.5)]

    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # nr_sources >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f'{i}: '
                txt_file_name = p.name
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
            curr_frames[i] = im0

            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print image dimensions
            imc = im0.copy() if save_crop else im0  # for save_crop

            annotator = Annotator(im0, line_width=2, pil=not ascii)
            if cfg.STRONGSORT.ECC:  # camera motion compensation
                strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add object count and name to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to strongsort
                t4 = time_sync()
                outputs[i] = strongsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4

                # draw boxes for visualization
                if len(outputs[i]) > 0:
                    for j, (output, conf) in enumerate(zip(outputs[i], confs)):
    
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]


                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                        if save_vid or save_crop or show_vid:  # Add bbox to image
                            c = int(cls)  # integer class
                            id = int(id)  # integer id
                            cx = (output[0] + output[2]) / 2
                            cy = (output[1] + output[3]) / 2

                            if(id==10):
                                we_path.append((cx, cy))
                            if(id==12):
                                ns_path.append((cx, cy))

                            label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {cx:.2f} {cy:.2f} {conf:.2f}'))
                            annotator.box_label(bboxes, label, color=colors(c, True))
                            if save_crop:
                                txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)

                
                if (frame_idx + 1) % 10 == 0:
                    LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), StrongSORT:({t5 - t4:.3f}s)') # print s and timing info every 10 frames
                


            else:
                strongsort_list[i].increment_ages()
                LOGGER.info('No detections')

            font        = cv2.FONT_HERSHEY_SIMPLEX
            font_scale  = 0.5
            thickness   = 1
            padding     = 2    # px of padding around the text
            bg_color    = (0, 0, 0)
            text_color  = (0, 0, 255)
            for j, text in enumerate(cross_display):
                (w, h), base = cv2.getTextSize(text, font, font_scale, thickness)
                x, y = 10, 30 + j * 30
                tl = (x - padding, y - h - base - padding)
                br = (x + w + padding, y + base + padding)
                cv2.rectangle(im0, tl, br, (255, 255, 255), cv2.FILLED)
                cv2.putText(
                    im0,                  # draw onto this image
                    text,                 # the string to draw
                    (x, y),              # bottom‐left corner of text (x, y)
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,           # font scale
                    text_color,          # color = red (B, G, R)
                    thickness,            # line thickness
                    cv2.LINE_AA
                )
            # Stream results
            im0 = annotator.result()
            if show_vid:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_vid:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

            prev_frames[i] = curr_frames[i]

            # ——— track centers each frame ———
            current_ids = set()

            # If we have any detections for this source:
            if outputs[i] is not None and len(outputs[i]) > 0:
                for out in outputs[i]:
                    tid = int(out[4])
                    cls_index=int(out[5])
                    cls_name= names[cls_index] if cls_index < len(names) else 'unknown'
                    id_to_class[tid] = cls_name  # map track ID to class label
                    # compute center
                    cx = (out[0] + out[2]) / 2
                    cy = (out[1] + out[3]) / 2

                    # init per‐ID storage
                    if tid not in coords:
                        coords[tid] = []

                    coords[tid].append((cx.item(), cy.item()))
                    current_ids.add(tid)

            # detect IDs that just disappeared
            lost = previous_ids - current_ids
            previous_ids = current_ids

            # count how many objects crossed from west to east
            
            for tid, points in coords.items():
                if tid in lost and len(points) > 10:  # if the track has disappeared
                    path_taken = "W-->E" if lcss(we_path, points, eps=10.0) > 0.5 else "N-->S" if lcss(ns_path, points, eps=10.0) > 0.5 else "unknown"
                    if not path_taken == "unknown":
                        count += 1
                        we.append(tid) if path_taken == "W-->E" else ns.append(tid) if path_taken == "N-->S" else None
                        msg = f"{id_to_class[tid]} ({tid}) crossed {path_taken}"
                        cross_display.append(msg) # message to display on video
                        print(str(tid) + ' (' + id_to_class.get(tid, 'unknown') + ') crossed from {path_taken}\n\n')
                    if id_to_class[tid] == 'car':
                        print(f"lcss_we: {lcss(we_path, points, eps=10.0)}, lcss_ns: {lcss(ns_path, points, eps=10.0)}, points: {points}\n\n")

                

    # ——— report ———

    print(f"{len(we)} objects crossed from west to east")
    print(f"{len(ne)} objects turned from north to east")
    print(f"{len(ns)} objects crossed from north to south")
    print(f"Tracked IDs (west to east): {[str(tid) + ' (' + id_to_class.get(tid, 'unknown') + ')' for tid in we]}")
    print(f"Tracked IDs (north to east): {[str(tid) + ' (' + id_to_class.get(tid, 'unknown') + ')' for tid in ne]}")
    print(f"Tracked IDs (north to south): {[str(tid) + ' (' + id_to_class.get(tid, 'unknown') + ')' for tid in ns]}")

    # print(f"we_path: {we_path}")
    # pure = [(float(a), float(b)) for a, b in ns_path]
    # print(f"ns_path: {pure}")

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms strong sort update per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=str, default=WEIGHTS / 'yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--strong-sort-weights', type=str, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--config-strongsort', type=str, default='strong_sort/configs/strong_sort.yaml')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)