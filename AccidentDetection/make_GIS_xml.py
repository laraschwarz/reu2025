import cv2
import sys
import os

def main(video_path, xml_path, bg_image=None, ftpp=0.25, n_approaches=4):
    # Open video and get properties
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Use video filename as background image if not provided
    if bg_image is None:
        bg_image = os.path.splitext(os.path.basename(video_path))[0] + ".png"

    fs = cv2.FileStorage(xml_path, cv2.FILE_STORAGE_WRITE)
    fs.write('height', height)
    fs.write('width', width)
    fs.write('ftpp', ftpp)
    fs.write('bg', bg_image)
    fs.write('n_approaches', n_approaches)
    # Add more fields as needed...
    fs.release()
    print(f"XML written to {xml_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python make_xml.py <video_file> <output_xml> [bg_image] [ftpp] [n_approaches]")
    else:
        video_file = sys.argv[1]
        output_xml = sys.argv[2]
        bg_image = sys.argv[3] if len(sys.argv) > 3 else None
        ftpp = float(sys.argv[4]) if len(sys.argv) > 4 else 0.25
        n_approaches = int(sys.argv[5]) if len(sys.argv) > 5 else 4
        main(video_file, output_xml, bg_image, ftpp, n_approaches)