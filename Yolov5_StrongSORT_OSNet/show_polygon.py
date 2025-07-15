# import argparse
# import cv2
# import numpy as np
# from shapely.geometry import Polygon

# def polygon_to_int_pts(poly: Polygon):
#     # Convert shapely polygon exterior coords to a NumPy array of int points
#     pts = np.array(poly.exterior.coords, dtype=np.int32)
#     return pts.reshape((-1, 1, 2))

# def main():
#     parser = argparse.ArgumentParser(
#         description="Overlay two polygons on a video and display it frame by frame.")
#     parser.add_argument("video_path", help="Path to input video file (e.g. video.mp4)")
#     args = parser.parse_args()

#     # 1. Define your polygons (in your original image coordinate space)
#     west_box = Polygon([(376, 1224), (0, 945), (0, 55), (1020, 479)])
#     east_box = Polygon([(1911, 845), (1533, 1450), (2561, 1450), (2561, 1210)])

#     # Precompute integer point arrays for OpenCV
#     west_pts = polygon_to_int_pts(west_box)
#     east_pts = polygon_to_int_pts(east_box)

#     # 2. Open the video
#     cap = cv2.VideoCapture(args.video_path)
#     if not cap.isOpened():
#         print(f"Error: could not open {args.video_path}")
#         return

#     # Read the first frame to get its size
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: empty video or cannot read frames.")
#         return

#     h, w = frame.shape[:2]

#     # If your polygons were defined on a different resolution,
#     # compute a scaling factor here.  For example, if they were
#     # drawn on a 2561×1450 canvas but your video is a different
#     # size, you’d do:
#     #
#     scale_x = w  / 2561
#     scale_y = h  / 1450
#     west_pts = (west_pts * [scale_x, scale_y]).astype(np.int32)
#     east_pts = (east_pts * [scale_x, scale_y]).astype(np.int32)
#     #
#     # If your video is already the same size you originally annotated,
#     # you can skip scaling.

#     alpha = 0.3             # polygon fill transparency
#     fill_color = (0, 255, 0)  # BGR fill color (green)
#     border_color = (0, 0, 255)  # BGR border color (red)
#     border_thickness = 2

#     # 3. Process every frame
#     while True:
#         # If this is the first frame, we already have it in `frame`;
#         # otherwise, read next
#         if 'first' not in locals():
#             first = True
#         else:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#         # Make overlay
#         overlay = frame.copy()
#         cv2.fillPoly(overlay, [west_pts], fill_color)
#         cv2.fillPoly(overlay, [east_pts], fill_color)

#         # Blend overlay onto frame
#         cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

#         # Draw borders on top
#         cv2.polylines(frame, [west_pts], isClosed=True, color=border_color, thickness=border_thickness)
#         cv2.polylines(frame, [east_pts], isClosed=True, color=border_color, thickness=border_thickness)

#         # Display
#         cv2.imshow("Video with Turn-Counting Zones", frame)
#         key = cv2.waitKey(30) & 0xFF
#         if key == 27 or key == ord('q'):  # ESC or 'q' to quit
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()


from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 1. Load your background image
img = mpimg.imread('stmarc_cover.png')

# 2. Define your polygon in pixel coordinates
west_box = Polygon([(376, 1224), (0, 945), (0, 55), (1020, 479)])
east_box = Polygon([(1911, 845), (1533, 1431), (2561, 1450), (2561, 1210)])
north_box = Polygon([(1050, 460), (1443, 0), (2400, 0), (1900, 845)])
south_box = Polygon([(707, 1440), (355, 1440), (259, 1370), (382, 1234)])
x1, y1 = west_box.exterior.xy
x2, y2 = east_box.exterior.xy
x3, y3 = north_box.exterior.xy
x4, y4 = south_box.exterior.xy

# 3. Plot the image and then overlay the polygon
fig, ax = plt.subplots()
ax.imshow(img)                      # draw image
ax.plot(x1, y1, linewidth=2)          # polygon border (default color=blue)
ax.fill(x1, y1, alpha=0.3)            # translucent fill
ax.plot(x2, y2, linewidth=2)          # polygon border (default color=blue)
ax.fill(x2, y2, alpha=0.3)   
ax.plot(x3, y3, linewidth=2)          
ax.fill(x3, y3, alpha=0.3)  
ax.plot(x4, y4, linewidth=2)
ax.fill(x4, y4, alpha=0.3)
ax.set_xlim(0, img.shape[1])        # match axes to image size
ax.set_ylim(img.shape[0], 0)        # flip Y so (0,0) is top-left
ax.set_axis_off()                   # optional: hide axes
plt.show()
