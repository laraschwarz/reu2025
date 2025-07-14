import cv2
import numpy as np

# Replace these with your actual points
image_points = np.array([[0, 0], [0, 1280-1], [720-1, 1280-1], [720-1, 0]], dtype=np.float32)
world_points = np.array([[0, 0], [0, 192], [108, 192], [108, 0]], dtype=np.float32)

H, status = cv2.findHomography(image_points, world_points)

# Save to XML
fs = cv2.FileStorage("trans_M_stmarc.xml", cv2.FILE_STORAGE_WRITE)
fs.write("Homography", H)
fs.release()
print("Transformation matrix saved to trans_M_stmarc.xml")


pts = np.array([[[100, 200]], [[400, 200]]], dtype=np.float32)
mapped_pts = cv2.perspectiveTransform(pts, H)
print(mapped_pts)