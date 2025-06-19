import cv2
import os

# === CONFIG ===
image_path = "/home/lschwarz/Code/MIO-TCD-Localization/images/train/00082017.jpg"
label_path = "/home/lschwarz/Code/MIO-TCD-Localization/labels/train/00082017.txt"
class_names = ['pedestrian', 'bicycle', 'car', 'motorcycle', 'pickup_truck', 'bus', 'single_unit_truck', 'articulated_truck']

# === Load image ===
image = cv2.imread(image_path)
height, width = image.shape[:2]

# === Load labels ===
if os.path.exists(label_path):
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center, y_center, w, h = map(float, parts[1:])

            # Convert normalized coords to pixel coords
            x1 = int((x_center - w / 2) * width)
            y1 = int((y_center - h / 2) * height)
            x2 = int((x_center + w / 2) * width)
            y2 = int((y_center + h / 2) * height)

            # Draw rectangle and label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = class_names[class_id] if class_id < len(class_names) else str(class_id)
            cv2.putText(image, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# === Show image ===
cv2.imshow("Labeled Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

