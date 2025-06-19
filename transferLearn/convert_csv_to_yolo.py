import os
import pandas as pd
from PIL import Image

# === CONFIG ===
csv_path = "/home/lschwarz/Code/MIO-TCD-Localization/gt_train.csv"
image_dir = "/home/lschwarz/Code/MIO-TCD-Localization/train"
output_label_dir = "/home/lschwarz/Code/MIO-TCD-Localization/labels/train"
os.makedirs(output_label_dir, exist_ok=True)

# === Class Map (update as needed) ===
CLASS_MAP = {
    'pedestrian': 0,
    'bicycle': 1,
    'car': 2,
    'motorcycle': 3,
    'pickup_truck': 4,
    'bus': 5,
    'single_unit_truck': 6,
    'articulated_truck': 7
}



# === Read CSV without headers ===
df = pd.read_csv(csv_path, header=None)
df.columns = ['frame', 'object_type', 'xmin', 'ymin', 'xmax', 'ymax']

# === Process Each Frame ===
for frame_name, group in df.groupby('frame'):
    image_filename = f"{int(frame_name):08d}.jpg"
    image_path = os.path.join(image_dir, image_filename)

    if not os.path.exists(image_path):
        print(f"Skipping missing image: {image_filename}")
        continue

    try:
        with Image.open(image_path) as img:
            img_width, img_height = img.size
    except Exception as e:
        print(f"Error loading image {image_filename}: {e}")
        continue

    label_lines = []
    for _, row in group.iterrows():
        class_name = row['object_type'].strip()
        if class_name not in CLASS_MAP:
            continue  # skip unknown classes

        class_id = CLASS_MAP[class_name]
        xmin, xmax = float(row['xmin']), float(row['xmax'])
        ymin, ymax = float(row['ymin']), float(row['ymax'])

        # Normalize to YOLO format
        x_center = (xmin + xmax) / 2 / img_width
        y_center = (ymin + ymax) / 2 / img_height
        box_width = (xmax - xmin) / img_width
        box_height = (ymax - ymin) / img_height

        label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

    # Write label file
    label_filename = f"{int(frame_name):08d}.txt"
    label_path = os.path.join(output_label_dir, label_filename)
    with open(label_path, 'w') as f:
        f.write("\n".join(label_lines))

print("✅ Conversion complete.")

