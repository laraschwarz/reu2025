import os
import random
import shutil

# Set your base path here
base_dir = "/home/lschwarz/Code/MIO-TCD-Localization"
original_train_dir = os.path.join(base_dir, "train")
output_train_dir = os.path.join(base_dir, "images/train")
output_val_dir = os.path.join(base_dir, "images/val")

# Create output directories if they don't exist
os.makedirs(output_train_dir, exist_ok=True)
os.makedirs(output_val_dir, exist_ok=True)

# List all image files in the original training directory
image_extensions = (".jpg", ".jpeg", ".png")
all_images = [f for f in os.listdir(original_train_dir) if f.lower().endswith(image_extensions)]

# Shuffle and split
random.seed(42)  # for reproducibility
random.shuffle(all_images)
val_count = int(0.1 * len(all_images))
val_images = set(all_images[:val_count])

# Move files
for img in all_images:
    src_path = os.path.join(original_train_dir, img)
    dest_dir = output_val_dir if img in val_images else output_train_dir
    shutil.copy2(src_path, os.path.join(dest_dir, img))

print(f"Total images: {len(all_images)}")
print(f"Moved {len(val_images)} images to validation set.")
print(f"Remaining {len(all_images) - len(val_images)} images in training set.")

