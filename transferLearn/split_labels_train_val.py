import os
import shutil


print(f"Here")

# === Paths ===
base_dir = "/home/lschwarz/Code/MIO-TCD-Localization"
image_val_dir = os.path.join(base_dir, "images/val")
label_src_dir = os.path.join(base_dir, "labels/train")
label_val_dir = os.path.join(base_dir, "labels/val")

os.makedirs(label_val_dir, exist_ok=True)

# === Find validation image basenames (without .jpg extension) ===
val_image_names = [
    os.path.splitext(f)[0]
    for f in os.listdir(image_val_dir)
    if f.lower().endswith((".jpg", ".png"))
]

# === Move matching label files to /labels/val ===
moved = 0
for name in val_image_names:
    label_file = f"{name}.txt"
    src = os.path.join(label_src_dir, label_file)
    dst = os.path.join(label_val_dir, label_file)
    if os.path.exists(src):
        shutil.move(src, dst)
        moved += 1

print(f"✅ Moved {moved} label files to 'labels/val/' to match validation images.")

