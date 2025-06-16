import os

# your ID to COCO ID mapping
id_map = {
    0: 5,   # BUS → bus
    2: 2,   # CAR → car
    4: 7,   # TRAILER → truck
    5: 7    # TRUCK → truck
}

input_dir = "/Users/laraschwarz/Code/reu2025/datasets/TUMTraf_Event_Dataset/val/yolo_labels_rgb"
output_dir = "/Users/laraschwarz/Code/reu2025/datasets/TUM_rgb_sample/new_lables"
os.makedirs(output_dir, exist_ok=True)

# for file in os.listdir(input_dir):
#     if file.endswith(".txt"):
#         with open(os.path.join(input_dir, file)) as f_in, open(os.path.join(output_dir, file), "w") as f_out:
#             for line in f_in:
#                 parts = line.strip().split()
#                 old_id = int(parts[0])
#                 if old_id in id_map:
#                     parts[0] = str(id_map[old_id])
#                     f_out.write(" ".join(parts) + "\n")

classes_to_keep = {0, 2, 3, 5}  # Example: only COCO-aligned classes

for file in os.listdir(input_dir):
    if file.endswith(".txt"):
        with open(os.path.join(input_dir, file)) as f_in, open(os.path.join(output_dir, file), "w") as f_out:
            for line in f_in:
                class_id = int(line.strip().split()[0])
                if class_id in classes_to_keep:
                    f_out.write(line)
