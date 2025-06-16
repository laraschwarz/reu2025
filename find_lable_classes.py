import os
import json

json_folder = '/Users/laraschwarz/Code/reu2025/datasets/TUMTraf_Event_Dataset/val/OPENLabel_labels_rgb'  # 🔁 Replace this with your actual folder
class_names = set()

for filename in os.listdir(json_folder):
    if filename.endswith('.json'):
        with open(os.path.join(json_folder, filename), 'r') as f:
            try:
                data = json.load(f)
                frames = data.get("openlabel", {}).get("frames", {})
                for frame in frames.values():
                    objects = frame.get("objects", {})
                    for obj in objects.values():
                        obj_type = obj.get("object_data", {}).get("type")
                        if obj_type:
                            class_names.add(obj_type)
            except Exception as e:
                print(f"Error reading {filename}: {e}")

print("Found classes:", sorted(class_names))


# import os

# label_dir = '/Users/laraschwarz/Code/reu2025/datasets/TUM_rgb_sample/labels'
# classes = set()

# for file in os.listdir(label_dir):
#     if file.endswith('.txt'):
#         with open(os.path.join(label_dir, file)) as f:
#             for line in f:
#                 parts = line.strip().split()
#                 if parts:
#                     classes.add(int(parts[0]))

# print("Class IDs found:", sorted(classes))
# print("Number of classes:", max(classes)+1)
