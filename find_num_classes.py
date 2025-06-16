import os

label_folder = '/Users/laraschwarz/Code/reu2025/datasets/TUM_rgb_sample/lables'
class_ids = set()

for file in os.listdir(label_folder):
    if file.endswith('.txt'):
        with open(os.path.join(label_folder, file), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_ids.add(int(parts[0]))

print(f"Number of classes: {max(class_ids)+1}")
print(f"Classes found: {sorted(class_ids)}")
