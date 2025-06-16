import os

label_dir = "/Users/laraschwarz/Code/reu2025/datasets/TUM_rgb_sample/labels"

for filename in os.listdir(label_dir):
    if filename.endswith(".txt"):
        path = os.path.join(label_dir, filename)
        with open(path, "r") as f:
            lines = f.readlines()

        cleaned = [line for line in lines if not line.startswith("-1")]

        with open(path, "w") as f:
            f.writelines(cleaned)
