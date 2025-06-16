import os

# 🔁 Set your labels directory here
labels_dir = "/Users/laraschwarz/Code/reu2025/datasets/TUM_rgb_sample/labels"

for filename in os.listdir(labels_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(labels_dir, filename)

        with open(file_path, "r") as file:
            lines = file.readlines()

        fixed_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5 and parts[0] == "6":
                parts[0] = "7"
            fixed_lines.append(" ".join(parts) + "\n")

        with open(file_path, "w") as file:
            file.writelines(fixed_lines)

print("✅ Done: All class 6 labels changed to class 7.")
