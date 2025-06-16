import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os

# CONFIG
json_path = '/Users/laraschwarz/Downloads/drone-mscoco.json'      # Path to your large COCO JSON file
image_folder = '/Users/laraschwarz/Downloads'                 # Folder containing the actual images
image_id_to_view = 0                          # Change this to any image ID you'd like to visualize

# Load only the necessary parts of the file
with open(json_path, 'r') as f:
    data = json.load(f)

# Build index for fast lookup
images = {img['id']: img for img in data['images']}
annotations_by_image = {}
for ann in data['annotations']:
    img_id = ann['image_id']
    if img_id not in annotations_by_image:
        annotations_by_image[img_id] = []
    annotations_by_image[img_id].append(ann)

# Get image info
if image_id_to_view not in images:
    raise ValueError(f"Image ID {image_id_to_view} not found.")

img_info = images[image_id_to_view]
img_path = os.path.join(image_folder, img_info['file_name'])
img = Image.open(img_path)

# Plot
fig, ax = plt.subplots()
ax.imshow(img)

for ann in annotations_by_image.get(image_id_to_view, []):
    x, y, w, h = ann['bbox']
    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)

plt.title(f"Image ID: {image_id_to_view}")
plt.axis('off')
plt.tight_layout()
plt.show()
