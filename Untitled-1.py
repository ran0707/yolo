# %%
import os
import json
import cv2
import random
import shutil

# Define dataset paths
json_path = "/home/idrone2/Desktop/tea_pest.json"  # Update this
images_dir = "/home/idrone2/Desktop/new"  # Update this
output_dir = "/home/idrone2/Desktop/Ranjith-works/yolo/yolo_dataset-1"

# Create train/val image and label folders
train_images_dir = os.path.join(output_dir, "images/train")
val_images_dir = os.path.join(output_dir, "images/val")
train_labels_dir = os.path.join(output_dir, "labels/train")
val_labels_dir = os.path.join(output_dir, "labels/val")

os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# Load JSON file
with open(json_path, "r") as f:
    data = json.load(f)

# Get image filenames
image_filenames = list(data.keys())
random.shuffle(image_filenames)

# Split dataset (80% train, 20% val)
train_size = int(0.8 * len(image_filenames))
train_images = image_filenames[:train_size]
val_images = image_filenames[train_size:]

# Class mapping (Ensure correct order)
class_names = ["rsc", "looper", "thrips", "jassid", "rsm", "tmb", "healthy"]
class_mapping = {name: idx for idx, name in enumerate(class_names)}

# Function to normalize points
def normalize_points(points, img_width, img_height):
    return [(x / img_width, y / img_height) for x, y in points]

# Process each image annotation
for image_name in image_filenames:
    image_path = os.path.join(images_dir, image_name)
    img = cv2.imread(image_path)

    if img is None:
        continue

    img_height, img_width, _ = img.shape
    label_file_name = image_name.replace(".jpg", ".txt").replace(".JPG", ".txt")

    if image_name in train_images:
        shutil.copy(image_path, os.path.join(train_images_dir, image_name))
        label_file = os.path.join(train_labels_dir, label_file_name)
    else:
        shutil.copy(image_path, os.path.join(val_images_dir, image_name))
        label_file = os.path.join(val_labels_dir, label_file_name)

    with open(label_file, "w") as label_out:
        for region in data[image_name]["regions"].values():
            shape = region["shape_attributes"]
            if shape["name"] == "polygon":
                all_points_x = shape["all_points_x"]
                all_points_y = shape["all_points_y"]

                # Normalize polygon points
                normalized_polygon = normalize_points(
                    list(zip(all_points_x, all_points_y)), img_width, img_height
                )

                # Get class label
                class_label = region["region_attributes"]["label"]
                class_id = class_mapping[class_label]

                # Convert to YOLOv11 segmentation format: class_id x1 y1 x2 y2 ... xn yn
                polygon_str = " ".join([f"{x} {y}" for x, y in normalized_polygon])
                label_out.write(f"{class_id} {polygon_str}\n")

print("✅ Conversion to YOLOv11 segmentation format completed!")


# %%
from ultralytics import YOLO

# Load YOLOv11 model
model = YOLO("/home/idrone2/Desktop/Ranjith-works/yolo/yolo11s-seg.pt")  # Ensure you have the pre-trained weights

# Train the model
model.train(
    data="/home/idrone2/Desktop/Ranjith-works/yolo/yolo_dataset-1/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device="cuda"  # Use "cpu" if no GPU is available
)


# %%



