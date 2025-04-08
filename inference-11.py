import os
import json
import io
import uvicorn
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import Counter
from fastapi import FastAPI
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# Initialize FastAPI app
app = FastAPI()

# Load trained YOLOv11 segmentation model
model = YOLO("/home/idrone2/Desktop/Ranjith-works/yolo/runs/segment/train3/weights/best.pt")

# Define class names and unique colors
class_names = ["rsc", "looper", "thrips", "jassid", "rsm", "tmb", "healthy"]
class_colors = {
    "rsc": (255, 0, 0),       
    "looper": (0, 255, 0),    
    "thrips": (0, 0, 255),    
    "jassid": (255, 255, 0),  
    "rsm": (255, 0, 255),     
    "tmb": (0, 255, 255),     
    "healthy": (128, 0, 128)  
}

# Define output root folder for predictions
output_root = "/home/idrone2/Desktop/Ranjith-works/yolo/predicted_outputs"
os.makedirs(output_root, exist_ok=True)

# Define input request model
class DatasetRequest(BaseModel):
    dataset_folder: str  # Expecting dataset path as a JSON field

@app.post("/predict-folder/")
async def predict_folder(request: DatasetRequest):
    dataset_folder = request.dataset_folder  # Extract dataset folder path

    if not os.path.exists(dataset_folder):
        return {"error": f"Dataset folder '{dataset_folder}' not found"}

    all_predictions = []  # Store all class predictions

    # Iterate through subfolders
    for subfolder in os.listdir(dataset_folder):
        subfolder_path = os.path.join(dataset_folder, subfolder)

        if not os.path.isdir(subfolder_path):
            continue  # Skip if not a directory

        # Create corresponding subfolder in output directory
        output_folder = os.path.join(output_root, subfolder)
        os.makedirs(output_folder, exist_ok=True)

        # Get all images in the current subfolder
        image_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for image_name in image_files:
            image_path = os.path.join(subfolder_path, image_name)

            # Read image
            image = Image.open(image_path).convert("RGB")
            image_cv = np.array(image)
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
            img_height, img_width, _ = image_cv.shape

            # Perform YOLOv11 segmentation inference
            results = model(image)

            # Process segmentation masks
            for result in results:
                if result.masks is None:
                    print(f"⚠️ No objects detected in {image_name}. Skipping...")
                    continue  # Skip this image if no detections

                for i, mask in enumerate(result.masks.xy):  
                    label = class_names[int(result.boxes.cls[i])]
                    all_predictions.append(label)  # Collect class predictions

                    # Convert polygon mask to NumPy array
                    mask_np = np.array(mask, np.int32)

                    # Get color for segmentation
                    color = class_colors.get(label, (255, 255, 255))

                    # Draw transparent segmentation mask
                    mask_overlay = np.zeros_like(image_cv, dtype=np.uint8)
                    cv2.fillPoly(mask_overlay, [mask_np], color)

                    # Blend mask with the image (50% opacity)
                    alpha = 0.5  
                    image_cv = cv2.addWeighted(image_cv, 1, mask_overlay, alpha, 0)

                    # Compute centroid of the mask for label placement
                    M = cv2.moments(mask_np)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    else:
                        cx, cy = mask_np[0][0], mask_np[0][1]  # Fallback

                    # Convert to PIL for better text rendering
                    segmented_image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(segmented_image_pil)

                    # Adjust font size based on image size
                    base_font_size = max(10, int(img_width / 20))  
                    try:
                        font = ImageFont.truetype("arial.ttf", base_font_size)
                    except IOError:
                        font = ImageFont.load_default()

                    # Draw label text with background
                    text_size = draw.textbbox((0, 0), label, font=font)
                    text_width = text_size[2] - text_size[0]
                    text_height = text_size[3] - text_size[1]

                    text_bg = [(cx - text_width // 2, cy - text_height - 10), (cx + text_width // 2 + 5, cy)]
                    draw.rectangle(text_bg, fill="black")
                    draw.text((cx - text_width // 2, cy - text_height - 10), label, font=font, fill="white")

                    # Convert back to OpenCV format
                    image_cv = cv2.cvtColor(np.array(segmented_image_pil), cv2.COLOR_RGB2BGR)

            # Save segmented output image in the correct subfolder
            segmented_image_path = os.path.join(output_folder, image_name)
            segmented_image = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
            segmented_image.save(segmented_image_path)

    # Generate and save pie chart of class-wise prediction percentages
    pie_chart_path = generate_pie_chart(all_predictions, output_root)

    return {"message": "Batch prediction completed", "pie_chart": pie_chart_path}

def generate_pie_chart(predictions, output_folder):
    """ Generate pie chart for class-wise predictions """
    class_counts = Counter(predictions)
    labels = class_counts.keys()
    sizes = class_counts.values()

    if not sizes:
        return "No objects detected in the dataset, no pie chart generated."

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=[class_colors[label] for label in labels])
    plt.title("Class-Wise Prediction Distribution")

    pie_chart_path = os.path.join(output_folder, "class_distribution_pie_chart.png")
    plt.savefig(pie_chart_path)
    plt.close()

    return pie_chart_path

# Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
