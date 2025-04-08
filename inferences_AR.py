import os
import json
import io
import uvicorn
import torch
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File
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

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_cv = np.array(image)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
        img_height, img_width, _ = image_cv.shape

        # Create overlay image for transparency
        overlay = image_cv.copy()

        # Perform YOLOv11 segmentation inference
        results = model(image)

        # Process segmentation masks
        for result in results:
            for i, mask in enumerate(result.masks.xy):  
                label = class_names[int(result.boxes.cls[i])]
                confidence = float(result.boxes.conf[i])

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

                # Ensure text is inside the mask by finding the lowest point in the mask
                mask_points = mask_np.reshape(-1, 2)
                min_y_idx = np.argmin(mask_points[:, 1])
                cx, cy = mask_points[min_y_idx]

                # Convert to PIL for better text rendering
                segmented_image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(segmented_image_pil)

                # Dynamically adjust font size based on image size
                base_font_size = max(10, int(img_width / 20))  # Adjusted for readability

                # Load font (fallback if unavailable)
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

        # Save output image
        segmented_image = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        output_path = "segmented_output.jpg"
        segmented_image.save(output_path)

        return {"predictions": class_names, "image": output_path}
    except Exception as e:
        return {"error": str(e)}

# Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
