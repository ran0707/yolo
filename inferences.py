from fastapi import FastAPI, UploadFile, File
import uvicorn
import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import io
import json

# Initialize FastAPI app
app = FastAPI()

# Load the trained YOLO model
model = YOLO("/home/idrone2/Desktop/Ranjith-works/yolo/runs/detect/train3/weights/best.pt")  # Load the trained model

# Define class names
class_names = ["rsc", "looper", "thrips", "jassid", "rsm", "tmb", "healthy"]

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        draw = ImageDraw.Draw(image)

        # Load a font with further increased size for better visibility
        try:
            font = ImageFont.truetype("arial.ttf", 60)  # Further increased font size
        except IOError:
            font = ImageFont.load_default()

        # Perform inference
        results = model(image)
        
        # Parse results
        predictions = []
        for result in results:
            for box in result.boxes:
                label = class_names[int(box.cls)]  # Get class label
                confidence = float(box.conf)  # Confidence score
                bbox = [int(coord) for coord in box.xyxy[0]]  # Bounding box coordinates
                
                predictions.append({
                    "label": label,
                    "confidence": confidence,
                    "bbox": bbox
                })
                
                # Draw bounding box with significantly increased width
                draw.rectangle(bbox, outline="red", width=10)  # Further increased box width
                
                # Draw background rectangle for text readability
                text_size = draw.textbbox((bbox[0], bbox[1] - 70), f"{label}: {confidence:.2f}", font=font)
                draw.rectangle(text_size, fill="black")
                
                # Draw text on the image
                draw.text((bbox[0], bbox[1] - 70), f"{label}: {confidence:.2f}", fill="yellow", font=font)
        
        return {"predictions": predictions}
    except Exception as e:
        return {"error": str(e)}

# Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
