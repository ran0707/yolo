from fastapi import FastAPI
import cv2
import torch
from ultralytics import YOLO
import numpy as np
import uvicorn
from fastapi.responses import StreamingResponse

# Initialize FastAPI app
app = FastAPI()

# Load the trained YOLO model
model = YOLO("/home/idrone2/Desktop/Ranjith-works/yolo/runs/detect/train3/weights/best.pt")

# Define class names and assign unique colors for each class
class_names = ["rsc", "looper", "thrips", "jassid", "rsm", "tmb", "healthy"]
class_colors = {
    "rsc": (255, 0, 0),       # Red
    "looper": (0, 255, 0),    # Green
    "thrips": (0, 0, 255),    # Blue
    "jassid": (255, 255, 0),  # Yellow
    "rsm": (255, 0, 255),     # Magenta
    "tmb": (0, 255, 255),     # Cyan
    "healthy": (128, 0, 128)  # Purple
}

# Video streaming generator
def generate_frames():
    cap = cv2.VideoCapture(0)  # Change to the appropriate index if multiple cameras are available
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection inference
        results = model(frame)
        
        for result in results:
            for i, box in enumerate(result.boxes):
                label = class_names[int(box.cls)]  # Get class label
                confidence = float(box.conf)  # Confidence score
                bbox = [int(coord) for coord in box.xyxy[0]]  # Bounding box coordinates
                x1, y1, x2, y2 = bbox
                
                # Get unique color for each class
                color = class_colors.get(label, (255, 255, 255))  # Default to white if not found
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label_text = f"{label}: {confidence:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                (w, h), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
                cv2.rectangle(frame, (x1, y1 - h - 5), (x1 + w + 5, y1), color, -1)
                cv2.putText(frame, label_text, (x1, y1 - 5), font, font_scale, (255, 255, 255), thickness)
        
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
