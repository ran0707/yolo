import requests
import cv2
import numpy as np
import json
from PIL import Image
import io
import matplotlib.pyplot as plt

# Define API endpoint
API_URL = "http://localhost:8000/predict/"

# Path to the test image
image_path = "/home/idrone2/Desktop/new/GOPR5989.JPG"  # Change this to your test image path

# Read and send the image to the API
with open(image_path, "rb") as image_file:
    files = {"file": image_file}
    response = requests.post(API_URL, files=files)

# Check response
if response.status_code == 200:
    result = response.json()
    print("Inference Response:", json.dumps(result, indent=4))

    # Load and display the segmented image (returned from API)
    segmented_image_path = result.get("image", "")
    
    if segmented_image_path:
        segmented_image = Image.open(segmented_image_path).convert("RGB")
        
        # Save segmented image
        output_path = "segmented_output-1.jpg"
        segmented_image.save(output_path)
        print(f"Segmented image saved as {output_path}")

        # Show image using Matplotlib
        plt.figure(figsize=(8, 6))
        plt.imshow(segmented_image)
        plt.axis("off")
        plt.show()
    else:
        print("No segmented image returned in response.")

else:
    print(f"Error: {response.status_code}, Message: {response.text}")
