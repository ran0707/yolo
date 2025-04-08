import requests
import json

# Define API endpoint
API_URL = "http://localhost:8000/predict-folder/"

# Path to the dataset folder that contains subfolders
dataset_folder = "/home/idrone2/Desktop/set400"  # Update this path

# Prepare request payload
payload = {"dataset_folder": dataset_folder}

# Send request with correct JSON body
response = requests.post(API_URL, json=payload)

# Check response
if response.status_code == 200:
    result = response.json()
    print("✅ Batch Processing Completed Successfully")
    print("📂 Segmented Images Saved in Subfolders at:", "/home/idrone2/Desktop/Ranjith-works/yolo/predicted_outputs/")
    print("📊 Pie Chart Saved at:", result["pie_chart"])
else:
    print("❌ Error:", response.status_code, response.text)
