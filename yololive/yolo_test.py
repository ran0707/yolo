import cv2
import requests
import numpy as np

# URL of the FastAPI video streaming endpoint
VIDEO_URL = "http://localhost:8000/video_feed"

# Open the video stream
stream = requests.get(VIDEO_URL, stream=True)

# Check if the request was successful
if stream.status_code == 200:
    byte_stream = b""
    for chunk in stream.iter_content(chunk_size=1024):
        byte_stream += chunk
        a = byte_stream.find(b'\xff\xd8')  # JPEG start marker
        b = byte_stream.find(b'\xff\xd9')  # JPEG end marker
        if a != -1 and b != -1:
            jpg = byte_stream[a:b+2]
            byte_stream = byte_stream[b+2:]
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                cv2.imshow("YOLO Real-Time Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
else:
    print("Failed to connect to the video feed")

cv2.destroyAllWindows()
