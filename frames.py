# Open the video file
import cv2
import os
import numpy as np
import torch
from sort import Sort  # Import the SORT tracking class

# Load YOLOv5 model
def load_yolov5():
    model = torch.hub.load('ultralytics/yolov5:v6.2', 'yolov5s')  # Load YOLOv5 small model
    return model

# Function to detect persons in the frame
def detect_persons(frame, model):
    results = model(frame)
    detections = results.xyxy[0].numpy()  # Get detections as a NumPy array
    boxes = []
    for *box, conf, cls in detections:
        if int(cls) == 0:  # Class 0 is for 'person'
            x1, y1, x2, y2 = map(int, box)
            boxes.append((x1, y1, x2, y2, conf))  # Add confidence score
    return boxes

# Define the path to the input video file
video_path = r'C:/Users/123/Desktop/datasets/xyz/New folder/hazlah-abassi-left-cam2_LloiteMe.mp4'
output_directory = 'market1501'

os.makedirs(output_directory, exist_ok=True)

# Load YOLOv5 model
model = load_yolov5()

# Initialize SORT tracker
tracker = Sort()

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get total number of frames and FPS
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Calculate the frame range for the last 60 seconds
last_60_seconds_start_frame = max(0, total_frames - int(fps * 60))

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Check if the current frame is in the range of the first or last 60 seconds
    if frame_count < int(fps * 60) or frame_count >= last_60_seconds_start_frame:
        # Detect persons in the frame
        detections = detect_persons(frame, model)

        # Format detections for SORT (x1, y1, x2, y2, score)
        detections_for_sort = np.array(detections)
        
        # Update tracker
        tracked_objects = tracker.update(detections_for_sort)

        # Loop through tracked objects and save each detected person
        for tracked in tracked_objects:
            x1, y1, x2, y2, track_id = map(int, tracked)  # Extract bounding box and ID

            cropped_person = frame[y1:y2, x1:x2]  # Crop the image
            
            # Construct filename
            frame_filename = f'{track_id:04d}_c1s1_{frame_count:06d}.jpg'
            frame_path = os.path.join(output_directory, frame_filename)
            
            # Save the cropped image
            cv2.imwrite(frame_path, cropped_person)

    frame_count += 1

# Release the video capture object
cap.release()
cv2.destroyAllWindows()

print(f'{frame_count} frames processed and persons saved in the "{output_directory}" directory.')
