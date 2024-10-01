import cv2
import os
import numpy as np
import torch
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(r'C:\Users\123\Desktop\datasets\Person_reID_baseline_pytorch\sort')
sys.path.append(r'C:\Users\123\Desktop\datasets\Person_reID_baseline_pytorch\yolov5')
from sort import Sort  # Import the SORT tracking class
import yolov5
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
output_directory = r'C:/Users/123/Desktop/Recyle Bin'
os.makedirs(output_directory, exist_ok=True)
# Load YOLOv5 model
model = load_yolov5()
# Initialize SORT tracker
tracker = Sort()
# Open the video file
cap = cv2.VideoCapture(video_path)
frame_count = 0
image_counter = {}  # Dictionary to count images per tracked ID
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Detect persons in the frame
    detections = detect_persons(frame, model)
    # Format detections for SORT (x1, y1, x2, y2, score)
    detections_for_sort = np.array(detections)
    
    # Update tracker
    tracked_objects = tracker.update(detections_for_sort)
    # Loop through tracked objects and save each detected person
    for tracked in tracked_objects:
        x1, y1, x2, y2, track_id = map(int, tracked)  # Extract bounding box and ID
        # Increment the counter for this track ID
        if track_id not in image_counter:
            image_counter[track_id] = 1
        else:
            image_counter[track_id] += 1
        
        cropped_person = frame[y1:y2, x1:x2]  # Crop the image
        
        # Construct filename with varying instance numbers
        instance_number = f'{image_counter[track_id]:02d}'  # Get instance number with leading zero
        frame_filename = f'{track_id:04d}_c1s1_{frame_count:06d}_{instance_number}.jpg'
        frame_path = os.path.join(output_directory, frame_filename)
        
        # Save the cropped image
        cv2.imwrite(frame_path, cropped_person)
    frame_count += 1
# Release the video capture object
cap.release()
cv2.destroyAllWindows()
print(f'{frame_count} frames processed and persons saved in the "{output_directory}" directory.')