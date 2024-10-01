import cv2
import os
import numpy as np

from torchvision import datasets, models, transforms
import torch
from scipy.spatial.distance import cosine
       
import sys
from model import ft_net, ft_net_dense

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from test import opt,load_network,extract_feature,extract_feature_single_image

sys.path.append(r'C:\Users\123\Desktop\datasets\Person_reID_baseline_pytorch\sort')
from sort import Sort  # Import the SORT tracking class

# Load YOLOv5 model
# model_structure = ft_net(opt.nclasses, opt.droprate, opt.stride)

def load_yolov5():
    # Remove local 'utils' module if it exists in sys.modules to avoid conflict
    if 'utils' in sys.modules:
        del sys.modules['utils']

    # Load YOLOv5 model from torch.hub
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
output_video_path = r'C:/Users/123/Desktop/datasets/Person_reID_baseline_pytorch/annotated_output.mp4'

data_dir = r'C:\Users\123\Desktop\datasets\Person_reID_baseline_pytorch\Market1501\pytorch'

gallery_folder = r'C:\Users\123\Desktop\datasets\Person_reID_baseline_pytorch\Market1501\pytorch\gallery'
h, w = 256, 128
data_transforms = transforms.Compose([
    transforms.Resize((h, w), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
    
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in ['gallery']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize, shuffle=False, num_workers=4) for x in ['gallery']}

# Load YOLOv5 model
model = load_yolov5()

# Initialize SORT tracker
tracker = Sort() 
reid_model_structure = ft_net_dense(opt.nclasses, opt.droprate, opt.stride)
reid_model = load_network(reid_model_structure)
reid_model.eval()
def precompute_gallery_embeddings(gallery_folder,reid_model):
        gallery_embeddings = {}
        
        for person_name in os.listdir(gallery_folder):
            person_folder = os.path.join(gallery_folder, person_name)
            
            if os.path.isdir(person_folder):
                embeddings = []  # To store embeddings for each image of the person
                for img_file in os.listdir(person_folder):
                    img_path = os.path.join(person_folder, img_file)
                    img = cv2.imread(img_path)
                    
                    # Compute feature extraction for each gallery image
                    if img is not None:
                        embedding = extract_feature_single_image(reid_model, img, opt)  # Using single image extraction
                        embeddings.append(embedding)

                # Store the average embedding for the person (or use another method to combine them)
                if embeddings:
                    gallery_embeddings[person_name] = np.mean(embeddings, axis=0)  # Average embedding

        return gallery_embeddings
gallery_embeddings=precompute_gallery_embeddings(gallery_folder,reid_model)


def find_closest_match(query_embedding, gallery_embeddings, threshold=0.5):
        best_match_name = None
        best_distance = float('inf')
    
        for name, gallery_embedding in gallery_embeddings.items():
            # Compute similarity (using cosine distance)
            distance = cosine(query_embedding.flatten(), gallery_embedding.flatten())


            # Check if distance is below the threshold and update the best match
            if distance < best_distance and distance < threshold:
                best_match_name = name
                best_distance = distance

        return best_match_name

 
# Open the video file
cap = cv2.VideoCapture(video_path)

# Get video properties for output video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the video writer to save annotated video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for the output video
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

frame_count = 0

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

    
    for tracked in tracked_objects:
        x1, y1, x2, y2, track_id = map(int, tracked)  # Extract bounding box and ID

        # Crop the detected person from the frame
        person_crop = frame[y1:y2, x1:x2]

        # Extract features using the modified extract_feature_single_image function
        features = extract_feature_single_image(reid_model, person_crop, opt)

        # Perform comparison with gallery embeddings or other operations
        best_match_name = find_closest_match(features, gallery_embeddings,threshold=0.5)

        # Annotate the frame with the best match name if found
        if best_match_name:
            label = f'ID: {track_id}, Name: {best_match_name}'
        else:
            label = f'ID: {track_id}, Name: Unknown'
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Write the annotated frame to the output video
    out.write(frame)

    frame_count += 1

# Release the video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()

print(f'{frame_count} frames processed and annotated video saved as "{output_video_path}".')
