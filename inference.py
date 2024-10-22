import os
import sys
import cv2
import torch
import random

import numpy as np
from scipy.spatial.distance import cosine
from torchvision import datasets, transforms
import warnings

seed = 42  # You can choose any seed value
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)





# Suppress specific FutureWarnings 
warnings.filterwarnings("ignore", category=FutureWarning)


# Import custom modules
from test import opt, load_network, extract_feature_single_image
from model import ft_net_dense
sys.path.append(r'C:\Users\123\Desktop\datasets\Person_reID_baseline_pytorch\sort')
from sort import Sort  # Import the SORT tracking class

# Global paths
video_path = r'C:/Users/123/Desktop/datasets/xyz/New folder/hazlah-abassi-left-cam2_LloiteMe.mp4'
output_video_path = r'C:/Users/123/Desktop/datasets/Person_reID_baseline_pytorch/annotated_output.mp4'
gallery_folder = r'C:\Users\123\Desktop\datasets\Person_reID_baseline_pytorch\Market1501\pytorch\gallery'
embeddings_file = r'C:\Users\123\Desktop\datasets\Person_reID_baseline_pytorch\gallery_embeddings.pth'  # Single .pth file for embeddings
names_file = r'C:\Users\123\Desktop\datasets\Person_reID_baseline_pytorch\names.npy'  # Separate .npy file for names

# Add flag for updating gallery embeddings
update_gallery = False

def load_yolov5():
    """Loads YOLOv5 model from torch hub."""
    if 'utils' in sys.modules:
        del sys.modules['utils']

    model = torch.hub.load('ultralytics/yolov5:v6.2', 'yolov5s')  # Load YOLOv5 small model
    return model

def detect_persons(frame, model):
    """Detects persons in a given frame using the provided YOLOv5 model."""
    results = model(frame)
    detections = results.xyxy[0].numpy()  # Get detections as a NumPy array
    boxes = []
    for *box, conf, cls in detections:
        if int(cls) == 0:  # Class 0 is for 'person'
            x1, y1, x2, y2 = map(int, box)
            boxes.append((x1, y1, x2, y2, conf))  # Add confidence score
    return boxes

def precompute_gallery_embeddings(gallery_folder, reid_model):
    """Precomputes and returns embeddings for the gallery images and saves them to a .pth file."""
    gallery_embeddings = {}
    for person_name in os.listdir(gallery_folder):
        person_folder = os.path.join(gallery_folder, person_name)
        if os.path.isdir(person_folder):
            embeddings = []  # To store embeddings for each image of the person
            for img_file in os.listdir(person_folder):
                img_path = os.path.join(person_folder, img_file)
                img = cv2.imread(img_path)
                if img is not None:
                    embedding = extract_feature_single_image(reid_model, img, opt)  # Using single image extraction
                    embeddings.append(embedding)

            # Store the average embedding for the person (or use another method to combine them)
            if embeddings:
                gallery_embeddings[person_name] = np.mean(embeddings, axis=0)  # Average embedding

    # Check names and embeddings before saving
    names_before_saving = list(gallery_embeddings.keys())
    embeddings_before_saving = {name: gallery_embeddings[name] for name in names_before_saving}

    # Save all computed embeddings to a single .pth file
    torch.save(gallery_embeddings, embeddings_file) 

    # Save names to a separate .npy file
    np.save(names_file, names_before_saving)

    print("Gallery embeddings have been precomputed and saved.")

    return gallery_embeddings
 
# def load_gallery_embeddings():
#     """Loads precomputed gallery embeddings from the .pth file. If none are found, recomputes them."""
#     if os.path.exists(embeddings_file):
#         gallery_embeddings = torch.load(embeddings_file)
#         print(f"Loaded {len(gallery_embeddings)} gallery embeddings from {embeddings_file}.")
#         return gallery_embeddings
#     else:
#         print("No precomputed embeddings found. Recomputing gallery embeddings.")
#         return None

def verify_embeddings_and_names(names_before, embeddings_before, loaded_gallery_embeddings):
    """Compares names and embeddings before saving and after loading, and prints the results."""
    loaded_names = sorted(loaded_gallery_embeddings.keys())

    # Check if names match
    if sorted(names_before) != loaded_names:
        print("Names do not match between before saving and after loading.")
    else:
        print("Names match between before saving and after loading.")

    # Check if embeddings match
    for name in names_before:
        if not np.array_equal(embeddings_before[name], loaded_gallery_embeddings[name]):
            print(f"Embeddings for {name} do not match.")
        else:
            print(f"Embeddings for {name} match.")
def l2_normalize(array):
    norm = np.linalg.norm(array, ord=2)
    return array / norm if norm > 0 else array

# Normalize qemb and gemb


def find_closest_match(query_embedding, gallery_embeddings, threshold=0.5):
    """Finds and returns the closest match name for a given query embedding."""
    best_match_name = None
    best_distance = float('inf')

    # Flatten and truncate the query embedding to the first 50 elements
    qemb = query_embedding.flatten()
    qemb_normalized = l2_normalize(qemb)

    for name, gallery_embedding in gallery_embeddings.items():
        # Flatten and truncate the gallery embedding to the first 50 elements
        gemb = gallery_embedding.flatten()
        gemb_normalized = l2_normalize(gemb)


        # Print embeddings for the current iteration
        # print(f"Processing {name}:")
        # print(f"Query embedding (first 50 elements): {qemb}")
        # print(f"Gallery embedding for {name} (first 50 elements): {gemb}")
        # print(f"Gallery embedding without name specified (first 50 elements): {gemb}")

        print()  # Empty line for better readability

        # Calculate distance using the embeddings
# Check shape
        print("qemb shape:", qemb.shape)
        print("gemb shape:", gemb.shape)

        # Check type
        print("qemb type:", type(qemb))
        print("gemb type:", type(gemb))

        # Check size
        print("qemb size:", qemb.size)  # Use .size for NumPy arrays
        print("gemb size:", gemb.size)   # Use .size for NumPy arrays
        distance = cosine(qemb_normalized, gemb_normalized)

        # Check for the best match within the threshold
        if distance < best_distance and distance < threshold:
            best_match_name = name
            best_distance = distance
            print(f"Distance between {best_match_name} and query: {distance}")


    return best_match_name

def process_video(video_path, output_video_path, model, reid_model, gallery_embeddings):
    """Processes the video frame-by-frame, performs detection, tracking, and annotation."""
    cap = cv2.VideoCapture(video_path)

    # Get video properties for output video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames_to_process = 30 * fps

    # Define the video writer to save annotated video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for the output video
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Initialize the SORT tracker
    tracker = Sort()
    frame_count = 0
    tracked_embeddings = {}

    while frame_count < frames_to_process:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_persons(frame, model)
        detections_for_sort = np.array(detections)
        tracked_objects = tracker.update(detections_for_sort)

        for tracked in tracked_objects:
            x1, y1, x2, y2, track_id = map(int, tracked)

            person_crop = frame[y1:y2, x1:x2]
            if track_id not in tracked_embeddings:
                features = extract_feature_single_image(reid_model, person_crop, opt)
                tracked_embeddings[track_id] = features
            else:
                features = tracked_embeddings[track_id]
            # features = torch.nn.functional.normalize(features, p=2, dim=0)


            best_match_name = find_closest_match(features, gallery_embeddings, threshold=0.5)
            label = best_match_name if best_match_name else "Unknown"

            text_color = (255, 255, 255)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 1)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f'{frame_count} frames processed and annotated video saved as "{output_video_path}".')

def load_gallery_embeddings():
    """Loads precomputed gallery embeddings from the .pth file. If none are found, recomputes them."""
    if os.path.exists(embeddings_file):
        gallery_embeddings = torch.load(embeddings_file)
        print(f"Loaded {len(gallery_embeddings)} gallery embeddings from {embeddings_file}.")
        return gallery_embeddings  # Load names if embeddings exist
    else:
        print("No precomputed embeddings found. Recomputing gallery embeddings.")
        return None # Return None for both if no embeddings are found
def main():
    model = load_yolov5()
    reid_model_structure = ft_net_dense(opt.nclasses, opt.droprate, opt.stride)
    reid_model = load_network(reid_model_structure)
    reid_model.eval()

  

    # Load or precompute gallery embeddings based on the flag
    loaded_gallery_embeddings = load_gallery_embeddings()

    if loaded_gallery_embeddings is None:
        # No precomputed embeddings found; recompute and save them
        gallery_embeddings = precompute_gallery_embeddings(gallery_folder, reid_model)
    else:     
        # If embeddings are loaded, use loaded names and create a dictionary for embeddings
        
        gallery_embeddings = loaded_gallery_embeddings  # Assign loaded embeddings to gallery_embeddings

    # Verify if we have any valid names and embeddings
    # if (names_before_saving and  
    #     len(embeddings_before_saving) > 0):
    #     verify_embeddings_and_names(names_before_saving, embeddings_before_saving, loaded_ gallery_embeddings)

    # Ensure gallery_embeddings is not None before proceeding
    if gallery_embeddings is not None:
        process_video(video_path, output_video_path, model, reid_model, gallery_embeddings)
    else:
        print("Gallery embeddings could not be loaded or computed. Exiting.")
       


if __name__ == '__main__':
    main()
