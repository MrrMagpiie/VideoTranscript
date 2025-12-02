import pickle
import cv2
import os
import numpy as np
import sys
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cosine

# Suppress logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from deepface import DeepFace

# --- CONFIGURATION ---
MODEL_NAME = "ArcFace"  
DETECTOR_BACKEND = "opencv" 
SAMPLES_PER_TRACK = 5   
PADDING_PCT = 0.4       
SIMILARITY_THRESHOLD = 0.40 # Cosine distance threshold for matching global ID


def get_face_crops(track, pyframes_path, samples=5):
    """
    Extracts N distinct face crops from a track.
    Returns a list of BGR images.
    """
    frames = track['track']['frame'].tolist()
    sizes = track['proc_track']['s']
    xs = track['proc_track']['x']
    ys = track['proc_track']['y']
    
    indexed_sizes = list(zip(range(len(sizes)), sizes))
    indexed_sizes.sort(key=lambda x: x[1], reverse=True)
    indices_to_sample = [x[0] for x in indexed_sizes[:samples]]
    
    crops = []
    for idx in indices_to_sample:
        frame_num = frames[idx]
        img_filename = f"{frame_num:06d}.jpg"
        img_path = os.path.join(pyframes_path, img_filename)
        
        if not os.path.exists(img_path): continue
        
        full_img = cv2.imread(img_path)
        if full_img is None: continue

        x = int(xs[idx])
        y = int(ys[idx])
        s = int(sizes[idx])
        padding = int(s * PADDING_PCT)
        
        x1 = max(0, x - s - padding)
        y1 = max(0, y - s - padding)
        x2 = min(full_img.shape[1], x + s + padding)
        y2 = min(full_img.shape[0], y + s + padding)
        
        crop = full_img[y1:y2, x1:x2]
        if crop.size > 0:
            crops.append(crop)
            
    return crops

def get_track_embedding(crops):
    embeddings = []
    for img in crops:
        try:
            result = DeepFace.represent(
                img_path=img, 
                model_name=MODEL_NAME, 
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False 
            )
            vec = result[0]["embedding"]
            embeddings.append(vec)
        except:
            continue
            
    if not embeddings:
        return None
        
    emb_array = np.array(embeddings)
    mean_vec = np.mean(emb_array, axis=0)
    norm = np.linalg.norm(mean_vec)
    if norm == 0: return None
    return mean_vec / norm


def load_registry(gloabl_registry_path):
    if os.path.exists(gloabl_registry_path):
        with open(gloabl_registry_path, 'rb') as f:
            return pickle.load(f)
    else:
        return {'faces': {}, 'next_id': 0}

def save_registry(registry,gloabl_registry_path):
    with open(gloabl_registry_path, 'wb') as f:
        pickle.dump(registry, f)

def match_to_global(local_vec, registry):
    """
    Finds the best matching Global ID for a local embedding.
    """
    best_id = None
    best_dist = 1.0
    
    for gid, global_vec in registry['faces'].items():
        dist = cosine(local_vec, global_vec)
        if dist < best_dist:
            best_dist = dist
            best_id = gid
            
    if best_id is not None and best_dist < SIMILARITY_THRESHOLD:
        return best_id, best_dist
    
    return None, best_dist

def process_reidentification(path,gloabl_registry_path):
    tracks_path = os.path.join(path,'pywork/tracks.pckl')
    pyframes_path = os.path.join(path,'pyframes')

    if not os.path.exists(tracks_path):
        print("Error: Tracks file not found.")
        sys.exit(1)

    print("Loading tracks...")
    with open(tracks_path, 'rb') as f:
        tracks = pickle.load(f)

    # 1. Load Global Registry
    registry = load_registry(gloabl_registry_path)
    print(f"Loaded Global Registry: {len(registry['faces'])} known faces.")

    # 2. Compute Embeddings for Current Chunk
    track_embeddings = {}
    print(f"Processing {len(tracks)} tracks...")
    
    for i, track in enumerate(tracks):
        crops = get_face_crops(track, pyframes_path, SAMPLES_PER_TRACK)
        if not crops: continue
        vector = get_track_embedding(crops)
        if vector is not None:
            track_embeddings[i] = vector

    # 3. Match Local Tracks to Global IDs
    replacements = {} # {Local_ID: Global_ID}
    
    for local_id, local_vec in track_embeddings.items():
        
        # Try to match existing global IDs
        global_id, dist = match_to_global(local_vec, registry)
        
        if global_id is not None:
            # MATCH FOUND
            print(f"  Track {local_id} matches Global {global_id} (dist: {dist:.3f})")
            replacements[local_id] = global_id
            
            # Update Global Average (Slowly adapt to lighting changes)
            # Weighted average: 90% old, 10% new (prevents drift)
            old_vec = registry['faces'][global_id]
            new_vec = (0.9 * old_vec) + (0.1 * local_vec)
            registry['faces'][global_id] = new_vec / np.linalg.norm(new_vec)
            
        else:
            # NEW IDENTITY
            new_global_id = registry['next_id']
            registry['next_id'] += 1
            registry['faces'][new_global_id] = local_vec
            
            print(f"  Track {local_id} is NEW -> Assigned Global {new_global_id}")
            replacements[local_id] = new_global_id

    # 4. Save Registry
    save_registry(registry,gloabl_registry_path)

    print("\n--- FINAL ID MAP (Local -> Global) ---")
    print(replacements)
    
    # This map will be passed to create_timeline to rename "0" to "5" (Global ID)
    return replacements

if __name__ == "__main__":
    path = "/var/home/magpie/Development/SerialKiller_2/output/BvWOje46Xp8/chunks/chunk_002"
    video_path = '/var/home/magpie/Development/SerialKiller_2/output/BvWOje46Xp8/global_faces.pkl'
    process_reidentification(path,video_path)