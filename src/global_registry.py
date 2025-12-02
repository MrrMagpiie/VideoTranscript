import os
import pickle
import numpy as np
import cv2
from scipy.spatial.distance import cosine
from deepface import DeepFace 

REGISTRY_FILE = "global_face_registry.pkl"
SIMILARITY_THRESHOLD = 0.40  # Cosine distance (Lower = Stricter match)

class GlobalFaceRegistry:
    def __init__(self, registry_path):
        self.path = registry_path
        self.known_faces = {} # Format: {Global_ID: Mean_Embedding_Vector}
        self.next_id = 0
        self.load()

    def load(self):
        if os.path.exists(self.path):
            with open(self.path, 'rb') as f:
                data = pickle.load(f)
                self.known_faces = data['faces']
                self.next_id = data['next_id']
                print(f"[Registry] Loaded {len(self.known_faces)} known identities.")
        else:
            print("[Registry] Starting new global registry.")

    def save(self):
        with open(self.path, 'wb') as f:
            pickle.dump({'faces': self.known_faces, 'next_id': self.next_id}, f)

    def get_embedding(self, img_path):
        """Extracts embedding from a single face crop."""
        try:
            # enforce_detection=False because TalkNet already detected it
            result = DeepFace.represent(img_path, model_name="ArcFace", enforce_detection=False)
            return result[0]["embedding"]
        except:
            return None

    def resolve_id(self, local_face_crops):
        """
        Takes a list of image paths for a Local Face.
        Returns the Global ID (int).
        """
        # 1. Get Average Embedding for this local track
        embeddings = []
        for img_path in local_face_crops:
            emb = self.get_embedding(img_path)
            if emb is not None:
                embeddings.append(emb)
        
        if not embeddings:
            return None # Garbage track, ignore
            
        local_vec = np.mean(embeddings, axis=0)
        
        # 2. Compare against Global Registry
        best_match_id = None
        best_dist = 1.0
        
        for gid, global_vec in self.known_faces.items():
            dist = cosine(local_vec, global_vec)
            if dist < best_dist:
                best_dist = dist
                best_match_id = gid
        
        # 3. Decision
        if best_match_id is not None and best_dist < SIMILARITY_THRESHOLD:
            # MATCH FOUND: Update the global average with new data (optional but good)
            # Simple moving average to drift with lighting changes
            self.known_faces[best_match_id] = (self.known_faces[best_match_id] + local_vec) / 2
            self.save()
            print(f"  -> Local Face matched Global ID {best_match_id} (Dist: {best_dist:.3f})")
            return best_match_id
        else:
            # NEW FACE
            new_id = self.next_id
            self.known_faces[new_id] = local_vec
            self.next_id += 1
            self.save()
            print(f"  -> New Global ID created: {new_id}")
            return new_id