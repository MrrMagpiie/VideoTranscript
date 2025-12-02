import torch
import json
import numpy as np
import pandas as pd
from pyannote.audio import Model, Inference
from pyannote.core import Segment
from sklearn.neighbors import KNeighborsClassifier
import os
import pickle # <--- For saving memory

VISUAL_CONFIDENCE_THRESHOLD = 0.70
VIDEO_FPS = 25.0
MANUAL_ANCHORS = []
TARGET_DURATION = 0.6
MAX_GAP = 0.3
MODEL_ID = "pyannote/wespeaker-voxceleb-resnet34-LM"
SAFETY_FLOOR = 20
BALANCE_RATIO = 1.2 


# ... [Load Resources section] ...

# ==========================================
# 2. MEMORY MANAGER (NEW)
# ==========================================
def load_global_memory(memory_file):
    if os.path.exists(memory_file):
        print(f"Loading global memory from {memory_file}...")
        with open(memory_file, 'rb') as f:
            data = pickle.load(f)
        return data['X'], data['y']
    else:
        print("No global memory found. Starting fresh.")
        return [], []

def save_global_memory(X, y,memory_file):
    print(f"Saving updated memory ({len(X)} anchors) to {memory_file}...")
    with open(memory_file, 'wb') as f:
        pickle.dump({'X': X, 'y': y}, f)

def get_expanded_window(index, all_words):

    center_word = all_words[index]
    start_t = center_word['start']
    end_t = center_word['end']
    
    left_idx = index - 1
    right_idx = index + 1
    
    while (end_t - start_t) < TARGET_DURATION:
        changed = False
        if left_idx >= 0:
            prev_word = all_words[left_idx]
            if (start_t - prev_word['end']) <= MAX_GAP:
                start_t = prev_word['start'] 
                left_idx -= 1
                changed = True
        
        if (end_t - start_t) >= TARGET_DURATION:
            break

        if right_idx < len(all_words):
            next_word = all_words[right_idx]
            if (next_word['start'] - end_t) <= MAX_GAP:
                end_t = next_word['end'] 
                right_idx += 1
                changed = True
        
        if not changed:
            break
    
    current_dur = end_t - start_t
    if current_dur < TARGET_DURATION:
        needed = TARGET_DURATION - current_dur
        start_t = start_t - needed/2
        end_t = end_t - needed/2
        
    return start_t, end_t

def train_transcript(chunk_path,memory_file):
    # ==========================================
    # 1. CONFIGURATION
    # ==========================================
    AUDIO_FILE = os.path.join(chunk_path,'pyavi/audio.wav')
    TALKNET_CSV = os.path.join(chunk_path,'timeline.csv')
    WHISPER_JSON = os.path.join(chunk_path,'whisper_transcript.json')
    HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
    
    global_X, global_y = load_global_memory(memory_file)
    # ==========================================
    # 2. LOAD RESOURCES
    # ==========================================
    print("Loading Models & Data...")

    # A. Pyannote
    try:
        embedding_model = Model.from_pretrained(MODEL_ID, use_auth_token=HF_TOKEN)
    except Exception as e:
        print(f"Error: {e}")
        exit()
    inference = Inference(embedding_model, window="whole")

    # B. TalkNet (With Sanitization)
    df_talknet = pd.read_csv(TALKNET_CSV)

    # C. Whisper (FIXED LOADING)
    with open(WHISPER_JSON, 'r') as f:
        whisper_data = json.load(f)
        

    # ==========================================
    # 3. EXTRACTION & ANCHORING
    # ==========================================

    local_X_train = []
    local_y_train = []
    X_all = []    
    valid_indices = [] 

    # Counters
    count_on_screen = 0
    count_off_screen = 0

    print(f"Processing {len(whisper_data)} words with context expansion...")

    for i, word in enumerate(whisper_data):
        
        # 1. CALCULATE CONTEXT WINDOW
        c_start, c_end = get_expanded_window(i, whisper_data)
        
        if (c_end - c_start) < 0.05:
            print(f"Skipping tiny segment: {c_start:.2f}-{c_end:.2f}")
            valid_indices.append(i) # Keep index sync
            continue

        # 2. EXTRACT EMBEDDING
        try:
            excerpt = Segment(c_start, c_end)
            embedding = inference.crop(AUDIO_FILE, excerpt)
        except ValueError:
            continue

        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy().flatten()
        else:
            embedding = embedding.flatten()

        if np.isnan(embedding).any():
            continue

        X_all.append(embedding)
        valid_indices.append(i)

        # 3. ANCHOR LOGIC 
        anchor_label = None
        is_off_screen_candidate = False
        
        # A. Manual
        for anchor in MANUAL_ANCHORS:
            if c_start >= anchor['start'] and c_end <= anchor['end']:
                anchor_label = anchor['label']
                break
                
        # B. Visual
        if anchor_label is None:
            mask = (df_talknet['time'] >= c_start) & (df_talknet['time'] <= c_end)
            current_faces = df_talknet.loc[mask]
            
            if not current_faces.empty:
                face_scores = current_faces.groupby('speaker')['normalized_score_smooth'].mean()
                best_speaker = face_scores.idxmax()
                best_score = face_scores.max()

                # CASE 1: High Confidence Face
                if best_score > VISUAL_CONFIDENCE_THRESHOLD:
                    anchor_label = best_speaker
                    print(f"  [+] Anchored Face: {best_speaker} ({best_score:.2f})")
                

        # 4. BALANCING LOGIC (CRITICAL FIX)
        if anchor_label is not None:
            should_add = False
            
            if not is_off_screen_candidate:
                # ALWAYS add faces. 
                should_add = True
                count_on_screen += 1
            else:
                # Only add Off-Screen if we need to balance
                if count_off_screen < SAFETY_FLOOR:
                    should_add = True
                elif count_off_screen < (count_on_screen * BALANCE_RATIO):
                    should_add = True
                
                if should_add:
                    count_off_screen += 1

            if should_add:
                local_X_train.append(embedding)
                local_y_train.append(str(anchor_label))

    print(f"Final Distribution: Faces={count_on_screen}, Off-Screen={count_off_screen}")

    # ==========================================
    # 4. CLASSIFICATION & OUTPUT
    # ==========================================

    full_X_train = list(global_X) + local_X_train
    full_y_train = list(global_y) + local_y_train

    print(f"Training on {len(full_X_train)} anchors ({len(global_X)} global + {len(local_X_train)} local)")

    if len(full_X_train) == 0:
        print("Error: No anchors found in memory or this chunk.")
        exit()

    # Train KNN on the ACCUMULATED data
    k = min(7, len(full_X_train))
    clf = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='cosine')
    clf.fit(full_X_train, full_y_train)

    # Predict current chunk
    y_pred = clf.predict(X_all)
    y_proba = clf.predict_proba(X_all)

    final_output = []
    OFF_SCREEN_CUTOFF = 0.75 

    for idx, list_index in enumerate(valid_indices):
        word_obj = whisper_data[list_index]
        predicted_label = str(y_pred[idx]) # Force string
        confidence = np.max(y_proba[idx])
        
        # --- VISUAL CONFIRMATION LOGIC ---
        c_start, c_end = get_expanded_window(list_index, whisper_data)
        
        mask = (df_talknet['time'] >= c_start) & (df_talknet['time'] <= c_end)
        current_faces = df_talknet.loc[mask]
        
        is_visually_active = False
        visual_speaker = None
        
        if not current_faces.empty:
            face_scores = current_faces.groupby('speaker')['normalized_score_smooth'].mean()
            best_vis_score = face_scores.max()
            
            if best_vis_score > VISUAL_CONFIDENCE_THRESHOLD:
                is_visually_active = True
                visual_speaker = str(face_scores.idxmax())

        # --- FINAL LABEL DECISION TREE ---
        
        # 1. Trust Eyes (Is a face clearly talking?)
        if is_visually_active:
            display_label = f"Speaker_{visual_speaker}"
            
        # 3. Trust Ears (Is audio confidence high?)
        elif confidence > OFF_SCREEN_CUTOFF:
            if "Speaker" in predicted_label:
                display_label = f"{predicted_label}"
            else:
                display_label = f"Speaker_{predicted_label}"
        
        # 4. Unknown
        else:
            display_label = "Unidetified_Speaker"

        final_output.append({
            "start": round(word_obj['start'], 2),
            "end": round(word_obj['end'], 2),
            "speaker": display_label,
            "visability": is_visually_active,
            "text": word_obj['word'],
            "confidence": round(confidence, 2)
        })

    df_out = pd.DataFrame(final_output)
    output_path = os.path.join(chunk_path,'per_word_transcript.csv')
    df_out.to_csv(output_path, index=False)
    print("Done.")
    print(df_out[['start', 'speaker', 'text']].head(10))
    save_global_memory(full_X_train, full_y_train, memory_file)

if __name__ == '__main__':
    chunk_path ='/var/home/magpie/Development/SerialKiller_2/output/BvWOje46Xp8/chunks/chunk_000'
    memory_file = '/var/home/magpie/Development/SerialKiller_2/output/BvWOje46Xp8/global_training_memory.pkl'
    train_transcript(chunk_path,memory_file)