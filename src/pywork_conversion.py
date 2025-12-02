import pickle
import json
import numpy as np
import os
import sys
import ffmpeg
import pandas as pd


class NumpyEncoder(json.JSONEncoder):
    """
    Magic helper to convert NumPy arrays to Python lists so JSON doesn't crash.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, np.int32) or isinstance(obj, np.int64):
            return int(obj)
        return super(NumpyEncoder, self).default(obj)


def get_fps(video_file):
    try:
        # Run probe
        probe = ffmpeg.probe(video_file)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        
        if video_stream is None:
            print("Error: No video stream found in file.")
            sys.exit(1)
            
        r_frame_rate = video_stream['r_frame_rate']
        
        if '/' in r_frame_rate:
            num, den = map(int, r_frame_rate.split('/'))
            return num / den
        else:
            return float(r_frame_rate)

    except ffmpeg.Error as e:
        # --- THIS IS THE FIX ---
        # Print the actual error message from FFmpeg
        print("\n--- FFPROBE ERROR ---")
        print(e.stderr.decode('utf8')) 
        print("---------------------")
        
        # Fallback (Optional: force 29.97 if you just want it to run)
        print("Defaulting to 29.97 FPS due to error.")
        return 29.97
    except Exception as e:
        print(f"General Error: {e}")
        return 29.97

def convert_pickles(chunk_folder):
    tracks_path = os.path.join(chunk_folder, 'tracks.pckl')
    scores_path = os.path.join(chunk_folder, 'scores.pckl')
    chunk_video = os.path.join(chunk_folder, 'pyavi','video.avi')

    fps = 25
    if not os.path.exists(tracks_path) or not os.path.exists(scores_path):
        print(f"Error: Could not find files in '{chunk_folder}'")
        sys.exit(1)

    print("Loading pickle files...")
    try:
        with open(tracks_path, 'rb') as f:
            tracks = pickle.load(f)
        with open(scores_path, 'rb') as f:
            scores = pickle.load(f)
    except Exception as e:
        print(f"Error reading pickles: {e}")
        sys.exit(1)
    master_timeline = []
    merged_data = {}
    print(f"Processing {len(tracks)} face tracks...")

    for i, track in enumerate(tracks):
        if i >= len(scores):
            break

        frame_indices = track['track']['frame']
        track_scores = scores[i]
        
        # Get coordinate arrays
        x_coords = track['proc_track']['x']
        y_coords = track['proc_track']['y']
        s_coords = track['proc_track']['s']

        # --- THE FIX IS HERE ---
        # We calculate the safe limit by taking the minimum length of all arrays
        # This prevents the "Index out of bounds" error if one list is shorter
        limit = min(len(frame_indices), len(track_scores), len(x_coords))

        person_data = {
            "track_id": i,
            "total_frames": limit,
            "timeline": []
        }

        for k in range(limit):
            # Now we use 'limit', so k will never exceed the size of track_scores
            frame = int(frame_indices[k])

            raw_score = float(track_scores[k])
            master_timeline.append({'speaker': i,'score':raw_score, 'frame':frame, 'time':frame/25})
            
            is_speaking = raw_score > 0
            if is_speaking:
                speaking_track = True
            frame_data = {
                "frame": int(frame_indices[k]),
                "time":  int(frame_indices[k])/fps,
                "score": round(raw_score, 4),
                "is_speaking": is_speaking,
                "face_box": {
                    "x": int(x_coords[k]),
                    "y": int(y_coords[k]),
                    "s": int(s_coords[k])
                }
            }
            person_data["timeline"].append(frame_data)

        merged_data[i] = person_data
        merged_data['master_timeline'] = master_timeline

    return merged_data

def get_segments(frames, scores, speaker_label):
    """
    Scans a list of frames and scores to find continuous 'speaking' blocks.
    Returns a list of dicts: [{'start': 10, 'end': 20, 'speaker': 'SPEAKER_01'}, ...]
    """
    segments = []
    
    current_start = None
    last_frame = None
    
    # We iterate through the frames and scores simultaneously
    # Limit ensures we don't crash if arrays are different sizes
    limit = min(len(frames), len(scores))
    
    for i in range(limit):
        frame = frames[i]
        score = float(scores[i])
        is_speaking = score > 0  # Threshold (you can adjust this to 0.1 or higher if noisy)

        if is_speaking:
            if current_start is None:
                # Start of a new segment
                current_start = frame
            
            # Check if we skipped frames (e.g. face was lost for a second)
            # If the gap between this frame and the last one is > 1, close the previous segment
            elif last_frame is not None and (frame - last_frame) > 1:
                segments.append({
                    "start": current_start,
                    "end": last_frame,
                    "speaker": speaker_label
                })
                current_start = frame
            
            # Update the end pointer
            last_frame = frame
            
        else:
            # We hit a silent frame. If we were recording a segment, close it.
            if current_start is not None:
                segments.append({
                    "start": current_start,
                    "end": last_frame,
                    "speaker": speaker_label
                })
                current_start = None
                last_frame = None

    # Edge Case: If the person is still speaking at the very end of the video
    if current_start is not None:
        segments.append({
            "start": current_start,
            "end": last_frame,
            "speaker": speaker_label
        })
        
    return segments




if __name__ == "__main__":
    chunk_folder = '/var/home/magpie/Development/SerialKiller_2/output/BvWOje46Xp8/chunks/chunk_000/pywork' 
    OUTPUT_FILE = '/var/home/magpie/Development/SerialKiller_2/output/BvWOje46Xp8/chunks/chunk_000/talknet_data.json'
    data = convert_pickles(chunk_folder)

    print(f"Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(data, f, cls=NumpyEncoder, indent=2)