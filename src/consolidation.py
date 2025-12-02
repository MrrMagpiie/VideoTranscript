import json
import os
import pandas as pd
import numpy as np


def consolidate_diarization(diarization_file, max_gap=2.0):
    """
    Merges adjacent segments from the same speaker if they are close together.
    max_gap: Maximum silence (in seconds) allowed to merge segments.
    """
    with open(diarization_file, 'r') as  f:
        diarization_segments = json.load(f)

    if not diarization_segments:
        return []

    consolidated = []
    current_seg = diarization_segments[0]

    for next_seg in diarization_segments[1:]:
        # Check if same speaker AND gap is small enough
        gap = next_seg["start"] - current_seg["end"]
        
        if next_seg["speaker"] == current_seg["speaker"] and gap <= max_gap:
            current_seg["end"] = next_seg["end"]
        else:
            current_seg['text'] = []
            consolidated.append(current_seg)
            current_seg = next_seg
            
    current_seg['text'] = []
    consolidated.append(current_seg)
    return consolidated

def assign_text_to_speakers(transcript, diarization_segments):
    d_idx = 0
    current_speaker = diarization_segments[d_idx]
    finished_transcript = []

    for item in transcript:
        if item['start'] > diarization_segments[d_idx]['start'] and item['start'] < diarization_segments[d_idx]['end']:
            current_speaker['text'].join(item['text'])
        else: 
            d_idx += 1
            current_speaker= diarization_segments[d_idx]

def merge_transcript_to_speakers(transcript, speaker_segments):
    final_script = []
    speak_idx = 0
    current_speaker_block = speaker_segments[speak_idx]

    for text_seg in transcript:
        t_start = text_seg['start']
        
        while speak_idx < len(speaker_segments) - 1:
            curr_sp = speaker_segments[speak_idx]
            next_sp = speaker_segments[speak_idx + 1]
            
            # If the text starts AFTER the current speaker ends, we define it as belonging to the next.
            # We use a small buffer (e.g. 0.5s) because humans trail off.
            if t_start > curr_sp["end"]: 
                speak_idx += 1
            else:
                break
        
        target_speaker = speaker_segments[speak_idx]["speaker"]

        # 2. If the speaker changed, push the old block and start a new one
        if target_speaker != current_speaker_block["speaker"]:
            final_script.append(current_speaker_block)
            current_speaker_block = speaker_segments[speak_idx]
        else:
            # Same speaker, just accumulate text
            text = text_seg['text']
            current_speaker_block['text'].append(text)

    # Append the final block
    final_script.append({
        "speaker": current_speaker_block["speaker"],
        "text": " ".join(current_speaker_block["text"])
    })

    return final_script

def annotate_transcript(whisper_words, talknet_df, threshold=-0.1):
    annotated_transcript = []

    for word_obj in whisper_words:
        w_start = word_obj['start']
        w_end = word_obj['end']
        
        # A. SLICE: Get all TalkNet rows that happen DURING this word
        # We add a small buffer (e.g., 0.1s) to catch edge frames
        mask = (talknet_df['time'] >= w_start-0.03) & (talknet_df['time'] <= w_end+0.03)
        window_df = talknet_df.loc[mask]

        if window_df.empty:
            # No video data for this timestamp (audio might be longer than video)
            assigned_speaker = "Unknown"
        else:
            # B. AGGREGATE: Group by face_id and get the mean score for this word duration
            face_summary = window_df.groupby('speaker')['score_smooth'].mean()
            
            # C. DECIDE: Find the face with the highest average score
            best_face_id = face_summary.idxmax()
            best_score = face_summary.max()

            if best_score > threshold:
                assigned_speaker = f"Speaker_{best_face_id}"
            else:
                assigned_speaker = "Off-Screen"

        # D. BUILD OUTPUT
        annotated_transcript.append({
            'start': w_start,
            'end': w_end,
            'word': word_obj['word'],
            'speaker': assigned_speaker,
            'confidence': best_score if not window_df.empty else 0.0
        })
    
    return annotated_transcript

def generate_readable_transcript(df):
    """
    Takes the word-level DataFrame and groups it into speaker segments.
    """
    transcript_segments = []
    
    if df.empty:
        return "No data to process."

    # 1. Initialize the first segment
    current_speaker = df.iloc[0]['speaker']
    current_start = df.iloc[0]['start']
    current_words = [df.iloc[0]['word']]

    # 2. Iterate through the rest of the rows
    for i in range(1, len(df)):
        row = df.iloc[i]
        speaker = row['speaker']
        word = row['word']
        
        # If the speaker has not changed, keep adding words to the buffer
        if speaker == current_speaker:
            current_words.append(word)
        
        # If the speaker CHANGED, save the previous block and start a new one
        else:
            # Save the completed segment
            segment_text = " ".join(current_words)
            transcript_segments.append({
                "start": current_start,
                "end": df.iloc[i-1]['end'], # End time of the previous word
                "speaker": current_speaker,
                "text": segment_text
            })
            
            # Reset for the new speaker
            current_speaker = speaker
            current_start = row['start']
            current_words = [word]

    # 3. Don't forget to save the very last segment after the loop finishes
    final_text = " ".join(current_words)
    transcript_segments.append({
        "start": current_start,
        "end": df.iloc[-1]['end'],
        "speaker": current_speaker,
        "text": final_text
    })

    return transcript_segments

def print_formatted_transcript(segments):
    print(f"{'TIMESTAMP':<10} | {'SPEAKER':<15} | {'TEXT'}")
    print("-" * 60)
    
    for seg in segments:
        # Convert seconds to MM:SS format
        m, s = divmod(seg['start'], 60)
        timestamp = f"{int(m):02d}:{int(s):02d}"
        
        print(f"{timestamp:<10} | {seg['speaker']:<15} | {seg['text']}")

# -------------------
# Timeline
# ------------------
def load_master_timeline(path):
    with open(path,'r') as f:
        data = json.load(f)
        return pd.DataFrame(data.get('master_timeline'))
def create_timeline(chunk_path, match_matrix):
    
    talknet_path = os.path.join(chunk_path,'talknet_data.json')
    timeline_path = os.path.join(chunk_path,'timeline.csv')
    timeline = load_master_timeline(talknet_path)
    transcript_path = os.path.join(chunk_path, 'whisper_transcript.json')
   
    with open(transcript_path,'r') as f:
        whisper_transcript = json.load(f)
    
    if match_matrix:
        timeline['speaker'] = timeline['speaker'].replace(match_matrix)

    # ------------------------------------------
    
    if timeline['score'].max() > 1.0 or timeline['score'].min() < 0.0:
        timeline['normalized_score'] = 1 / (1 + np.exp(-timeline['score']))
        print(f"Scores normalized. New Max: {timeline['normalized_score'].max():.4f}")

    timeline['score_smooth'] = timeline.groupby('speaker')['score'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
        )

    timeline['normalized_score_smooth'] = timeline.groupby('speaker')['normalized_score'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
        )

    timeline.to_csv(timeline_path)
    return timeline

if __name__ == "__main__":
    chunk_path = '/var/home/magpie/Development/SerialKiller_2/output/BvWOje46Xp8/chunks/chunk_000'
    create_timeline(chunk_path)