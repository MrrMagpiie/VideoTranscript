import os
import json
import time  # <--- Added time module
from datetime import timedelta # <--- For pretty printing

from youtube_dl import download_video_from_youtube, get_transcript_from_youtube, download_native_video
from ffmpeg_process import chunk_with_progress
from talknet_queue import talknet_queue, get_chunk_folders, get_chunks
from whisper_transcription import run_whisper_transcription, run_pyannote_diarization
from pywork_conversion import convert_pickles
from face_reidentification import process_reidentification
from consolidation import create_timeline
from new_analysis import train_transcript
from global_registry import GlobalFaceRegistry

# --- HELPER FOR TIMING ---
timers = {}
def start_timer(name):
    timers[name] = time.time()

def stop_timer(name, accumulators):
    if name in timers:
        duration = time.time() - timers[name]
        # Add to total accumulation for this task
        accumulators[name] = accumulators.get(name, 0.0) + duration
        return duration
    return 0.0

def print_stats(accumulators, total_time):
    print("\n" + "="*40)
    print(f"{'TASK NAME':<25} | {'DURATION':<15}")
    print("-" * 40)
    
    for task, duration in accumulators.items():
        # Format seconds into HH:MM:SS
        pretty_time = str(timedelta(seconds=int(duration)))
        print(f"{task:<25} | {pretty_time}")
    
    print("-" * 40)
    print(f"{'TOTAL RUNTIME':<25} | {str(timedelta(seconds=int(total_time)))}")
    print("="*40 + "\n")

# --- MAIN SCRIPT ---

if __name__ == '__main__':
    # Initialize stats container
    execution_stats = {}
    program_start_time = time.time()

    HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
    VIDEO_ID = "6GzxbrO0DHM"
    OUTPUT_PATH = '/var/home/magpie/Development/SerialKiller_2/benchmarking'

    # create Video dir in output if it doesnt exist
    if not os.path.exists(os.path.join(OUTPUT_PATH,VIDEO_ID)):
        os.mkdir(os.path.join(OUTPUT_PATH,VIDEO_ID))

    # --- 1. DOWNLOAD PHASE ---
    start_timer('Download')
    '''video_path = download_native_video(VIDEO_ID,OUTPUT_PATH)
    yt_transcript = get_transcript_from_youtube(VIDEO_ID)
    with open(f'{OUTPUT_PATH}/{VIDEO_ID}/yt_transcript.json','w') as f:
        json.dump(yt_transcript,f,indent=4)'''
    stop_timer('Download', execution_stats)

    # --- 2. CHUNKING PHASE ---
    start_timer('FFmpeg Chunking')
    #use ffmpeg to chunk the video
    video_path = '/var/home/magpie/Development/SerialKiller_2/benchmarking/6GzxbrO0DHM/6GzxbrO0DHM_video.mp4'
    #chunk_path = chunk_with_progress(video_path,segment_time=60)
    chunk_path = '/var/home/magpie/Development/SerialKiller_2/benchmarking/6GzxbrO0DHM/chunks'
    stop_timer('FFmpeg Chunking', execution_stats)

    # --- 3. TALKNET PHASE ---
    start_timer('TalkNet Processing')
    # use talknet to process the video chunks
    # processed_chunks = talknet_queue(chunk_path)
    stop_timer('TalkNet Processing', execution_stats)

    # Setup Paths
    video_files = get_chunks(f'/var/home/magpie/Development/SerialKiller_2/benchmarking/{VIDEO_ID}/chunks')
    processed_chunks = get_chunk_folders(video_files)
    print(processed_chunks)
    
    global_face_registry = f'/var/home/magpie/Development/SerialKiller_2/benchmarking/{VIDEO_ID}/global_faces.pkl'
    training_registry = f'/var/home/magpie/Development/SerialKiller_2/benchmarking/{VIDEO_ID}/global_training_memory.pkl'

    # --- 4. CHUNK LOOP ---
    print(f"\nStarting processing on {len(processed_chunks)} chunks...")
    
    for i, chunk in enumerate(processed_chunks):
        print(f'\n--- Chunk {i+1}/{len(processed_chunks)}: {os.path.basename(chunk)} ---')
        
        audio_path = os.path.join(chunk,'pyavi/audio.wav')

        # A. Whisper
        start_timer('Whisper Transcription')
        whisper_transcript = os.path.join(chunk,'whisper_transcript.json')
        words = run_whisper_transcription(audio_path)
        with open(whisper_transcript, "w") as f:
            json.dump(words,f,indent=4)
        stop_timer('Whisper Transcription', execution_stats)
        
        # B. Pyannote (Commented out in your code, wrapped just in case)
        ''' 
        start_timer('Pyannote Diarization')
        pyannote_diarization = os.path.join(chunk,'diarization_data.json')
        speakers = run_pyannote_diarization(audio_path, HF_TOKEN)
        with open(pyannote_diarization, "w") as f:
            json.dump(speakers, f,indent=4)
        stop_timer('Pyannote Diarization', execution_stats)
        '''
        
        # C. Pickle Conversion
        start_timer('Pickle Conversion')
        pywork_path = os.path.join(chunk,'pywork')
        converted_path = os.path.join(chunk,'talknet_data.json')
        converted_data = convert_pickles(pywork_path)
        with open(converted_path, "w") as f:
            json.dump(converted_data, f,indent=4)
        stop_timer('Pickle Conversion', execution_stats)

        # D. Face Re-ID
        start_timer('Face Re-ID')
        match_matrix = process_reidentification(chunk, global_face_registry)
        stop_timer('Face Re-ID', execution_stats)

        # E. Timeline Creation
        start_timer('Timeline Creation')
        timeline = create_timeline(chunk, match_matrix)
        stop_timer('Timeline Creation', execution_stats)

        # F. Model Training/Inference
        start_timer('Training & Inference')
        train_transcript(chunk, training_registry)
        stop_timer('Training & Inference', execution_stats)

    # --- FINAL STATS ---
    total_duration = time.time() - program_start_time
    print_stats(execution_stats, total_duration)