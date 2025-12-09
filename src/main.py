import os
import json
import time
from datetime import timedelta
from youtube_dl import download_video_from_youtube, get_transcript_from_youtube, download_native_video
from ffmpeg_process import chunk_with_progress
from talknet_queue import talknet_queue, get_chunk_folders, get_chunks
from whisper_transcription import run_whisper_transcription, run_pyannote_diarization
from pywork_conversion import convert_pickles
from face_reidentification import process_reidentification
from consolidation import create_timeline
from new_analysis import train_transcript
from global_registry import GlobalFaceRegistry

# ==========================================
# PERFORMANCE MONITOR CLASS
# ==========================================
class PerformanceMonitor:
    def __init__(self):
        self.global_start = time.time()
        self.task_start_times = {}
        
        # Stores total duration per task: {'Whisper': 50.0, ...}
        self.global_totals = {} 
        
        # Stores list of per-chunk stats: [{'Chunk': 1, 'Whisper': 5.0}, ...]
        self.chunk_history = [] 
        
        # Temporary holder for the current chunk being processed
        self.current_chunk_stats = {}

    def start_chunk(self, chunk_index, chunk_name):
        """Reset temporary stats for a new chunk."""
        self.current_chunk_stats = {'ID': chunk_index, 'Name': os.path.basename(chunk_name)}

    def start_task(self, task_name):
        self.task_start_times[task_name] = time.time()

    def stop_task(self, task_name):
        if task_name in self.task_start_times:
            duration = time.time() - self.task_start_times[task_name]
            
            # 1. Add to current chunk stats
            self.current_chunk_stats[task_name] = duration
            
            # 2. Add to global totals
            self.global_totals[task_name] = self.global_totals.get(task_name, 0.0) + duration
            return duration
        return 0.0

    def end_chunk(self):
        """Save the current chunk's stats to history."""
        self.chunk_history.append(self.current_chunk_stats)

    def print_report(self):
        total_pipeline_time = time.time() - self.global_start
        
        # Collect all unique task names encountered
        all_tasks = list(self.global_totals.keys())
        
        print("\n" + "="*80)
        print(f"{'PERFORMANCE REPORT':^80}")
        print("="*80)

        # 1. PER CHUNK BREAKDOWN
        # Header
        header = f"{'Chunk':<5} | " + " | ".join([f"{t[:10]:<10}" for t in all_tasks])
        print(header)
        print("-" * len(header))

        for stats in self.chunk_history:
            row = f"{stats['ID']:<5} | "
            for task in all_tasks:
                dur = stats.get(task, 0.0)
                # Format: 12.5s
                row += f"{dur:10.1f} | "
            print(row)

        print("-" * len(header))

        # 2. GLOBAL SUMMARY
        print("\n" + "="*40)
        print(f"{'TASK SUMMARY':<25} | {'TOTAL':<10} | {'AVG/CHUNK':<10}")
        print("-" * 40)
        
        num_chunks = len(self.chunk_history) if len(self.chunk_history) > 0 else 1
        
        for task in all_tasks:
            total = self.global_totals[task]
            avg = total / num_chunks
            print(f"{task:<25} | {str(timedelta(seconds=int(total))):<10} | {avg:<10.1f}s")
            
        print("-" * 40)
        print(f"{'TOTAL RUNTIME':<25} | {str(timedelta(seconds=int(total_pipeline_time)))}")
        print("="*40 + "\n")

# ==========================================
# MAIN SCRIPT
# ==========================================

if __name__ == '__main__':
    monitor = PerformanceMonitor()

    HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
    VIDEO_ID = "BvWOje46Xp8"
    OUTPUT_PATH = '/var/home/magpie/Development/SerialKiller_2/output'

    if not os.path.exists(os.path.join(OUTPUT_PATH,VIDEO_ID)):
        os.mkdir(os.path.join(OUTPUT_PATH,VIDEO_ID))

    # --- PRE-PROCESSING (Global Tasks) ---
    monitor.start_task('Global_Process')
    monitor.start_task('Download')

    # --- 1. DOWNLOAD PHASE ---
    '''video_path = download_native_video(VIDEO_ID,OUTPUT_PATH)
    yt_transcript = get_transcript_from_youtube(VIDEO_ID)
    with open(f'{OUTPUT_PATH}/{VIDEO_ID}/yt_transcript.json','w') as f:
        json.dump(yt_transcript,f,indent=4)'''

    monitor.stop_task('Download')

    monitor.start_task('ffmpeg_chunking')
    # --- 2. CHUNKING PHASE ---
    #use ffmpeg to chunk the video
    video_path = os.path.join(OUTPUT_PATH, f'{VIDEO_ID}/{VIDEO_ID}_video.mp4')
    #chunk_path = chunk_with_progress(video_path,segment_time=60)

    monitor.stop_task('ffmpeg_chunking')

    monitor.start_task('Talknet_Process')

        # --- 3. TALKNET PHASE ---
    # use talknet to process the video chunks
    #processed_chunks = talknet_queue(chunk_path)
    monitor.stop_task('Talknet_Process')
    
    # Setup Paths
    '''video_files = get_chunks(chunk_path)
    processed_chunks = get_chunk_folders(video_files)
    print(processed_chunks)'''
    processed_chunks = []
    for chunk in range(10,31):
        processed_chunks.append(f'/var/home/magpie/Development/SerialKiller_2/output/BvWOje46Xp8/chunks/chunk_0{chunk}')
    print(processed_chunks)
    global_face_registry = os.path.join(OUTPUT_PATH,f'{VIDEO_ID}/global_faces.pkl')
    training_registry = os.path.join(OUTPUT_PATH,f'{VIDEO_ID}/global_training_memory.pkl')
    
    monitor.stop_task('Global_Process')

    # --- CHUNK LOOP ---
    print(f"\nStarting processing on {len(processed_chunks)} chunks...")
    
    for i, chunk in enumerate(processed_chunks):
        chunk_name = os.path.basename(chunk)
        print(f'\n>>> Processing Chunk {i+1}: {chunk_name}')
        
        # Start recording stats for this specific chunk
        monitor.start_chunk(i+1, chunk_name)
        
        audio_path = os.path.join(chunk,'pyavi/audio.wav')

        # A. Whisper
        monitor.start_task('Whisper')
        whisper_transcript = os.path.join(chunk,'whisper_transcript.json')
        words = run_whisper_transcription(audio_path)
        with open(whisper_transcript, "w") as f:
            json.dump(words,f,indent=4)
        monitor.stop_task('Whisper')
        
        # B. TalkNet Conversion
        monitor.start_task('Pickle Conv')
        pywork_path = os.path.join(chunk,'pywork')
        converted_path = os.path.join(chunk,'talknet_data.json')
        converted_data = convert_pickles(pywork_path)
        with open(converted_path, "w") as f:
            json.dump(converted_data, f,indent=4)
        monitor.stop_task('Pickle Conv')

        # C. Face Re-ID
        monitor.start_task('Face Re-ID')
        match_matrix = process_reidentification(chunk, global_face_registry)
        monitor.stop_task('Face Re-ID')

        # D. Timeline Creation
        monitor.start_task('Timeline')
        timeline = create_timeline(chunk, match_matrix)
        monitor.stop_task('Timeline')

        # E. Model Training/Inference
        monitor.start_task('Inference')
        train_transcript(chunk, training_registry)
        monitor.stop_task('Inference')

        # Close out this chunk's stats
        monitor.end_chunk()

    # --- FINAL REPORT ---
    monitor.print_report()