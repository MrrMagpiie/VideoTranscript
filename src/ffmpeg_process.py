import ffmpeg
import sys
import os
import re
import subprocess
from tqdm import tqdm

import pickle
import json
import os
import sys
import ffmpeg

# --- CONFIGURATION ---


def chunk_video(input_file,output_folder=None ,output_pattern="chunk_%03d.mp4", segment_time=300):
    """
    Converts video to mp4 and segments it.
    
    Args:
        input_file (str): Path to source video (e.g., 'video.webm')
        output_pattern (str): Output filename pattern (e.g., 'chunk_%03d.mp4')
        segment_time (int): Length of each chunk in seconds.
    """
    
    
    if not output_folder:
        output_folder = os.path.join(os.path.dirname(input_file),'chunks')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created directory: {output_folder}")

    final_output_pattern = os.path.join(output_folder, output_pattern)
            
    try:
        print(f"Processing {input_file} into {segment_time}s chunks...")
        
        (
            ffmpeg
            .input(input_file)
            .output(
                final_output_pattern,
                vcodec='libx264',
                crf=23,
                preset='fast',
                
                acodec='aac',
                audio_bitrate='128k',
                
                f='segment',           
                segment_time=segment_time,
                reset_timestamps=1     
            )
            .run(capture_stdout=True, capture_stderr=True)
        )
        print("Done! Chunks created.")

    except ffmpeg.Error as e:
        print("An error occurred with FFmpeg:")
        print(e.stderr.decode('utf8'))
        sys.exit(1)

def chunk_with_progress(input_file,output_folder=None ,output_pattern="chunk_%03d.mp4", segment_time=300):
    
    if not output_folder:
        output_folder = os.path.join(os.path.dirname(input_file),'chunks')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created directory: {output_folder}")

    final_output_pattern = os.path.join(output_folder, output_pattern)

    # 2. Use ffmpeg-python to get the video duration (for the bar total)
    try:
        probe = ffmpeg.probe(input_file)
        # We need the duration in seconds to know when the bar is full
        total_duration = float(probe['format']['duration'])
    except ffmpeg.Error as e:
        print(f"Could not probe video: {e.stderr}")
        sys.exit(1)

    print(f"Video Duration: {total_duration/60:.2f} minutes")

    # 3. Use ffmpeg-python to BUILD the command
    # We do NOT use .run() here. We use .compile()
    stream = ffmpeg.input(input_file)
    stream = ffmpeg.output(
        stream,
        final_output_pattern,
        vcodec='libx264',
        crf=23,
        preset='fast',
        
        acodec='aac',
        audio_bitrate='128k',
        
        f='segment',           
        segment_time=segment_time,
        reset_timestamps=1     
    )
    
    # This generates the list: ['ffmpeg', '-i', 'video.webm', ...]
    cmd = ffmpeg.compile(stream)

    # 4. Run the command manually to intercept progress
    # We capture 'stderr' because that's where FFmpeg prints "time=00:01:20"
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        encoding='utf-8' # Handles text decoding
    )

    # 5. The Progress Loop
    print(f"Chunking into '{output_folder}/'...")
    
    # Initialize the progress bar with the total duration
    with tqdm(total=total_duration, unit="s", dynamic_ncols=True) as pbar:
        
        # Regex to find "time=HH:MM:SS.ms" in the output
        pattern = re.compile(r"time=(\d{2}):(\d{2}):(\d{2}\.\d{2})")
        
        last_time = 0

        # Read the output line by line as it happens
        for line in process.stderr:
            match = pattern.search(line)
            if match:
                # Convert the timestamp to total seconds
                h, m, s = map(float, match.groups())
                current_time = (h * 3600) + (m * 60) + s
                
                # Update the bar by the amount of time passed since the last update
                increment = current_time - last_time
                if increment > 0:
                    pbar.update(increment)
                    last_time = current_time

    # Clean up
    process.wait()
    
    if process.returncode == 0:
        return output_folder
    else:
        print("Error during chunking.")


if __name__ == "__main__":
    pass