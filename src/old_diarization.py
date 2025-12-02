from dotenv import load_dotenv
import os
import torch
import shutil
import json
import yt_dlp
from pyannote.audio import Pipeline
from huggingface_hub import HfFolder
from pyannote.audio.pipelines.utils.hook import ProgressHook
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import JSONFormatter



def save_token_to_cache(token):
    if not token or token == "YOUR_HUGGING_FACE_TOKEN_GOES_HERE":
        return False
    try:
        HfFolder.save_token(token)
        return True
    except Exception:
        return False

def check_ffmpeg():
    return shutil.which("ffmpeg") is not None

def get_transcript_from_youtube(video_id):
    ytt_api = YouTubeTranscriptApi()
    try:
        transcript = ytt_api.fetch(video_id)
    except Exception as e:
        raise e

    formatter = JSONFormatter()
    json_formatted = json.loads(formatter.format_transcript(transcript))
    endstamp_formatted = add_end_timestamps_transcript(json_formatted)
    
    try:
        with open(f'{video_id}_transcript.json', 'w', encoding='utf-8') as f:
            json.dump(endstamp_formatted,f,indent=4)
    except FileNotFoundError as e:
        raise e
    

def add_end_timestamps_transcript(transcript):
    for item in transcript:
        item['end'] = item['start'] + item['duration']
    return transcript

def download_audio_from_youtube(video_id, output_filename):
    """Downloads audio if it doesn't exist."""
    final_wav_path = f"{output_filename}.wav"
    
    if os.path.exists(final_wav_path):
        print(f"Audio file found ({final_wav_path}). Skipping download.")
        return final_wav_path

    video_url = f'https://www.youtube.com/watch?v={video_id}'
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_filename,
        'quiet': False,
        'postprocessors': [{'key': 'FFmpegExtractAudio','preferredcodec': 'wav','preferredquality': '192'}],
        'postprocessor_args': ['-ar', '16000', '-ac', '1'],
    }

    print(f"Downloading Audio...")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        return final_wav_path
    except Exception as e:
        print(f"Download Error: {e}")
        return None

def run_diarization_task(audio_path, hf_token, output_json_path):
    print(f"\n--- TASK 1: Running Diarization on {audio_path} ---")
    
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
        cache_dir="./models"
    )

    if torch.cuda.is_available():
        print("GPU detected. Using CUDA.")
        pipeline.to(torch.device("cuda"))

    print("Analyzing audio (this takes time)...")
    with ProgressHook() as hook:
        diarization = pipeline(audio_path, hook=hook)

    results = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        results.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })

    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Diarization complete! Data saved to: {output_json_path}")


def merge_task(video_id, json_path):
    print(f"\n--- TASK 2: Merging Transcript with {json_path} ---")
    
    if not os.path.exists(json_path):
        print("Error: Diarization JSON file not found. Run Task 1 first.")
        return

    with open(json_path, "r") as f:
        speaker_segments = json.load(f)

    transcript = get_transcript_from_youtube(video_id)

    full_text_output = []
    current_speaker = None
    
    for segment in transcript:
        text = segment['text']
        start = segment['start']
        duration = segment['duration']
        midpoint = start + (duration / 2)

        active_speaker = "Unknown"
        for sp_seg in speaker_segments:
            if sp_seg["start"] <= midpoint <= sp_seg["end"]:
                active_speaker = sp_seg["speaker"]
                break
        
        if active_speaker != current_speaker:
            full_text_output.append(f"\n\n**{active_speaker}** [{int(start)}s]:")
            current_speaker = active_speaker
        
        full_text_output.append(text)

    final_string = " ".join(full_text_output).replace("  ", " ").strip()
    
    print("\n" + "="*20 + " RESULT " + "="*20)
    print(final_string[:500] + "...\n(truncated for console)") 
    print("="*50)
    
    with open("final_script.txt", "w", encoding="utf-8") as f:
        f.write(final_string)
    print("Full text saved to 'final_script.txt'")


def youtube_transcript_diarization_pipeline(video_id = None, audio_file = None,diarized_file = None):
    load_dotenv()
    HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

    if not check_ffmpeg() or not save_token_to_cache(HF_TOKEN):
        exit(1)

    if not os.path.exists(diarized_file):
        print("Diarization data not found. Starting Task 1...")
        audio_path = download_audio_from_youtube(video_id, audio_file)
        if audio_path:
            run_diarization_task(audio_path, HF_TOKEN, diarized_file)
    else:
        print(f"Found existing diarization data ({diarized_file}). Skipping AI processing.")

    merge_task(video_id, diarized_file)

if __name__ == "__main__":
    diarized_file = "diarization_data.json"
    audio_file = "downloaded_audio"
    video_id = "BvWOje46Xp8" 

    #youtube_transcript_diarization_pipeline(video_id,audio_file,diarized_file)
    get_transcript_from_youtube(video_id)
