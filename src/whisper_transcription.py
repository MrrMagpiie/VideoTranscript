import os
import json
import torch
import yt_dlp
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.console import Console
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from huggingface_hub import HfFolder
from pyannote.audio.pipelines.utils.hook import ProgressHook

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"

def download_audio_from_youtube(video_id, output_file):
    """Downloads audio if it doesn't exist."""
    final_wav_path = f"{output_file}/{video_id}/{video_id}_audio.wav"

    video_url = f'https://www.youtube.com/watch?v={video_id}'
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': final_wav_path,
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

def download_video_from_youtube(video_id,output_file):
    """Downloads audio if it doesn't exist."""
    final_output_path = f"{output_file}/{video_id}/{video_id}_video"

    video_url = f'https://www.youtube.com/watch?v={video_id}'
    
    ydl_opts = {
        'outtmpl': final_output_path,
        'cookiesfrombrowser': ('firefox'),
    }
    
    print(f"Downloading Video...")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        return final_output_path
    except Exception as e:
        print(f"Download Error: {e}")
        return None

def run_whisper_transcription(audio_path):
    """
    Runs Whisper locally to get text with word-level timestamps.
    Uses Rich to display a progress bar.
    """
    console = Console()
    console.print(f"\n[bold green]--- TASK: Running Whisper on {DEVICE} ---[/bold green]")
    
    model_size = "medium.en" 

    model = WhisperModel(model_size, device=DEVICE, compute_type=COMPUTE_TYPE)

    segments, info = model.transcribe(audio_path, word_timestamps=True)

    console.print(f"Detected language '[bold]{info.language}[/bold]' with probability {info.language_probability:.2f}")
    
    word_level_data = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        
        task = progress.add_task("[cyan]Transcribing...", total=info.duration)
        
        for segment in segments:
            progress.update(task, completed=segment.end)
            
            for word in segment.words:
                word_level_data.append({
                    "word": word.word,
                    "start": word.start,
                    "end": word.end,
                    "confidence": word.probability
                })
            
    console.print(f"[bold green]Transcription complete.[/bold green] {len(word_level_data)} words detected.")
    return word_level_data

def run_pyannote_diarization(audio_path, hf_token):
    """
    Runs Pyannote to get speaker timestamps.
    """
    print(f"\n--- TASK: Running Diarization ---")
    
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token
    ).to(torch.device(DEVICE))

    with ProgressHook() as hook:
        diarization = pipeline(audio_path, hook=hook)

    speaker_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
        speaker_segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })
        
    return speaker_segments

def merge_whisper_and_pyannote(words, speaker_segments):
    """
    Matches specific words to speakers based on timestamps.
    """
    final_script = []
    speaker_idx = 0
    
    current_speaker_block = None
    
    for word_data in words:
        w_start = word_data['start']
        w_end = word_data['end']
        while speaker_idx < len(speaker_segments) - 1:
            curr_seg = speaker_segments[speaker_idx]
            next_seg = speaker_segments[speaker_idx + 1]
            
            if w_start > curr_seg["end"]:
                speaker_idx += 1
            else:
                break
        
        curr_seg = speaker_segments[speaker_idx]
        
    
        

        if w_start > curr_seg["end"] and speaker_idx == len(speaker_segments) - 1:
             # End of audio edge case
             active_speaker = curr_seg["speaker"]
        elif w_start < curr_seg["start"] and speaker_idx == 0:
             # Start of audio edge case
             active_speaker = curr_seg["speaker"]

        elif w_start < curr_seg["end"]:
            active_speaker = curr_seg["speaker"]
        
        else:
            print(f'{w_start} is after {curr_seg['speaker']} ends at {curr_seg['ends']}')

        # Build the Output lines
        if current_speaker_block and current_speaker_block["speaker"] == active_speaker:
            current_speaker_block["text"].append((word_data["word"],w_start,w_end))
            current_speaker_block['end'] = curr_seg['end']
        else:
            if current_speaker_block:
                final_script.append(current_speaker_block)
            
            current_speaker_block = {
                "speaker": active_speaker,
                "start": curr_seg['start'],
                'end': curr_seg['end'],
                "text": [(word_data["word"],w_start,w_end)]
            }

    # Append final block
    if current_speaker_block:
        final_script.append(current_speaker_block)

    return final_script

def format_output(script_data):
    text_out = ""
    for block in script_data:
        # Convert list of words back to string
        sentence = "".join([w for w in block["text"]])
        
        # Basic timestamp formatting
        mins = int(block["start"] // 60)
        secs = int(block["start"] % 60)
        timestamp = f"{mins:02}:{secs:02}"
        
        text_out += f"\n[{timestamp}] {block['speaker']}: {sentence}"
    return text_out.strip()

def check_output(video_id,output_path):
    video_path = f'{output_path}/{video_id}'
    if not os.path.exists(video_path):
        os.makedirs(video_path)
    return video_path
        
def transciption_diarization_pipeline(video_id,output_path,token):
    # 1. Get Audio
    save_path = check_output(video_id,output_path)

    audio_file = f'{save_path}/{video_id}_audio.wav'
    diarization_data_file = f'{save_path}/diarization_data.json'
    whisper_transcript = f'{save_path}/transcript_data.json'
    merge_file = f'{save_path}/merged_data.json'

    if not os.path.exists(audio_file):
        download_audio_from_youtube(video_id,audio_file)

    # 2. Get Words (Whisper)
    if os.path.exists(whisper_transcript):
        with open(whisper_transcript, "r") as f:
            words = json.load(f)
    else:
        words = run_whisper_transcription(audio_file)
        with open(whisper_transcript, "w") as f:
            json.dump(words,f,indent=4)

    # 3. Get Speakers (Pyannote)
    if os.path.exists(diarization_data_file):
        with open(diarization_data_file, "r") as f:
            speakers = json.load(f)
    else:
        speakers = run_pyannote_diarization(audio_file, token)
        with open(diarization_data_file, "w") as f:
            json.dump(speakers, f,indent=4)

    # 4. Merge
    final_data = merge_whisper_and_pyannote(words, speakers)
    with open(merge_file, "w") as f:
            json.dump(final_data,f,indent=4)
    
    # 5. Save
    readable_text = format_output(final_data)
    
    with open("final_transcript_local.txt", "w", encoding="utf-8") as f:
        f.write(readable_text)
        
    print("\nDone! Check 'final_transcript_local.txt'")

if __name__ == "__main__":
    VIDEO_ID = "BvWOje46Xp8"
    OUTPUT_PATH = '/var/home/magpie/Development/SerialKiller_2/output'
    HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

    segments = run_pyannote_diarization('/var/home/magpie/Development/SerialKiller_2/output/BvWOje46Xp8/chunks/chunk_000/pyavi/audio.wav',HF_TOKEN)
    print(segments)
    