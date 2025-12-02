import json
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import JSONFormatter
import json

def yt_list(video_id):

    video_url = f'https://www.youtube.com/watch?v={video_id}'
    
    ydl_opts = {
        'listformats': True,  # This option instructs yt-dlp to list formats
        'cookies': '/var/home/magpie/Development/SerialKiller_2/resources/cookies.txt',
    }
    
    

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=False)
        return info_dict
    except Exception as e:
        print(f"Download Error: {e}")
        return None

def get_transcript_from_youtube(video_id):
    ytt_api = YouTubeTranscriptApi()
    try:
        transcript = ytt_api.fetch(video_id)
    except Exception as e:
        raise e

    formatter = JSONFormatter()
    json_formatted = json.loads(formatter.format_transcript(transcript))
    
    return json_formatted

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

def download_video_from_youtube(video_id,output_path):
    """Downloads video from youtube."""
    final_output_path = f"{output_path}/{video_id}/{video_id}_native_video"

    video_url = f'https://www.youtube.com/watch?v={video_id}'
    
    ydl_opts = {
        'outtmpl': final_output_path,
        'cookies': '/var/home/magpie/Development/SerialKiller_2/resources/cookies.txt',
    }
    
    print(f"Downloading Video...")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        return final_output_path
    except Exception as e:
        print(f"Download Error: {e}")
        return None

def get_best_native_format(video_id):
    url = f'https://www.youtube.com/watch?v={video_id}'
    ydl_opts_extraction = {
        'quiet': True,
        'skip_download': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts_extraction) as ydl:
        try:
            print("Extracting metadata...")
            info = ydl.extract_info(url, download=False)
            formats = info.get('formats', [])

            native_formats = []
            for f in formats:
                video_url = f.get('url', '')
                if 'sr=1' in video_url or 'sr%3D1' in video_url:
                    continue
                
                if f.get('vcodec') != 'none':
                    native_formats.append(f)

            if not native_formats:
                print("No native video formats found (odd!).")
                return None

            native_formats.sort(key=lambda x: (
                x.get('height') or 0, 
                x.get('tbr') or 0
            ), reverse=True)

            best_video = native_formats[0]
            print(f"Detected Best Native Resolution: {best_video.get('height')}p (ID: {best_video['format_id']})")
            
            return best_video['format_id']

        except Exception as e:
            print(f"Error extraction info: {e}")
            return None

def download_native_video(video_id,output_path):
    url = f'https://www.youtube.com/watch?v={video_id}'
    target_video_id = get_best_native_format(video_id)
    final_output_path = f"{output_path}/{video_id}/{video_id}_video.mp4"


    if not target_video_id:
        print("Could not determine a target format.")
        return

    final_opts = {
        'format': f'{target_video_id}+bestaudio/best',
        'outtmpl': final_output_path,
        'merge_output_format': 'mp4'
    }

    print(f"Downloading with format selector: {final_opts['format']}")
    with yt_dlp.YoutubeDL(final_opts) as ydl:
        ydl.download([url])

if __name__ == '__main__':
    video_id = "6GzxbrO0DHM&t=2s"
    download_native_video(video_id,'/var/home/magpie/Development/SerialKiller_2/output')
    with open('video_dict.json','w') as f:
        json.dump(yt_list(video_id),f,indent=4)


