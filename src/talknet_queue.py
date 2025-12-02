import os
import subprocess
import glob

talknet_dir = '/var/home/magpie/Development/SerialKiller_2/talknet/TalkNet-ASD'

def get_chunks(chunk_folder):
    video_files = sorted(glob.glob(f"{chunk_folder}/chunk_*.mp4"))

    return video_files

def process_queue(video_files):
    chunk_folders = []
    for video in video_files:
        print(f"--- Processing {video} ---")
        video_folder = os.path.dirname(video)
        video_name = os.path.splitext(os.path.basename(video))[0]

        cmd = f"python demoTalkNet.py --videoFolder {video_folder} --videoName {video_name}"
        
        subprocess.run(cmd,cwd=talknet_dir, shell=True)
        
        chunk_folders.append(f'{video_folder}/{video_name}')
    return chunk_folders

def talknet_queue(chunk_folder):
    video_files = get_chunks(chunk_folder)
    process_queue(video_files)
    return video_files

def get_chunk_folders(video_files):
    chunk_folders = []
    for video in video_files:
        print(f"--- Processing {video} ---")
        video_folder = os.path.dirname(video)
        video_name = os.path.splitext(os.path.basename(video))[0]
        chunk_folders.append(f'{video_folder}/{video_name}')
    return chunk_folders


if __name__ == '__main__':
    video_files = get_chunks('/var/home/magpie/Development/SerialKiller_2/output/BvWOje46Xp8/chunks')
    chunk_folders = get_chunk_folders(video_files)
    print(chunk_folders)