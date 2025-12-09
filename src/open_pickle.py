import os
import pickle

OUTPUT_DIR = '/var/home/magpie/Development/SerialKiller_2/benchmarking/6GzxbrO0DHM'
CHUNKS_DIR = os.path.join(OUTPUT_DIR, 'chunks')
MEMORY_FILE = os.path.join(OUTPUT_DIR, 'global_training_memory.pkl')
FACES_FILE = os.path.join(OUTPUT_DIR, 'global_faces.pkl')

files = [FACES_FILE]

for file in files:
    with open(file, 'rb') as f:
            data = pickle.load(f)
            print(data)
