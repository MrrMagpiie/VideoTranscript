import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import os
import glob
import re
import time
from tqdm import tqdm 
from sklearn.decomposition import PCA

# ==========================================
# CONFIGURATION
# ==========================================
OUTPUT_DIR = '/var/home/magpie/Development/SerialKiller_2/benchmarking/6GzxbrO0DHM'
CHUNKS_DIR = os.path.join(OUTPUT_DIR, 'chunks')
MEMORY_FILE = os.path.join(OUTPUT_DIR, 'global_training_memory.pkl')
FACES_FILE = os.path.join(OUTPUT_DIR, 'global_faces.pkl')

# Output paths for your images
PLOT_AUDIO_TSNE = os.path.join(OUTPUT_DIR, 'figure_1a_audio_tsne.png')
PLOT_AUDIO_PCA  = os.path.join(OUTPUT_DIR, 'figure_1b_audio_pca.png')
PLOT_FACE_CLUSTERS = os.path.join(OUTPUT_DIR, 'figure_3_face_clusters.png')
PLOT_TIMELINE_PATH = os.path.join(OUTPUT_DIR, 'figure_2_timeline.png')

# ==========================================
# 1. VISUALIZE MEMORY (The Audio Anchors)
# ==========================================
def plot_memory_clusters(memory_file):
    print(f"\n--- [1/3] Processing Audio Memory ---")
    
    if not os.path.exists(memory_file):
        print(">> Error: Memory file not found.")
        return

    with open(memory_file, 'rb') as f:
        data = pickle.load(f)
    
    X = np.array(data['X']) 
    y = np.array(data['y']) 

    # --- 1. CHECK DIMENSIONS ---
    # This will print: (1585, 256) if it's 256-dim
    print(f">> Data Shape: {X.shape} (Anchors, Dimensions)") 
    print(f">> Unique Speakers: {len(np.unique(y))}")

    # --- 2. t-SNE (Local Clusters) ---
    print(f">> Running t-SNE...")
    perp = min(30, len(X) - 1) 
    tsne = TSNE(n_components=2, random_state=42, perplexity=perp)
    X_tsne = tsne.fit_transform(X)

    # --- 3. PCA (Global Variance) ---
    print(f">> Running PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Calculate how much info is preserved
    var_ratio = pca.explained_variance_ratio_
    print(f">> PCA Explained Variance: {var_ratio[0]+var_ratio[1]:.2%} of data preserved")

   # ---------------------------------------
    # FIGURE 1A: t-SNE (Local Clusters)
    # ---------------------------------------
    print(f">> Generating Figure 1A (t-SNE)...")
    perp = min(30, len(X) - 1) 
    tsne = TSNE(n_components=2, random_state=42, perplexity=perp, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X)

    df_tsne = pd.DataFrame(X_tsne, columns=['x', 'y'])
    df_tsne['label'] = y

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df_tsne, x='x', y='y', hue='label', 
        palette='tab10', s=100, alpha=0.8, edgecolor='w'
    )
    plt.title("Figure 1a: Audio Embeddings t-SNE (Local Identity Clusters)", fontsize=14)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Speakers")
    plt.tight_layout()
    
    print(f"   -> Saving to {PLOT_AUDIO_TSNE}")
    plt.savefig(PLOT_AUDIO_TSNE, dpi=300, bbox_inches='tight')
    plt.close() # clear memory

    # ---------------------------------------
    # FIGURE 1B: PCA (Global Variance)
    # ---------------------------------------
    print(f">> Generating Figure 1B (PCA)...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    var_ratio = pca.explained_variance_ratio_
    explained = var_ratio[0] + var_ratio[1]
    print(f"   -> PCA Explained Variance: {explained:.2%}")

    df_pca = pd.DataFrame(X_pca, columns=['x', 'y'])
    df_pca['label'] = y

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df_pca, x='x', y='y', hue='label', 
        palette='tab10', s=100, alpha=0.8, edgecolor='w'
    )
    plt.title(f"Figure 1b: Audio Embeddings PCA (Global Variance: {explained:.1%})", fontsize=14)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Speakers")
    plt.tight_layout()

    print(f"   -> Saving to {PLOT_AUDIO_PCA}")
    plt.savefig(PLOT_AUDIO_PCA, dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# 2. VISUALIZE FACE CLUSTERS
# ==========================================
def plot_face_clusters():
    print(f"\n--- [2/3] Processing Face Memory ---")
    
    if not os.path.exists(FACES_FILE):
        print(f">> Skipping: No global faces pickle found at {FACES_FILE}")
        return

    print(f"Loading Face Memory from {FACES_FILE}...")
    with open(FACES_FILE, 'rb') as f:
        file = pickle.load(f)
        data = file.get('faces')
    X = []
    y = []

    # Handle data structure
    if isinstance(data, dict):
        for speaker_id, embedding in data.items():
                # Ensure flattening if the array is 2D (1, 2622)
                flat_embedding = np.array(embedding).flatten()
                X.append(flat_embedding)
                y.append(speaker_id)
    elif isinstance(data, list):
        for item in data:
            if 'embedding' in item and 'label' in item:
                X.append(item['embedding'])
                y.append(item['label'])
            elif 'embedding' in item and 'id' in item:
                X.append(item['embedding'])
                y.append(item['id'])
        
    X = np.array(X)
    y = np.array(y)
    if len(X) == 0:
        print(">> Error: Could not extract embeddings from face file.")
        return

    # --- REPORTING ---
    unique_faces = np.unique(y)
    print(f">> Loaded {len(X)} face embeddings.")
    print(f">> Unique Identities: {len(unique_faces)}")

    print(f">> Running t-SNE on faces...")
    start_time = time.time()
    
    perp = min(30, len(X) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perp, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X)

    print(f">> t-SNE finished in {time.time() - start_time:.2f} seconds.")

    df_plot = pd.DataFrame(X_embedded, columns=['x', 'y'])
    df_plot['label'] = y

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df_plot,
        x='x', y='y',
        hue='label',
        palette='tab10',
        s=100,
        alpha=0.8,
        edgecolor='w'
    )
    
    plt.title(f"DeepFace Embeddings: {len(unique_faces)} Identities", fontsize=14)
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Face IDs")
    plt.tight_layout()
    
    print(f">> Saving face clusters to {PLOT_FACE_CLUSTERS}...")
    plt.savefig(PLOT_FACE_CLUSTERS, dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# 3. TIMELINE HELPERS
# ==========================================
def get_sorted_chunk_files(chunks_dir):
    pattern = os.path.join(chunks_dir, "chunk_*", "per_word_transcript.csv")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No transcript files found in {chunks_dir}")
        return []

    def extract_chunk_id(filepath):
        match = re.search(r'chunk_(\d+)', filepath)
        return int(match.group(1)) if match else -1

    return sorted(files, key=extract_chunk_id)

def load_and_stitch_transcripts(files):
    print(f"\n--- [3/3] Stitching Timelines ---")
    print(f">> Found {len(files)} chunk transcripts.")
    
    all_dfs = []
    running_offset = 0.0
    
    # --- PROGRESS BAR 1: LOADING FILES ---
    # using tqdm to show progress of loading files
    for filepath in tqdm(files, desc="Stitching Chunks", unit="file"):
        try:
            df = pd.read_csv(filepath)
            
            if df.empty:
                continue
            
            df['start'] = df['start'] + running_offset
            df['end'] = df['end'] + running_offset
            
            all_dfs.append(df)
            
            local_max = df['end'].max()
            if local_max > running_offset:
                running_offset = local_max
                
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    if not all_dfs:
        return pd.DataFrame()

    global_df = pd.concat(all_dfs, ignore_index=True)
    print(f">> Final Video Duration: {running_offset/60:.2f} minutes")
    print(f">> Total Words Processed: {len(global_df)}")
    return global_df

def plot_global_timeline(df):
    if df.empty:
        print("Error: DataFrame is empty.")
        return

    # --- STATS REPORTING ---
    unique_speakers = sorted(df['speaker'].unique())
    print(f">> Timeline Speakers: {unique_speakers}")
    
    visual_confirms = df[df['visability'] == True].shape[0]
    total_words = df.shape[0]
    ratio = (visual_confirms / total_words) * 100
    print(f">> Visual Confirmation Rate: {ratio:.2f}% ({visual_confirms}/{total_words} words)")

    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_speakers)))
    color_map = dict(zip(unique_speakers, colors))

    total_minutes = df['end'].max() / 60
    plot_width = max(15, int(total_minutes * 2))
    
    fig, ax = plt.subplots(figsize=(plot_width, 8))
    y_positions = {speaker: i for i, speaker in enumerate(unique_speakers)}

    # --- PROGRESS BAR 2: PLOTTING ---
    # Using tqdm here because plotting 10k+ bars can be slow
    print(f">> Drawing Gantt Chart elements...")
    
    # We convert to dict records for slightly faster iteration than iterrows
    records = df.to_dict('records')
    
    for row in tqdm(records, desc="Plotting Words", unit="word"):
        start = row['start']
        duration = row['end'] - row['start']
        speaker = row['speaker']
        hatch = '///' if row['visability'] else None
        
        ax.broken_barh(
            xranges=[(start, duration)], 
            yrange=(y_positions[speaker] - 0.4, 0.8),
            facecolors=color_map[speaker],
            edgecolor='white',
            linewidth=0.5,
            hatch=hatch,
            alpha=0.9
        )

    ax.set_yticks(range(len(unique_speakers)))
    ax.set_yticklabels(unique_speakers)
    ax.set_xlabel("Time (seconds)")
    ax.set_title(f"Dynamic Timeline ({total_minutes:.1f} mins) - Visual Confirm: {ratio:.1f}%", fontsize=16)
    
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(60))
    plt.grid(True, axis='x', which='minor', linestyle=':', alpha=0.5, color='black')
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', label='Audio Only'),
        Patch(facecolor='gray', hatch='///', label='Visual Confirmation')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    print(f">> Saving timeline plot to {PLOT_TIMELINE_PATH}...")
    plt.savefig(PLOT_TIMELINE_PATH, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # 1. AUDIO CLUSTERS
    try:
        plot_memory_clusters(MEMORY_FILE)
    except Exception as e:
        print(f"Skipping Training Cluster Plot: {e}")

    # 2. FACE CLUSTERS
    try:
        plot_face_clusters()
    except Exception as e:
        print(f"Skipping Face Cluster Plot: {e}")

    # 3. TIMELINE
    files = get_sorted_chunk_files(CHUNKS_DIR)
    global_df = load_and_stitch_transcripts(files)
    
    try: 
        plot_global_timeline(global_df)
    except Exception as e:
        print(f"Skipping Timeline Plot: {e}")
        
    print("\n--- DONE ---")