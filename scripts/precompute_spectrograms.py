"""Pre-compute and cache mel-spectrograms from manifest to speed up training."""
import argparse
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from multiprocessing import Pool
from utils import ROOT_DIR, normalize_audio, apply_highpass_filter


def process_single_file(args):
    """Process a single audio file and save its spectrogram."""
    file_path, output_path, sr, segment_duration, target_loudness, hp_freq = args
    
    try:
        # Load and preprocess audio
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            y, loaded_sr = librosa.load(file_path, sr=None, mono=True)
        
        # Resample
        if loaded_sr != sr:
            y = librosa.resample(y, orig_sr=loaded_sr, target_sr=sr)
        
        # Loudness normalize
        y = normalize_audio(y, method='db', target=target_loudness)
        
        # Trim silence
        y, _ = librosa.effects.trim(y, top_db=40)
        
        # Random crop to fixed-length segment
        segment_samples = int(segment_duration * sr)
        if len(y) >= segment_samples:
            max_start = len(y) - segment_samples
            start_idx = np.random.randint(0, max_start + 1)
            y = y[start_idx:start_idx + segment_samples]
        else:
            pad_width = segment_samples - len(y)
            y = np.pad(y, (0, pad_width), mode='constant')
        
        # High-pass filter
        y = apply_highpass_filter(y, sr, cutoff_freq=hp_freq)
        
        # Compute spectrogram
        spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        spec_db = librosa.power_to_db(spec, ref=np.max)
        
        # Normalize
        spec_db = (spec_db - spec_db.mean()) / (spec_db.std() + 1e-7)
        spec_db = spec_db.astype(np.float32)
        
        # Add channel dimension: (1, 128, time_steps)
        spec_db = np.expand_dims(spec_db, axis=0)
        
        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, spec_db)
        
        return (output_path, True, None)
    except Exception as e:
        return (output_path, False, str(e))


def precompute_spectrograms(
    manifest_path,
    output_dir,
    sr=16000,
    segment_duration=5.0,
    target_loudness=-20.0,
    hp_freq=20,
    num_workers=8
):
    """Pre-compute spectrograms from manifest and cache them."""
    
    df = pd.read_csv(manifest_path)
    print(f"Found {len(df)} samples in manifest\n")
    
    # Create output directory structure
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare processing tasks
    tasks = []
    for idx, row in df.iterrows():
        file_path = row['filepath']
        label = int(row['label'])
        
        # Output: output_dir/{label}/{idx}.npy
        output_path = output_dir / f"{label}" / f"{idx:06d}.npy"
        
        tasks.append((
            file_path,
            str(output_path),
            sr,
            segment_duration,
            target_loudness,
            hp_freq
        ))
    
    print(f"Processing {len(tasks)} files with {num_workers} workers...\n")
    
    # Process in parallel
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_single_file, tasks, chunksize=10),
            total=len(tasks),
            desc="Computing spectrograms"
        ))
    
    # Summary
    successful = sum(1 for _, success, _ in results if success)
    failed = sum(1 for _, success, _ in results if not success)
    
    print(f"\n{'='*70}")
    print(f"✓ Processed: {successful}/{len(tasks)} successfully")
    if failed > 0:
        print(f"✗ Failed: {failed}")
        print("\nFailed files:")
        for output_path, success, error in results:
            if not success:
                print(f"  {output_path}: {error}")
    
    # Save manifest with output paths
    manifest_output = output_dir / "manifest.csv"
    output_df = df.copy()
    output_df['spectrogram_path'] = [
        str(output_dir / f"{int(row['label'])}" / f"{idx:06d}.npy")
        for idx, row in df.iterrows()
    ]
    output_df.to_csv(manifest_output, index=False)
    print(f"✓ Saved manifest to: {manifest_output}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pre-compute mel-spectrograms from manifest")
    parser.add_argument('--manifest', type=str, default=None, 
                       help='Path to input manifest CSV')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save cached spectrograms')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of parallel workers')
    args = parser.parse_args()
    
    if not args.manifest:
        args.manifest = str(Path(ROOT_DIR) / "data/trainset/manifest.csv")
    if not args.output_dir:
        args.output_dir = str(Path(ROOT_DIR) / "data/cached_spectrograms")
    
    print("="*70)
    print("🎵 Pre-computing Mel-Spectrograms")
    print("="*70 + "\n")
    print(f"Input manifest:  {args.manifest}")
    print(f"Output directory: {args.output_dir}")
    print(f"Workers:         {args.workers}\n")
    
    precompute_spectrograms(
        args.manifest,
        args.output_dir,
        num_workers=args.workers
    )
