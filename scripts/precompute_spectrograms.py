"""Pre-compute and cache mel-spectrograms from manifest to speed up training."""
import argparse
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from utils import ROOT_DIR
from preprocess import load_and_preprocess_audio, audio_to_mel_spectrogram


def process_single_file(args):
    """Process a single audio file and save its spectrogram."""
    file_path, output_path, sr, segment_duration, target_loudness, hp_freq = args
    
    try:
        audio = load_and_preprocess_audio(
            file_path, sr=sr, segment_duration=segment_duration,
            target_loudness=target_loudness, hp_freq=hp_freq, crop="random"
        )
        if audio is None:
            return (output_path, False, "failed to load/preprocess audio")
        
        # Kept at natural time length; downstream training pads/resizes per data source
        spec_db = audio_to_mel_spectrogram(audio, sr=sr, resize_to=None)
        
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
