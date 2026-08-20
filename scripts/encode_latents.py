"""
Encode audio through neural codecs with equal dataset split.
Splits dataset randomly and equally among multiple codecs.
Processes each codec in parallel with a dedicated worker process.
"""

import torch
import librosa
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool, Manager
import os
from functools import partial
from utils import normalize_audio, apply_highpass_filter
from neural_codec_confounders import MetaEnCodecWrapper, DACWrapper, GriffinMelCodec

CODECS = {
    "encodec": MetaEnCodecWrapper,
    "dac": DACWrapper,
    "griffin": GriffinMelCodec
}

# Categorize codecs by device type
CPU_CODECS = {"griffin"}
GPU_CODECS = {"encodec", "dac"}

def preprocess_audio(
    file_path,
    sr=16000,
    segment_duration=5.0,
    target_loudness=-20.0,
    hp_freq=20
):
    """
    Follow preprocessing pipeline from preprocess.py:
    1. Load audio
    2. Resample to target sr
    3. Loudness normalize
    4. Trim silence
    5. Random crop to fixed-length segment
    6. High-pass filter
    """
    try:
        # 1. Load audio
        y, loaded_sr = librosa.load(file_path, sr=None, mono=True)
        
        # 2. Resample
        if loaded_sr != sr:
            y = librosa.resample(y, orig_sr=loaded_sr, target_sr=sr)
        
        # 3. Loudness normalize
        y = normalize_audio(y, method='db', target=target_loudness)
        
        # 4. Trim silence
        y, _ = librosa.effects.trim(y, top_db=40)
        
        # 5. Random crop to fixed-length segment
        segment_samples = int(segment_duration * sr)
        if len(y) >= segment_samples:
            max_start = len(y) - segment_samples
            start_idx = np.random.randint(0, max_start + 1)
            y = y[start_idx:start_idx + segment_samples]
        else:
            pad_width = segment_samples - len(y)
            y = np.pad(y, (0, pad_width), mode='constant')
        
        # 6. High-pass filter
        y = apply_highpass_filter(y, sr, cutoff_freq=hp_freq)
        
        return y
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def encode_decode_save(audio_file, output_path, codec):
    """Encode and decode audio through codec, save decoded audio."""
    try:
        # Preprocess
        audio = preprocess_audio(audio_file)
        if audio is None:
            return False
        
        # Use codec's process_audio method which handles encode-decode cycle
        # This ensures type compatibility across all codec types
        decoded_audio = codec.process_audio(audio)
        
        # Save decoded audio as numpy
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, decoded_audio)
        
        return True
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return False

def assign_codecs(df, codec_names):
    """
    Assign codecs equally and randomly to files.
    
    Args:
        df: DataFrame with file list
        codec_names: List of codec names
    
    Returns:
        List of assigned codec names (one per file)
    """
    num_files = len(df)
    num_codecs = len(codec_names)
    
    # Repeat codec list and shuffle
    codec_assignments = (codec_names * ((num_files // num_codecs) + 1))[:num_files]
    np.random.shuffle(codec_assignments)
    
    return codec_assignments


def process_codec_worker(codec_name, df, output_dir, CODECS, codec_kwargs=None):
    """
    Worker function to process all files for a single codec.
    Called in parallel for each codec.
    
    Args:
        codec_name: Name of codec to process
        df: Full DataFrame with codec assignments
        output_dir: Output directory path
        CODECS: Dictionary of codec classes
        codec_kwargs: Dict of codec-specific kwargs (e.g., {"encodec": {"bandwidth": 6}})
    
    Returns:
        Tuple of (codec_name, success_count, total_count)
    """
    if codec_kwargs is None:
        codec_kwargs = {}
    
    
    # Get codec-specific parameters
    kwargs = codec_kwargs.get(codec_name, {})
    
    # Load codec in this process with 16kHz sample rate (matches audio preprocessing)
    codec_class = CODECS[codec_name]
    codec = codec_class(sr=16000, **kwargs)
    
    # Only call eval() if codec has a model attribute (PyTorch models)
    if hasattr(codec, 'model'):
        codec.model.eval()
    
    # Filter files for this codec
    codec_files = df[df["assigned_codec"] == codec_name]
    
    success_count = 0
    total_count = len(codec_files)
    
    # Process all files for this codec
    for _, row in tqdm(codec_files.iterrows(), total=total_count, desc=f"{codec_name}"):
        file_path = row["filepath"]
        label = row["label"]
        
        output_path = output_dir / codec_name / str(label) / f"{Path(file_path).stem}.npy"
        
        success = encode_decode_save(file_path, output_path, codec)
        if success:
            success_count += 1
    
    return codec_name, success_count, total_count

if __name__ == "__main__":
    import argparse
    import multiprocessing
    
    # Required for CUDA with multiprocessing: use 'spawn' instead of 'fork'
    multiprocessing.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Path to CSV manifest")
    parser.add_argument("--output_dir", required=True, help="Output directory for latents")
    parser.add_argument("--codecs", nargs="+", default=["encodec", "dac", "griffin"], 
                        help="List of codecs to use (real models only: encodec, dac, griffin)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel codec workers (default: number of codecs)")
    parser.add_argument("--encodec_bw", type=float, default=1.5,
                        help="EnCodec bandwidth in kbps (1.5, 3, 6, 12, 24) - default: 1.5 (heavy compression)")
    args = parser.parse_args()
    
    # Load manifest
    df = pd.read_csv(args.manifest)
    output_dir = Path(args.output_dir)
    
    df_filtered = df.reset_index(drop=True)
    
    # Filter out files that already exist in output directory (any codec)
    all_codecs = ["encodec", "dac", "griffin"]
    def file_already_processed(row):
        file_stem = Path(row["filepath"]).stem
        label = str(row["label"])
        for codec in all_codecs:
            output_path = output_dir / codec / label / f"{file_stem}.npy"
            if output_path.exists():
                return True
        return False
    
    df_filtered = df_filtered[~df_filtered.apply(file_already_processed, axis=1)]
    df_filtered = df_filtered.reset_index(drop=True)
    
    # Assign codecs randomly and equally to remaining files
    codec_assignments = assign_codecs(df_filtered, args.codecs)
    df_filtered["assigned_codec"] = codec_assignments
    
    print(f"📊 Dataset split ({len(df_filtered)} files to process):")
    for codec in args.codecs:
        count = (df_filtered["assigned_codec"] == codec).sum()
        if count > 0:
            print(f"   {codec}: {count} files")
    
    # Separate codecs by type
    cpu_codecs = [c for c in args.codecs if c in CPU_CODECS]
    gpu_codecs = [c for c in args.codecs if c in GPU_CODECS]
    
    # Prepare codec-specific parameters
    codec_kwargs = {}
    if args.encodec_bw:
        codec_kwargs["encodec"] = {"bandwidth": args.encodec_bw}
    
    print(f"📚 Codec categorization: {len(cpu_codecs)} CPU, {len(gpu_codecs)} GPU")
    if cpu_codecs:
        print(f"   CPU codecs: {', '.join(cpu_codecs)}")
    if gpu_codecs:
        print(f"   GPU codecs: {', '.join(gpu_codecs)}")
    if codec_kwargs:
        print(f"   Codec parameters: {codec_kwargs}")
    
    results = []
    
    # Process GPU codecs with 2 parallel processes
    if gpu_codecs:
        num_gpu_workers = min(2, len(gpu_codecs))
        print(f"\n🔄 Processing {len(gpu_codecs)} GPU codecs with {num_gpu_workers} parallel GPU processes...")
        with Pool(num_gpu_workers) as pool:
            worker_fn = partial(process_codec_worker, df=df_filtered, output_dir=output_dir, CODECS=CODECS, codec_kwargs=codec_kwargs)
            gpu_results = pool.map(worker_fn, gpu_codecs)
            results.extend(gpu_results)
    

    # Process CPU codecs with worker pool
    if cpu_codecs:
        workers = args.workers if args.workers else len(cpu_codecs)
        print(f"\n🔄 Processing {len(cpu_codecs)} CPU codecs with {workers} parallel workers...")
        with Pool(workers) as pool:
            worker_fn = partial(process_codec_worker, df=df_filtered, output_dir=output_dir, CODECS=CODECS, codec_kwargs=codec_kwargs)
            cpu_results = pool.map(worker_fn, cpu_codecs)
            results.extend(cpu_results)
    

    # Print results
    print("\n✅ Processing complete!")
    print(f"\n📈 Results:")
    total_success = 0
    total_files = 0
    for codec_name, success_count, total_count in results:
        total_success += success_count
        total_files += total_count
        success_rate = (success_count / total_count * 100) if total_count > 0 else 0
        print(f"   {codec_name}: {success_count}/{total_count} ({success_rate:.1f}%)")
    
    print(f"\n   Total: {total_success}/{total_files} ({total_success/total_files*100:.1f}%)")
