"""
Convert encoded latent files to mel-spectrograms for 2D CNN training.

Processes .npy latent files from data/encoded_latents/ (preprocessed audio waveforms)
and extracts mel-spectrograms, saving them to data/encoded_trainset/ in the same
codec/class structure. Output is ready for training a 2D CNN classifier.

Usage:
    python decode_latents_to_audio.py
    python decode_latents_to_audio.py --sr 16000 --output data/encoded_trainset
    python decode_latents_to_audio.py --subset encodec/0 --n-mels 128 --workers 4
"""

import numpy as np
import librosa
from pathlib import Path
import argparse
from tqdm import tqdm
from multiprocessing import Pool


def normalize_spectrogram(mel_spec_db, mean=None, std=None):
    """Normalize mel spectrogram to zero mean and unit variance."""
    if mean is None:
        mean = mel_spec_db.mean()
    if std is None:
        std = mel_spec_db.std() + 1e-7
    return (mel_spec_db - mean) / std


def extract_mel_spectrogram(audio, sr=16000, n_mels=128):
    """Extract and normalize mel spectrogram from audio array."""
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return normalize_spectrogram(mel_spec_db)


def pad_or_crop_spectrogram(mel_spec_db, target_shape=(128, 128)):
    """Pad or crop spectrogram to target shape (128 mel bins × 128 time bins)."""
    # Frequency dimension (mel bins)
    if mel_spec_db.shape[0] < target_shape[0]:
        pad_width = target_shape[0] - mel_spec_db.shape[0]
        mel_spec_db = np.pad(mel_spec_db, ((0, pad_width), (0, 0)), mode='constant')
    else:
        mel_spec_db = mel_spec_db[:target_shape[0], :]
    
    # Time dimension
    if mel_spec_db.shape[1] < target_shape[1]:
        pad_width = target_shape[1] - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_spec_db = mel_spec_db[:, :target_shape[1]]
    
    return mel_spec_db


def process_file(npy_file, rel_path, output_dir, sr, n_mels, target_shape, skip_existing):
    """Core file processing logic."""
    try:
        output_path = Path(output_dir)
        output_subdir = output_path / rel_path
        output_subdir.mkdir(parents=True, exist_ok=True)
        output_file = output_subdir / f"{npy_file.stem}.npy"
        
        if output_file.exists() and skip_existing:
            return None
        
        audio = np.load(npy_file)
        if audio.ndim > 1:
            audio = audio.squeeze()
        
        mel_spec = extract_mel_spectrogram(audio, sr=sr, n_mels=n_mels)
        mel_spec = pad_or_crop_spectrogram(mel_spec, target_shape)
        np.save(output_file, mel_spec.astype(np.float32))
        
        return npy_file.name
    except Exception as e:
        print(f"Error: {npy_file.name}: {e}")
        return None


def process_file_wrapper(task):
    """Wrapper that unpacks a single tuple argument for imap_unordered."""
    npy_file, rel_path, output_dir, sr, n_mels, target_shape, skip_existing = task
    return process_file(npy_file, rel_path, output_dir, sr, n_mels, target_shape, skip_existing)


def decode_latents_to_spectrograms(
    input_dir="data/encoded_latents",
    output_dir="data/encoded_trainset",
    sr=16000,
    n_mels=128,
    target_shape=(128, 128),
    subset=None,
    skip_existing=True,
    num_workers=1
):
    """
    Convert npy latent files (preprocessed audio) to mel-spectrogram images for 2D CNN.
    
    Args:
        input_dir: Source directory containing latent files (codec/class/file.npy)
        output_dir: Destination directory for saved spectrograms (codec/class/file.npy)
        sr: Sample rate (Hz)
        n_mels: Number of mel frequency bins
        target_shape: Target spectrogram shape (mel_bins, time_bins) for 2D CNN input
        subset: If set, only process this codec/class (e.g., "encodec/0")
        skip_existing: Skip files that already exist in output directory
        num_workers: Number of parallel worker processes (default: 1)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Collect all npy files
    npy_files = []
    
    if subset:
        # Process specific codec/class
        subset_path = input_path / subset
        if not subset_path.exists():
            raise FileNotFoundError(f"Subset path not found: {subset_path}")
        npy_files = list(subset_path.glob("*.npy"))
        npy_files = [(f, subset) for f in npy_files]
    else:
        # Process all codecs and classes
        for codec_dir in sorted(input_path.iterdir()):
            if not codec_dir.is_dir():
                continue
            
            for class_dir in sorted(codec_dir.iterdir()):
                if not class_dir.is_dir():
                    continue
                
                for npy_file in sorted(class_dir.glob("*.npy")):
                    rel_path = npy_file.relative_to(input_path)
                    npy_files.append((npy_file, str(rel_path.parent)))
    
    print(f"Found {len(npy_files)} latent files to process")
    print(f"Parameters: sr={sr}, n_mels={n_mels}, target_shape={target_shape}")
    print(f"Using {num_workers} worker(s)\n")
    
    # Prepare args for each task
    tasks = [
        (npy_file, rel_path, str(output_path), sr, n_mels, target_shape, skip_existing)
        for npy_file, rel_path in npy_files
    ]
    
    # Process files in parallel with proper progress tracking
    processed = 0
    with Pool(num_workers) as pool:
        for result in tqdm(pool.imap_unordered(process_file_wrapper, tasks, chunksize=10),
                          total=len(tasks), desc="Converting to spectrograms"):
            if result is not None:
                processed += 1
    
    skipped = len(npy_files) - processed
    
    print(f"\n✅ Conversion complete!")
    print(f"   Processed: {processed} files")
    print(f"   Skipped: {skipped} files")
    print(f"   Output directory: {output_path.absolute()}")
    print(f"   Output shape per file: {target_shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert encoded latents to mel-spectrograms for 2D CNN training"
    )
    parser.add_argument(
        "--input",
        default="data/encoded_latents",
        help="Input directory with encoded latents (default: data/encoded_latents)"
    )
    parser.add_argument(
        "--output",
        default="data/encoded_trainset",
        help="Output directory for spectrograms (default: data/encoded_trainset)"
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=16000,
        help="Sample rate in Hz (default: 16000)"
    )
    parser.add_argument(
        "--n-mels",
        type=int,
        default=128,
        help="Number of mel frequency bins (default: 128)"
    )
    parser.add_argument(
        "--target-shape",
        type=int,
        nargs=2,
        default=[128, 128],
        help="Target spectrogram shape for 2D CNN input (default: 128 128)"
    )
    parser.add_argument(
        "--subset",
        help="Process only specific codec/class (e.g., 'encodec/0')"
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-process files that already exist"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (default: 1)"
    )
    
    args = parser.parse_args()
    
    decode_latents_to_spectrograms(
        input_dir=args.input,
        output_dir=args.output,
        sr=args.sr,
        n_mels=args.n_mels,
        target_shape=tuple(args.target_shape),
        subset=args.subset,
        skip_existing=not args.no_skip,
        num_workers=args.workers
    )
