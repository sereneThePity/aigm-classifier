import os
import numpy as np
import librosa
import pandas as pd
import torch
from tqdm import tqdm
from multiprocessing import Pool
from utils import (
    normalize_audio,
    apply_highpass_filter,
    normalize_spectrogram
)
from neural_codec_confounders import NeuralCodecConfounder


# ===== Audio Preprocessing Pipeline =====

def load_and_prep_audio(
    file_path,
    sr=44100,
    segment_duration=5.0,
    target_loudness=-20.0,
    hp_freq=20,
    codec_name=None,
    codec_confounder=None
):
    """Load and preprocess audio: resample, normalize, trim, crop, filter, apply codec."""
    try:
        # Load audio
        y, loaded_sr = librosa.load(file_path, sr=None, mono=True)
        
        # Resample if needed
        if loaded_sr != sr:
            y = librosa.resample(y, orig_sr=loaded_sr, target_sr=sr)
        
        # Loudness normalize
        y = normalize_audio(y, method='db', target=target_loudness)
        
        # Trim silence
        y, _ = librosa.effects.trim(y, top_db=40)
        
        # Random crop to fixed-length segment
        segment_samples = int(segment_duration * sr)
        if len(y) >= segment_samples:
            start_idx = np.random.randint(0, len(y) - segment_samples + 1)
            y = y[start_idx:start_idx + segment_samples]
        else:
            y = np.pad(y, (0, segment_samples - len(y)), mode='constant')
        
        # High-pass filter
        y = apply_highpass_filter(y, sr, cutoff_freq=hp_freq)
        
        # Apply neural codec if specified
        if codec_name is not None and codec_confounder is not None:
            codec_audio = codec_confounder.apply_codec(y, codec_name)
            if codec_audio is not None:
                y = codec_audio
        
        return y
    
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None


def extract_mel_spectrogram(audio, sr=44100, n_mels=128):
    """Extract and normalize mel spectrogram from audio array."""
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return normalize_spectrogram(mel_spec_db)


def pad_or_crop_spectrogram(mel_spec_db, target_shape=(128, 128)):
    """Pad or crop spectrogram to target shape."""
    # Frequency dimension
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


# Module-level codec management for multiprocessing workers
_worker_codec_confounder = None
_worker_codec_cache = {}


def _init_worker(codec_name):
    """Initialize codec confounder once per worker process (called once per worker)."""
    global _worker_codec_confounder, _worker_codec_cache
    
    if codec_name is None:
        _worker_codec_confounder = None
        return
    
    _worker_codec_confounder = NeuralCodecConfounder(sr=44100, init_only=codec_name)
    _worker_codec_cache = {}


def _process_audio_file(args_tuple):
    """Process a single audio file: preprocess, extract mel spectrogram, pad/crop."""
    global _worker_codec_confounder, _worker_codec_cache
    
    filepath, label, segment_duration, target_loudness, hp_freq, n_mels, target_shape, codec_name = args_tuple
    
    try:
        # Initialize codec confounder if needed
        local_codec_confounder = None
        if codec_name is not None:
            if codec_name in _worker_codec_cache:
                local_codec_confounder = _worker_codec_cache[codec_name]
            elif _worker_codec_confounder is not None:
                local_codec_confounder = _worker_codec_confounder
            else:
                local_codec_confounder = NeuralCodecConfounder(sr=44100, init_only=codec_name)
                _worker_codec_cache[codec_name] = local_codec_confounder
        
        # Preprocess audio
        audio = load_and_prep_audio(
            filepath,
            sr=44100,
            segment_duration=segment_duration,
            target_loudness=target_loudness,
            hp_freq=hp_freq,
            codec_name=codec_name,
            codec_confounder=local_codec_confounder
        )
        
        if audio is None:
            return None, None
        
        # Extract mel spectrogram and normalize
        mel_spec_db = extract_mel_spectrogram(audio, sr=44100, n_mels=n_mels)
        
        # Pad or crop to target shape
        mel_spec_db = pad_or_crop_spectrogram(mel_spec_db, target_shape)
        
        return mel_spec_db, label
    
    except Exception as e:
        print(f"⚠️  Error processing {filepath}: {e}")
        return None, None


def scan_latent_files(latent_dir):
    """Scan latent directory and return list of (filepath, label) tuples without loading data."""
    file_list = []
    print(f"   Scanning codec subdirectories...")
    for codec_dir in os.listdir(latent_dir):
        codec_subdir = os.path.join(latent_dir, codec_dir)
        if not os.path.isdir(codec_subdir):
            continue
        for label_dir in os.listdir(codec_subdir):
            label_subdir = os.path.join(codec_subdir, label_dir)
            if not os.path.isdir(label_subdir):
                continue
            try:
                label = int(label_dir)
            except ValueError:
                continue
            for filename in os.listdir(label_subdir):
                if filename.endswith('.npy'):
                    file_list.append((os.path.join(label_subdir, filename), label))
    print(f"   Found {len(file_list)} latent files.")
    return file_list


class LazyLatentDataset(torch.utils.data.Dataset):
    """Lazily loads .npy latent files one at a time to avoid OOM."""

    def __init__(self, file_list, target_shape=(128, 128)):
        """
        Args:
            file_list: list of (filepath, label) tuples
            target_shape: desired 2D shape for each latent
        """
        self.file_list = file_list
        self.target_shape = target_shape

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path, label = self.file_list[idx]
        latent = np.load(path)

        # Handle 1D latent arrays (codec codes) - reshape to 2D
        if latent.ndim == 1:
            size = int(np.prod(self.target_shape))
            if latent.size >= size:
                latent = latent[:size].reshape(self.target_shape)
            else:
                latent = np.pad(latent, (0, size - latent.size))
                latent = latent.reshape(self.target_shape)
        else:
            latent = pad_or_crop_spectrogram(latent, self.target_shape)

        # Add channel dimension: (H, W) -> (1, H, W)
        latent = latent[np.newaxis, ...]
        return torch.from_numpy(latent).float(), label


def load_precomputed_latents(latent_dir, target_shape=(128, 128)):
    """Load all pre-encoded latents (.npy files) from codec subdirectories.
    
    Recursively loads all .npy files from latent_dir/codec/label/ structure.
    Handles both 1D codec codes and 2D mel spectrograms.
    """
    file_list = scan_latent_files(latent_dir)
    
    if not file_list:
        print(f"❌ No latent files found in {latent_dir}")
        return np.array([]), np.array([])
    
    X, y = [], []
    pbar = tqdm(total=len(file_list), desc="Loading latents", unit="file")
    
    for latent_path, label in file_list:
        try:
            latent = np.load(latent_path)
            
            # Handle 1D latent arrays (codec codes) - reshape to 2D
            if latent.ndim == 1:
                size = np.prod(target_shape)
                if latent.size >= size:
                    latent = latent[:size].reshape(target_shape)
                else:
                    latent = np.pad(latent, (0, size - latent.size))
                    latent = latent.reshape(target_shape)
            else:
                latent = pad_or_crop_spectrogram(latent, target_shape)
            
            X.append(latent)
            y.append(label)
        except Exception as e:
            print(f"\n⚠️  Error loading {latent_path}: {e}")
        pbar.update(1)
    
    pbar.close()
    
    if not X:
        print(f"❌ No latents loaded from {latent_dir}")
        return np.array([]), np.array([])
    
    X = np.array(X)[..., np.newaxis]
    y = np.array(y)
    
    print(f"✅ Loaded {len(X)} precomputed latents from {latent_dir}")
    return X, y


def load_dataset_comprehensive(
    manifest_csv, 
    n_mels=128, 
    target_shape=(128, 128),
    segment_duration=5.0,
    target_loudness=-20.0,
    hp_freq=20,
    workers=12,
    latent_mode='random',
    latent_dir=None
):
    """
    Load dataset with preprocessing pipeline.
    
    Two modes:
    1. latent_mode='precomputed': Load pre-encoded latents from latent_dir (ignores manifest_csv)
    2. latent_mode='random' or codec_name: Apply CPU codecs to audio files from manifest_csv in parallel
    
    Args:
        manifest_csv: Path to CSV with 'filepath' and 'label' columns (used when latent_mode != 'precomputed')
        n_mels: Number of mel frequency bins (default 128)
        target_shape: Target shape for latents/spectrograms (freq, time)
        segment_duration: Fixed segment duration in seconds (default 5.0s)
        target_loudness: Target RMS level in dB (default -20 dB)
        hp_freq: High-pass filter frequency in Hz (default 20 Hz)
        workers: Number of processes for multiprocessing
        latent_mode: 'precomputed', 'random' (random CPU codec per sample), or specific codec name
        latent_dir: Directory with precomputed latents (required if latent_mode='precomputed')
    
    Returns:
        X: Array of shape (n_samples, freq, time, 1)
        y: Array of labels
    """
    # Path 1: Use precomputed latents
    if latent_mode == 'precomputed':
        if latent_dir is None:
            raise ValueError("latent_dir must be provided when latent_mode='precomputed'")
        return load_precomputed_latents(latent_dir, target_shape)
    
    # Path 2: Process audio files with codec augmentation
    df = pd.read_csv(manifest_csv)
    X, y = [], []
    
    # Determine codec assignments
    assigned_codecs = None
    if latent_mode == 'random':
        # Create balanced codec assignments across available CPU codecs
        temp_confounder = NeuralCodecConfounder(sr=44100, device_type='cpu')
        available_codecs = temp_confounder.get_available_codecs()
        
        num_files = len(df)
        num_codecs = len(available_codecs)
        repetitions = (num_files + num_codecs - 1) // num_codecs
        
        assigned_codecs = (available_codecs * repetitions)[:num_files]
        np.random.shuffle(assigned_codecs)
    
    # Build argument tuples for multiprocessing
    args_list = []
    for i, (filepath, label) in enumerate(zip(df["filepath"], df["label"])):
        codec = assigned_codecs[i] if assigned_codecs is not None else latent_mode
        args_list.append((filepath, label, segment_duration, target_loudness, hp_freq, n_mels, target_shape, codec))
    
    # Process files in parallel
    init_codec = None if assigned_codecs is not None else latent_mode
    with Pool(workers, initializer=_init_worker, initargs=(init_codec,)) as pool:
        results = list(tqdm(pool.imap(_process_audio_file, args_list), 
                           total=len(df), desc="Processing"))
    
    # Aggregate results
    for mel_spec, label in results:
        if mel_spec is not None and label is not None:
            X.append(mel_spec)
            y.append(label)
    
    X = np.array(X)[..., np.newaxis]  # Add channel dimension (n_samples, freq, time, 1)
    y = np.array(y)
    
    return X, y

