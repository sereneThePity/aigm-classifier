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


# ===== Encoded Latents Loading =====

def load_encoded_latents_for_training(
    latent_dir,
    num_samples_per_class=25,
    test_split=0.15,
    val_split=0.15,
    random_state=42
):
    """
    Load encoded latents from all encoder subdirectories and prepare train/val/test datasets.
    
    Args:
        latent_dir: Path to encoded_latents directory (contains encoder subdirs like encodec, dac, etc.)
        num_samples_per_class: Max samples to load per class per encoder (None for all)
        test_split: Test set fraction
        val_split: Validation set fraction
        random_state: Random seed for reproducibility
    
    Returns:
        train_dataset, val_dataset, test_dataset: TensorDataset objects
        input_shape: Shape of input features
    """
    from sklearn.model_selection import train_test_split
    
    all_data = []
    all_labels = []
    
    if not os.path.exists(latent_dir):
        raise ValueError(f"Latent directory not found: {latent_dir}")
    
    # Find all encoder subdirectories
    encoders = [d for d in os.listdir(latent_dir) 
                if os.path.isdir(os.path.join(latent_dir, d))]
    
    if not encoders:
        raise ValueError(f"No encoder subdirectories found in {latent_dir}")
    
    print(f"Found encoders: {encoders}")
    
    # Load from each encoder's class subdirectories
    for encoder_name in encoders:
        encoder_path = os.path.join(latent_dir, encoder_name)
        print(f"\nLoading from {encoder_name}...")
        
        for class_label in range(2):
            class_dir = os.path.join(encoder_path, str(class_label))
            if not os.path.exists(class_dir):
                continue
            
            npy_files = sorted([f for f in os.listdir(class_dir) if f.endswith('.npy')])
            if num_samples_per_class is not None:
                npy_files = npy_files[:num_samples_per_class]
            
            for npy_file in tqdm(npy_files, desc=f"  {encoder_name} class {class_label}", leave=False):
                try:
                    latent = np.load(os.path.join(class_dir, npy_file))
                    all_data.append(latent)
                    all_labels.append(class_label)
                except Exception as e:
                    pass
    
    if not all_data:
        raise ValueError("No latent files found")
    
    # Find max length and pad all latents to that size
    max_len = max(x.shape[0] if hasattr(x, 'shape') else len(x) for x in all_data)
    print(f"Max latent length: {max_len}")
    
    data_padded = []
    for x in all_data:
        if len(x) < max_len:
            x = np.pad(x, (0, max_len - len(x)), mode='constant')
        data_padded.append(x)
    
    # Stack and normalize
    data = np.array(data_padded, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)
    
    print(f"\nLoaded {len(data)} samples, shape: {data.shape}")
    
    # Add channel dimension if 1D
    if data.ndim == 2:
        data = np.expand_dims(data, axis=1)
    
    # Normalize
    data = (data - data.mean()) / (data.std() + 1e-7)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        data, labels, test_size=test_split + val_split,
        random_state=random_state, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_split / (test_split + val_split),
        random_state=random_state, stratify=y_temp
    )
    
    # Convert to torch
    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long()
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).long()
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_test).long()
    )
    
    print(f"Split: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}\n")
    
    return train_dataset, val_dataset, test_dataset, X_train.shape[1:]


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

