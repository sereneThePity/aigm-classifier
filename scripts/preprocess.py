"""Simplified preprocessing for two data loading approaches: cached specs and neural codec latents."""
import os
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from typing import Tuple
from torch.utils.data import default_collate
from utils import normalize_spectrogram, normalize_audio, apply_highpass_filter


# ===== Shared audio preprocessing pipeline =====
# Single source of truth for audio-level preprocessing: precompute_spectrograms.py,
# encode_latents.py, and evaluate_model.py all call these two functions so training
# and evaluation stay aligned.

def load_and_preprocess_audio(
    file_path,
    sr=16000,
    segment_duration=5.0,
    target_loudness=-20.0,
    hp_freq=20,
    trim_top_db=40,
    crop="random",
    transform_fn=None
):
    """
    Load and preprocess a single audio file: load -> resample -> loudness-normalize ->
    trim silence -> crop/pad to a fixed segment -> optional transform -> high-pass filter.

    Args:
        file_path: Path to audio file
        sr: Target sample rate
        segment_duration: Fixed segment length in seconds
        target_loudness: Target loudness in dB for normalize_audio
        hp_freq: High-pass filter cutoff frequency in Hz
        trim_top_db: Threshold (dB) below reference to consider as silence for trimming
        crop: 'random' for a random segment (training) or 'center' for a deterministic
            center crop (evaluation)
        transform_fn: Optional (audio, sr) -> audio hook applied after crop/pad and
            before the high-pass filter (e.g. pitch shift, time stretch)

    Returns:
        Preprocessed waveform (np.ndarray), or None if the file failed to load.
    """
    import librosa
    import warnings

    if crop not in ("random", "center"):
        raise ValueError(f"Unknown crop mode: {crop!r}. Use 'random' or 'center'.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y, loaded_sr = librosa.load(file_path, sr=None, mono=True)

        if loaded_sr != sr:
            y = librosa.resample(y, orig_sr=loaded_sr, target_sr=sr)

        y = normalize_audio(y, method='db', target=target_loudness)
        y, _ = librosa.effects.trim(y, top_db=trim_top_db)

        segment_samples = int(segment_duration * sr)
        if len(y) >= segment_samples:
            max_start = len(y) - segment_samples
            start_idx = np.random.randint(0, max_start + 1) if crop == "random" else max_start // 2
            y = y[start_idx:start_idx + segment_samples]
        else:
            y = np.pad(y, (0, segment_samples - len(y)), mode='constant')

        if transform_fn is not None:
            y = transform_fn(y, sr)

        y = apply_highpass_filter(y, sr, cutoff_freq=hp_freq)

        return y
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def audio_to_mel_spectrogram(audio, sr=16000, n_mels=128, resize_to=(128, 128)):
    """
    Convert a waveform to a normalized mel-spectrogram: melspec -> power_to_db ->
    mean/std normalize -> optional resize -> add channel dimension.

    Args:
        audio: Waveform array
        sr: Sample rate
        n_mels: Number of mel bins
        resize_to: (freq, time) shape to resize to via zoom, or None to keep the
            natural time dimension (used by the cached-spectrogram training path)

    Returns:
        Spectrogram of shape (1, n_mels, time) as float32.
    """
    import librosa
    from scipy.ndimage import zoom

    spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    spec_db = librosa.power_to_db(spec, ref=np.max)
    spec_db = normalize_spectrogram(spec_db)

    if resize_to is not None:
        zoom_factors = (resize_to[0] / spec_db.shape[0], resize_to[1] / spec_db.shape[1])
        spec_db = zoom(spec_db, zoom_factors, order=1)

    return np.expand_dims(spec_db.astype(np.float32), axis=0)


def collate_fn_skip_none(batch):
    """Custom collate function that filters out None values from batch."""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)


class CachedSpectrogramDataset(torch.utils.data.Dataset):
    """Fast dataset that loads pre-computed spectrograms from disk."""
    
    def __init__(self, manifest_path, split_indices=None):
        """
        Args:
            manifest_path: Path to CSV manifest with 'label' and 'spectrogram_path' columns
            split_indices: Indices to use for this split (None = use all)
        """
        self.df = pd.read_csv(manifest_path)
        if split_indices is not None:
            self.df = self.df.iloc[split_indices].reset_index(drop=True)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            spec_path = row['spectrogram_path']
            label = int(row['label'])
            
            # Load pre-computed spectrogram
            spec_db = np.load(spec_path)  # (1, 128, time_steps)
            
            return torch.from_numpy(spec_db).float(), torch.tensor(label, dtype=torch.long)
        except Exception as e:
            # Skip files with errors (missing, corrupted, etc.)
            return None


def load_cached_spectrograms_for_training(
    manifest_path,
    num_samples=None,
    test_split=0.10,
    val_split=0.10,
    random_state=42
):
    """
    Load pre-computed spectrograms from cached manifest and prepare train/val/test datasets.
    
    Args:
        manifest_path: Path to manifest CSV with 'label' and 'spectrogram_path' columns
        num_samples: Max samples to load (None for all)
        test_split: Test set fraction
        val_split: Validation set fraction
        random_state: Random seed for reproducibility
    
    Returns:
        train_dataset, val_dataset, test_dataset: CachedSpectrogramDataset objects
        input_shape: Shape of input features (1, 128, mel_bins)
    """
    # Read manifest
    df = pd.read_csv(manifest_path)
    if num_samples is not None:
        df = df.sample(n=min(num_samples, len(df)), random_state=random_state)
    
    print(f"Found {len(df)} samples in cached spectrogram manifest")
    
    # Get labels for stratified split
    labels = df['label'].values
    all_indices = np.arange(len(df))
    
    # Split indices
    train_idx, temp_idx, _, temp_labels = train_test_split(
        all_indices, labels, test_size=test_split + val_split,
        random_state=random_state, stratify=labels
    )
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx, temp_labels, test_size=test_split / (test_split + val_split),
        random_state=random_state, stratify=temp_labels
    )
    
    print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}\n")
    
    # Create cached spectrogram datasets
    train_dataset = CachedSpectrogramDataset(manifest_path, split_indices=train_idx)
    val_dataset = CachedSpectrogramDataset(manifest_path, split_indices=val_idx)
    test_dataset = CachedSpectrogramDataset(manifest_path, split_indices=test_idx)
    
    # Input shape for model (1, 128, mel_bins)
    input_shape = (1, 128, 128)
    
    return train_dataset, val_dataset, test_dataset, input_shape


def load_spectrogram_latents_for_training(
    latent_dir,
    num_samples_per_class=None,
    test_split=0.10,
    val_split=0.10,
    random_state=42
):
    """
    Load mel-spectrogram latents from codec subdirectories for 2D CNN training.
    
    Args:
        latent_dir: Path to spectrogram directory (contains codec subdirs with class/file.npy)
        num_samples_per_class: Max samples to load per class per codec (None for all)
        test_split: Test set fraction
        val_split: Validation set fraction
        random_state: Random seed for reproducibility
    
    Returns:
        train_dataset, val_dataset, test_dataset: TensorDataset objects
        input_shape: Shape of input features (should be (1, 128, 128) for spectrograms)
    """
    all_data = []
    all_labels = []
    
    if not os.path.exists(latent_dir):
        raise ValueError(f"Latent directory not found: {latent_dir}")
    
    # Find all codec subdirectories
    codecs = sorted([d for d in os.listdir(latent_dir) 
                     if os.path.isdir(os.path.join(latent_dir, d))])
    
    if not codecs:
        raise ValueError(f"No codec subdirectories found in {latent_dir}")
    
    print(f"Found codecs: {codecs}")
    
    # Load from each codec's class subdirectories
    for codec_name in codecs:
        codec_path = os.path.join(latent_dir, codec_name)
        print(f"Loading from {codec_name}...")
        
        for class_label in range(2):
            class_dir = os.path.join(codec_path, str(class_label))
            if not os.path.exists(class_dir):
                continue
            
            npy_files = sorted([f for f in os.listdir(class_dir) if f.endswith('.npy')])
            if num_samples_per_class is not None:
                npy_files = npy_files[:num_samples_per_class]
            
            for npy_file in tqdm(npy_files, desc=f"  {codec_name} class {class_label}", leave=False):
                try:
                    npy_path = os.path.join(class_dir, npy_file)
                    spec = np.load(npy_path)
                    all_data.append(spec)
                    all_labels.append(class_label)
                except Exception:
                    pass
    
    if not all_data:
        raise ValueError("No spectrogram files found")
    
    # Stack data (should all be (128, 128) already)
    data = np.array(all_data, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)
    
    print(f"\nLoaded {len(data)} samples, shape: {data.shape}")
    
    # Add channel dimension: (N, 128, 128) → (N, 1, 128, 128)
    if data.ndim == 3:
        data = np.expand_dims(data, axis=1)
    
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


def load_encoded_latents_for_training(
    latent_dir,
    num_samples_per_class=25,
    test_split=0.10,
    val_split=0.10,
    random_state=42
):
    """
    Load encoded latents from neural codec subdirectories for 1D CNN training.
    
    Args:
        latent_dir: Path to encoded_latents directory (contains codec subdirs like encodec, dac, etc.)
        num_samples_per_class: Max samples to load per class per codec (None for all)
        test_split: Test set fraction
        val_split: Validation set fraction
        random_state: Random seed for reproducibility
    
    Returns:
        train_dataset, val_dataset, test_dataset: TensorDataset objects
        input_shape: Shape of input features
    """
    all_data = []
    all_labels = []
    
    if not os.path.exists(latent_dir):
        raise ValueError(f"Latent directory not found: {latent_dir}")
    
    # Find all encoder subdirectories
    encoders = sorted([d for d in os.listdir(latent_dir) 
                       if os.path.isdir(os.path.join(latent_dir, d))])
    
    if not encoders:
        raise ValueError(f"No encoder subdirectories found in {latent_dir}")
    
    print(f"Found encoders: {encoders}")
    
    # Load from each encoder's class subdirectories
    for encoder_name in encoders:
        encoder_path = os.path.join(latent_dir, encoder_name)
        print(f"Loading from {encoder_name}...")
        
        for class_label in range(2):
            class_dir = os.path.join(encoder_path, str(class_label))
            if not os.path.exists(class_dir):
                continue
            
            npy_files = sorted([f for f in os.listdir(class_dir) if f.endswith('.npy')])
            if num_samples_per_class is not None:
                npy_files = npy_files[:num_samples_per_class]
            
            for npy_file in tqdm(npy_files, desc=f"  {encoder_name} class {class_label}", leave=False):
                try:
                    npy_path = os.path.join(class_dir, npy_file)
                    latent = np.load(npy_path)
                    all_data.append(latent)
                    all_labels.append(class_label)
                except Exception:
                    pass
    
    if not all_data:
        raise ValueError("No latent files found")
    
    # Find max length and pad
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


def load_all_from_manifest(manifest_path, num_samples=None, use_cached=True, sample_rate=16000):
    """
    Load all samples from a manifest for evaluation.
    Prefers cached spectrograms if available, falls back to raw audio.
    Skips files with codec errors or loading issues.
    
    Args:
        manifest_path: Path to manifest CSV
        num_samples: Max samples to load (None for all)
        use_cached: Try to load from cached spectrograms first
    
    Returns:
        Tuple of (X, y) where X is (N, 1, 128, time) and y is (N,)
    """
    df = pd.read_csv(manifest_path)
    
    if num_samples is not None:
        df = df.sample(n=min(num_samples, len(df)), random_state=42)
    
    X_list = []
    y_list = []
    skipped_count = 0
    total_count = len(df)
    
    # Check if cached spectrograms are available
    if use_cached and 'spectrogram_path' in df.columns:
        print(f"Loading {total_count} samples from cached spectrograms...")
        for idx, row in tqdm(df.iterrows(), total=total_count, desc="Loading cached specs", leave=False):
            try:
                spec_path = row['spectrogram_path']
                if os.path.exists(spec_path):
                    spec = np.load(spec_path)  # (1, 128, time_steps)
                    label = int(row['label'])
                    X_list.append(spec)
                    y_list.append(label)
                else:
                    skipped_count += 1
            except Exception:
                skipped_count += 1
                continue
    else:
        # Fall back to loading raw audio and creating spectrograms
        print(f"Loading {total_count} samples from raw audio files...")
        for idx, row in tqdm(df.iterrows(), total=total_count, desc="Processing audio", leave=False):
            try:
                filepath = row['filepath']
                if not os.path.exists(filepath):
                    skipped_count += 1
                    continue
                
                # Same pipeline used to build training data; center crop for determinism
                audio = load_and_preprocess_audio(filepath, sr=sample_rate, crop="center")
                if audio is None:
                    skipped_count += 1
                    continue
                
                # Kept at natural time length; padded to a common length below
                spec_db = audio_to_mel_spectrogram(audio, sr=sample_rate, resize_to=None)
                
                label = int(row['label'])
                X_list.append(spec_db)
                y_list.append(label)
            except Exception:
                skipped_count += 1
                continue
    
    if not X_list:
        raise ValueError("No valid samples loaded from manifest")
    
    # Pad all spectrograms to same time dimension (128)
    max_time = max(x.shape[2] for x in X_list)
    target_time = max(128, min(max_time, 256))  # Cap at 256 to avoid huge tensors
    
    X_padded = []
    for spec in X_list:
        if spec.shape[2] < target_time:
            pad_width = ((0, 0), (0, 0), (0, target_time - spec.shape[2]))
            spec = np.pad(spec, pad_width, mode='constant')
        elif spec.shape[2] > target_time:
            spec = spec[:, :, :target_time]
        X_padded.append(spec)
    
    X = np.array(X_padded, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    
    print(f"Loaded {len(X)} samples with shape {X.shape} (skipped {skipped_count}/{total_count} problematic files)")
    return X, y


def load_all_from_manifest_with_transforms(manifest_path, target_shape=(128, 128), n_mels=128, transform="random"):
    """
    Load all samples from manifest with audio augmentation/transforms applied.
    Uses raw audio files and applies transforms before creating spectrograms.
    Skips files with codec errors or loading issues.
    
    Args:
        manifest_path: Path to manifest CSV
        target_shape: Target (freq, time) shape for spectrograms
        n_mels: Number of mel bins
        transform: Type of transform ('random', 'pitch_shift', 'time_stretch', etc.)
    
    Returns:
        Tuple of (X, y) where X is (N, 1, freq, time) and y is (N,)
    """
    import librosa

    def apply_transform(audio, sr):
        if transform == "pitch_shift":
            return librosa.effects.pitch_shift(audio, sr=sr, n_steps=np.random.randint(-2, 3))
        elif transform == "time_stretch":
            return librosa.effects.time_stretch(audio, rate=np.random.uniform(0.9, 1.1))
        elif transform == "random":
            tf = np.random.choice(['pitch', 'stretch', 'none'])
            if tf == 'pitch':
                return librosa.effects.pitch_shift(audio, sr=sr, n_steps=np.random.randint(-1, 2))
            elif tf == 'stretch':
                return librosa.effects.time_stretch(audio, rate=np.random.uniform(0.95, 1.05))
        return audio

    df = pd.read_csv(manifest_path)
    
    X_list = []
    y_list = []
    skipped_count = 0
    total_count = len(df)
    
    print(f"Loading {total_count} samples with transforms...")
    for idx, row in tqdm(df.iterrows(), total=total_count, desc="Processing audio with transforms"):
        try:
            filepath = row['filepath']
            if not os.path.exists(filepath):
                skipped_count += 1
                continue
            
            # Same pipeline used to build training data; center crop for determinism
            audio = load_and_preprocess_audio(filepath, sr=16000, crop="center", transform_fn=apply_transform)
            if audio is None:
                skipped_count += 1
                continue
            
            spec_tensor = audio_to_mel_spectrogram(audio, sr=16000, n_mels=n_mels, resize_to=target_shape)
            
            label = int(row['label'])
            X_list.append(spec_tensor)
            y_list.append(label)
        except Exception:
            skipped_count += 1
            continue
    
    if not X_list:
        print("❌ No valid samples processed with transforms.")
        return np.array([]), np.array([])
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    
    print(f"Loaded {len(X)} samples with transforms, shape {X.shape} (skipped {skipped_count}/{total_count} problematic files)")
    return X, y
