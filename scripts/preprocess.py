import os
import numpy as np
import librosa
import torch
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from typing import Optional, Tuple
from utils import normalize_spectrogram, normalize_audio, apply_highpass_filter


class SpectrogramDataset(torch.utils.data.Dataset):
    """Memory-efficient dataset that computes spectrograms on-the-fly."""
    
    def __init__(self, manifest_path, split_indices=None, sr=44100, segment_duration=5.0, 
                 target_loudness=-20.0, hp_freq=20):
        """
        Args:
            manifest_path: Path to CSV manifest with 'filepath' and 'label' columns
            split_indices: Indices to use for this split (None = use all)
            sr: Sample rate for resampling
            segment_duration: Audio segment duration in seconds
            target_loudness: Target loudness in dB
            hp_freq: Highpass filter cutoff frequency
        """
        self.df = pd.read_csv(manifest_path)
        if split_indices is not None:
            self.df = self.df.iloc[split_indices].reset_index(drop=True)
        
        self.sr = sr
        self.segment_duration = segment_duration
        self.target_loudness = target_loudness
        self.hp_freq = hp_freq
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = row['filepath']
        label = int(row['label'])
        
        # Load and preprocess audio on-the-fly
        audio = self._load_and_preprocess(file_path)
        
        # Compute spectrogram
        spec = librosa.feature.melspectrogram(y=audio, sr=self.sr, n_mels=128)
        spec_db = librosa.power_to_db(spec, ref=np.max)
        
        # Add channel dimension and normalize
        spec_db = np.expand_dims(spec_db, axis=0).astype(np.float32)
        spec_db = (spec_db - spec_db.mean()) / (spec_db.std() + 1e-7)
        
        return torch.from_numpy(spec_db).float(), torch.tensor(label, dtype=torch.long)
    
    def _load_and_preprocess(self, file_path):
        """Load and preprocess audio following the standard pipeline."""
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                y, loaded_sr = librosa.load(file_path, sr=None, mono=True)
            
            # Resample
            if loaded_sr != self.sr:
                y = librosa.resample(y, orig_sr=loaded_sr, target_sr=self.sr)
            
            # Loudness normalize
            y = normalize_audio(y, method='db', target=self.target_loudness)
            
            # Trim silence
            y, _ = librosa.effects.trim(y, top_db=40)
            
            # Random crop to fixed-length segment
            segment_samples = int(self.segment_duration * self.sr)
            if len(y) >= segment_samples:
                max_start = len(y) - segment_samples
                start_idx = np.random.randint(0, max_start + 1)
                y = y[start_idx:start_idx + segment_samples]
            else:
                pad_width = segment_samples - len(y)
                y = np.pad(y, (0, pad_width), mode='constant')
            
            # High-pass filter
            y = apply_highpass_filter(y, self.sr, cutoff_freq=self.hp_freq)
            
            return y
        except Exception:
            # Return silence on error
            return np.zeros(int(self.segment_duration * self.sr))


# ===== Audio Preprocessing =====

def preprocess_audio(
    file_path,
    sr=44100,
    segment_duration=5.0,
    target_loudness=-20.0,
    hp_freq=20
):
    """
    Preprocess audio file following the standard pipeline:
    1. Load audio
    2. Resample to target sr
    3. Loudness normalize
    4. Trim silence
    5. Random crop to fixed-length segment
    6. High-pass filter
    
    Args:
        file_path: Path to audio file
        sr: Target sample rate (default 44100)
        segment_duration: Segment duration in seconds (default 5.0)
        target_loudness: Target loudness in dB (default -20.0)
        hp_freq: High-pass filter cutoff frequency (default 20)
    
    Returns:
        np.ndarray: Preprocessed audio array, or None if error
    """
    try:
        # 1. Load audio - suppress warnings from libmpg123
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
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
    except (OSError, RuntimeError, Exception):
        # OSError: file not found or corrupted
        # RuntimeError: librosa errors
        # Exception: any other error (timeout, etc)
        return None


# ===== Dataset Loaders =====

def load_dataset(manifest_path, num_samples=None):
    """
    Load raw audio from manifest and compute mel-spectrograms.
    
    Args:
        manifest_path: Path to CSV manifest with 'filepath' and 'label' columns
        num_samples: Max number of samples to load (None for all)
    
    Returns:
        Tuple of (X, y) where X is spectrogram array, y is labels
    """
    df = pd.read_csv(manifest_path)
    if num_samples is not None:
        df = df.sample(n=min(num_samples, len(df)), random_state=42)
    
    all_specs = []
    all_labels = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading audio"):
        file_path = row['filepath']
        label = row['label']
        
        if not os.path.exists(file_path):
            continue
        
        try:
            audio = preprocess_audio(file_path)
            if audio is None:
                continue
            
            spec = librosa.feature.melspectrogram(y=audio, sr=44100, n_mels=128)
            spec_db = librosa.power_to_db(spec, ref=np.max)
            
            all_specs.append(spec_db)
            all_labels.append(label)
        except Exception:
            pass
    
    if not all_specs:
        raise ValueError("No valid audio files loaded")
    
    X = np.array(all_specs, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    
    print(f"Loaded {len(X)} samples")
    
    # Normalize
    X = (X - X.mean()) / (X.std() + 1e-7)
    
    return X, y


def load_dataset_comprehensive(manifest_path, num_samples=None, num_workers=None):
    """
    Load audio and extract spectrograms using multiprocessing.
    
    Args:
        manifest_path: Path to CSV manifest with 'filepath' and 'label' columns
        num_samples: Max samples to load (None for all)
        num_workers: Number of worker processes (default: 4, max: 8)
    
    Returns:
        Tuple of (X, y) where X is spectrogram array, y is labels
    """
    df = pd.read_csv(manifest_path)
    if num_samples is not None:
        df = df.sample(n=min(num_samples, len(df)), random_state=42)
    
    data_rows = [(idx, row['filepath'], int(row['label'])) for idx, row in df.iterrows()]
    
    if num_workers is None:
        num_workers = min(4, max(1, cpu_count() - 1))
    
    print(f"Using {num_workers} worker processes")
    
    try:
        with Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(_process_audio_sample, data_rows, chunksize=8),
                total=len(data_rows),
                desc="Loading audio"
            ))
    except BrokenPipeError:
        print("BrokenPipeError: Retrying with fewer workers...")
        with Pool(2) as pool:
            results = list(tqdm(
                pool.imap(_process_audio_sample, data_rows, chunksize=4),
                total=len(data_rows),
                desc="Loading audio (retry)"
            ))
    
    results = [r for r in results if r is not None]
    if not results:
        raise ValueError("No valid audio files loaded")
    
    all_data, all_labels = zip(*results)
    X = np.array(all_data, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    
    print(f"Loaded {len(X)} samples, X shape: {X.shape}")
    X = (X - X.mean()) / (X.std() + 1e-7)
    
    return X, y




def load_manifest_for_training(
    manifest_path,
    num_samples=None,
    test_split=0.15,
    val_split=0.15,
    random_state=42
):
    """
    Load spectrograms from manifest CSV and prepare train/val/test datasets.
    Uses streaming dataset (SpectrogramDataset) to avoid loading all spectrograms into memory.
    
    Args:
        manifest_path: Path to manifest CSV with 'filepath' and 'label' columns
        num_samples: Max samples to load (None for all)
        test_split: Test set fraction
        val_split: Validation set fraction
        random_state: Random seed for reproducibility
    
    Returns:
        train_dataset, val_dataset, test_dataset: SpectrogramDataset objects
        input_shape: Shape of input features (1, 128, mel_bins)
    """
    from sklearn.model_selection import train_test_split
    
    # Read manifest to get indices
    df = pd.read_csv(manifest_path)
    if num_samples is not None:
        df = df.sample(n=min(num_samples, len(df)), random_state=random_state)
    
    print(f"Found {len(df)} samples in manifest")
    
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
    
    # Create streaming datasets
    train_dataset = SpectrogramDataset(manifest_path, split_indices=train_idx)
    val_dataset = SpectrogramDataset(manifest_path, split_indices=val_idx)
    test_dataset = SpectrogramDataset(manifest_path, split_indices=test_idx)
    
    # Input shape for model (1, 128, mel_bins)
    input_shape = (1, 128, 128)
    
    return train_dataset, val_dataset, test_dataset, input_shape



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
    
    # Collect all file paths and labels
    all_data = []
    all_labels = []
    
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


def load_spectrogram_latents_for_training(
    latent_dir,
    num_samples_per_class=None,
    test_split=0.15,
    val_split=0.15,
    random_state=42
):
    """
    Load mel-spectrogram latents from all codec subdirectories for 2D CNN training.
    
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
    from sklearn.model_selection import train_test_split
    
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
    all_data = []
    all_labels = []
    
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




def _process_audio_sample(row_data: Tuple) -> Optional[Tuple[np.ndarray, int]]:
    """Worker function for audio preprocessing."""
    try:
        idx, file_path, label = row_data
        
        if not os.path.exists(file_path):
            return None
        
        audio = preprocess_audio(file_path)
        if audio is None:
            return None
        
        spec = librosa.feature.melspectrogram(y=audio, sr=44100, n_mels=128)
        spec_db = librosa.power_to_db(spec, ref=np.max)
        return (spec_db, label)
    except Exception as e:
        return None
