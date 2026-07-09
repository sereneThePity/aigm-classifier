"""Simplified preprocessing for two data loading approaches: cached specs and neural codec latents."""
import os
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from typing import Tuple
from utils import normalize_spectrogram


def collate_fn_skip_none(batch):
    """Custom collate function that filters out None values from batch."""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data._utils.default_collate(batch)


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
