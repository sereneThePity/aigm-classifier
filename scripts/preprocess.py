import os
import numpy as np
import librosa
import torch
from tqdm import tqdm
from utils import normalize_spectrogram


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
    for codec_name in codecs:
        codec_path = os.path.join(latent_dir, codec_name)
        print(f"\nLoading from {codec_name}...")
        
        for class_label in range(2):
            class_dir = os.path.join(codec_path, str(class_label))
            if not os.path.exists(class_dir):
                continue
            
            npy_files = sorted([f for f in os.listdir(class_dir) if f.endswith('.npy')])
            if num_samples_per_class is not None:
                npy_files = npy_files[:num_samples_per_class]
            
            for npy_file in tqdm(npy_files, desc=f"  {codec_name} class {class_label}", leave=False):
                try:
                    spec = np.load(os.path.join(class_dir, npy_file))
                    all_data.append(spec)
                    all_labels.append(class_label)
                except Exception as e:
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

