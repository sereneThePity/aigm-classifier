import os
import numpy as np
import librosa
import librosa.display
from tqdm import tqdm
import pandas as pd
from transforms import apply_transform


# Audio to mel

def extract_mel_spectrogram(file_path, n_mels=128, duration=15, sr=22050):
    try:
        y, sr = librosa.load(file_path, sr=sr, duration=duration)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_dataset(manifest_csv, target_shape=(128, 128)):
    df = pd.read_csv(manifest_csv)
    X, y = [], []

    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        mel = extract_mel_spectrogram(row["filepath"])
        if mel is None:
            continue

        # --- Pad or crop to match training ---
        if mel.shape[1] < target_shape[1]:
            pad_width = target_shape[1] - mel.shape[1]
            mel = np.pad(mel, ((0, 0), (0, pad_width)), mode="constant")
        else:
            mel = mel[:, :target_shape[1]]

        # --- Apply same per-sample normalization ---
        mel = (mel - mel.min()) / (mel.max() - mel.min())

        X.append(mel)
        y.append(row["label"])

    X = np.array(X)[..., np.newaxis]  # add channel dim
    y = np.array(y)
    print(f"Loaded dataset: {X.shape}, labels: {y.shape}")
    return X, y


def load_dataset_with_transforms(manifest_csv, target_shape=(128, 128), n_mels=128, transform="random"):
    """
    Load dataset and apply random transforms to audio before mel extraction.
    
    Args:
        manifest_csv: Path to CSV with 'filepath' and 'label' columns
        target_shape: Target shape for mel spectrogram (freq, time)
        n_mels: Number of mel frequency bins
        transform: Transform type to apply ("random" or specific)
    
    Returns:
        X: array of shape (n_samples, freq, time, 1)
        y: array of labels
    """
    df = pd.read_csv(manifest_csv)
    X, y = [], []
    
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        filepath = row["filepath"]
        label = row["label"]
        
        try:
            # Load audio
            audio, sr = librosa.load(filepath, sr=22050, duration=15)
            
            # Apply random transforms
            audio_transformed = apply_transform(audio, sr, transform=transform)
            
            # Extract mel spectrogram
            mel_spec = librosa.feature.melspectrogram(y=audio_transformed, sr=sr, n_mels=n_mels)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Pad or crop to target shape
            if mel_spec_db.shape[1] < target_shape[1]:
                pad_width = target_shape[1] - mel_spec_db.shape[1]
                mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode="constant")
            else:
                mel_spec_db = mel_spec_db[:, :target_shape[1]]
            
            # Normalize per-sample
            mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
            
            X.append(mel_spec_db)
            y.append(label)
                
        except Exception as e:
            print(f"⚠️  Error processing {filepath}: {e}")
            continue
    
    X = np.array(X)[..., np.newaxis]  # add channel dim
    y = np.array(y)
    print(f"Loaded dataset with transforms: {X.shape}, labels: {y.shape}")
    return X, y

