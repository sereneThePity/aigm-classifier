import os
import numpy as np
import librosa
import librosa.display
from tqdm import tqdm
import pandas as pd


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

