import os
import numpy as np
import librosa
import librosa.display
from tqdm import tqdm

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
    import pandas as pd
    df = pd.read_csv(manifest_csv)
    X, y = [], []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        mel = extract_mel_spectrogram(row['filepath'])
        if mel is None:
            continue
        # resize/crop
        if mel.shape[1] > target_shape[1]:
            mel = mel[:, :target_shape[1]]
        elif mel.shape[1] < target_shape[1]:
            pad_width = target_shape[1] - mel.shape[1]
            mel = np.pad(mel, ((0, 0), (0, pad_width)), mode='constant')

        X.append(mel)
        y.append(row['label'])
    X = np.array(X)[..., np.newaxis]
    y = np.array(y)
    return X, y
