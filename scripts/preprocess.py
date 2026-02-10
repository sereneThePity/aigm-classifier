import os
import numpy as np
import librosa
import librosa.display
from tqdm import tqdm
import pandas as pd
import argparse
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
    
def extract_spectrograms_from_manifest(
    manifest_csv,
    output_dir=None,
    n_mels=128,
    target_shape=(128, 128),
    sr=22050,
    duration=15
):
    """
    Extract mel-spectrograms from audio files listed in a manifest CSV.
    
    This function reads a manifest CSV with audio filepaths and labels, loads each
    audio file, extracts its mel-spectrogram, and returns arrays suitable for model
    activation extraction.
    
    Args:
        manifest_csv (str): Path to manifest CSV with columns 'filepath' and 'label'
        output_dir (str): Directory to save .npy files (optional). If provided,
                         saves X_spectrograms.npy and y_labels.npy
        n_mels (int): Number of mel frequency bins (default: 128)
        target_shape (tuple): Target shape (freq_bins, time_steps) for spectrograms
        sr (int): Sample rate to resample audio to (default: 22050 Hz)
        duration (float): Maximum audio duration in seconds to load (default: 15s)
    
    Returns:
        X (np.ndarray): Array of shape (n_samples, freq_bins, time_steps, 1)
                        containing normalized mel-spectrograms
        y (np.ndarray): Array of shape (n_samples,) with binary labels (0=real, 1=fake)
        
    Example:
        >>> X, y = extract_spectrograms_from_manifest(
        ...     'data/testset/manifest.csv',
        ...     output_dir='data/processed',
        ...     n_mels=128,
        ...     target_shape=(128, 128)
        ... )
        >>> print(X.shape, y.shape)
        (123, 128, 128, 1) (123,)
    """
    df = pd.read_csv(manifest_csv)
    X, y = [], []
    
    print(f"📊 Extracting mel-spectrograms from {len(df)} audio files...")
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing audio"):
        filepath = row["filepath"]
        label = row["label"]
        
        try:
            # Load audio with librosa
            audio, _ = librosa.load(filepath, sr=sr, duration=duration, mono=True)
            
            # Extract mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=sr, n_mels=n_mels
            )
            
            # Convert to dB scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Pad or crop frequency dimension
            if mel_spec_db.shape[0] < target_shape[0]:
                pad_width = target_shape[0] - mel_spec_db.shape[0]
                mel_spec_db = np.pad(
                    mel_spec_db, 
                    ((0, pad_width), (0, 0)), 
                    mode="constant"
                )
            else:
                mel_spec_db = mel_spec_db[:target_shape[0], :]
            
            # Pad or crop time dimension
            if mel_spec_db.shape[1] < target_shape[1]:
                pad_width = target_shape[1] - mel_spec_db.shape[1]
                mel_spec_db = np.pad(
                    mel_spec_db,
                    ((0, 0), (0, pad_width)),
                    mode="constant"
                )
            else:
                mel_spec_db = mel_spec_db[:, :target_shape[1]]
            
            # Per-sample normalization
            min_val = mel_spec_db.min()
            max_val = mel_spec_db.max()
            mel_spec_db = (mel_spec_db - min_val) / (max_val - min_val + 1e-8)
            
            X.append(mel_spec_db)
            y.append(label)
            
        except Exception as e:
            print(f"⚠️  Error processing {filepath}: {e}")
            continue
    
    # Stack all spectrograms and add channel dimension
    X = np.array(X)[..., np.newaxis]
    y = np.array(y)
    
    print(f"✅ Extracted spectrograms: X.shape={X.shape}, y.shape={y.shape}")
    
    # Save if output directory is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        X_path = os.path.join(output_dir, "X_spectrograms.npy")
        y_path = os.path.join(output_dir, "y_labels.npy")
        
        np.save(X_path, X)
        np.save(y_path, y)
        
        print(f"💾 Saved X_spectrograms.npy to {X_path} ({X.nbytes / 1024**2:.2f} MB)")
        print(f"💾 Saved y_labels.npy to {y_path}")
    
    return X, y




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
            print(f"Transformed audio shape: {audio_transformed.shape}")
            # Extract mel spectrogram
            mel_spec = librosa.feature.melspectrogram(y=audio_transformed, sr=sr, n_mels=n_mels)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Pad or crop to target shape (both frequency and time dimensions)
            # Frequency dimension
            if mel_spec_db.shape[0] < target_shape[0]:
                pad_width = target_shape[0] - mel_spec_db.shape[0]
                mel_spec_db = np.pad(mel_spec_db, ((0, pad_width), (0, 0)), mode="constant")
            else:
                mel_spec_db = mel_spec_db[:target_shape[0], :]
            
            # Time dimension
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract mel-spectrograms from manifest CSV and save as .npy arrays"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/testset/manifest.csv",
        help="Path to manifest CSV (default: data/testset/manifest.csv)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed",
        help="Output directory for .npy files (default: data/processed)"
    )
    parser.add_argument(
        "--n-mels",
        type=int,
        default=128,
        help="Number of mel frequency bins (default: 128)"
    )
    parser.add_argument(
        "--freq",
        type=int,
        default=128,
        help="Frequency dimension for spectrograms (default: 128)"
    )
    parser.add_argument(
        "--time",
        type=int,
        default=128,
        help="Time dimension for spectrograms (default: 128)"
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=22050,
        help="Sample rate in Hz (default: 22050)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=15,
        help="Max audio duration in seconds (default: 15)"
    )
    
    args = parser.parse_args()
    
    X, y = extract_spectrograms_from_manifest(
        manifest_csv=args.manifest,
        output_dir=args.output,
        n_mels=args.n_mels,
        target_shape=(args.freq, args.time),
        sr=args.sr,
        duration=args.duration
    )
