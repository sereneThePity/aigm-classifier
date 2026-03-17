import os
import numpy as np
import librosa
import librosa.display
from tqdm import tqdm
import pandas as pd
import argparse
from scipy.signal import butter, sosfilt
from multiprocessing import Pool
from transforms import apply_transform
from utils import ROOT_DIR, DATA_DIR
from neural_codec_confounders import NeuralCodecConfounder, get_available_codecs


# ===== Comprehensive Preprocessing Pipeline =====

def load_and_prep_audio(
    file_path,
    sr=44100,
    segment_duration=5.0,
    target_loudness=-20.0,
    hp_freq=20,
    codec_name=None,
    codec_confounder=None
):
    """
    Complete preprocessing pipeline:
    1. Load audio
    2. Convert to mono
    3. Resample to target sr (44100 Hz)
    4. Loudness normalize (RMS-based target)
    5. Trim silence
    6. Random crop to fixed-length segment
    7. High-pass filter
    8. (Optional) Apply neural codec confounder
    
    Args:
        file_path: Path to audio file
        sr: Target sample rate (default 44100 Hz)
        segment_duration: Duration of fixed segment in seconds (default 5.0s)
        target_loudness: Target RMS loudness in dB (default -20 dB)
        hp_freq: High-pass filter frequency in Hz (default 20 Hz)
        codec_name: Optional neural codec name to apply as confounder
        codec_confounder: Optional NeuralCodecConfounder instance
    
    Returns:
        audio: Preprocessed audio array, or None if failed
    """
    try:
        # 1. Load audio and automatically convert to mono in librosa.load
        y, loaded_sr = librosa.load(file_path, sr=None, mono=True)
        
        # 2. Mono conversion (already done by librosa with mono=True)
        # y is now mono
        
        # 3. Resample to target sr if needed
        if loaded_sr != sr:
            y = librosa.resample(y, orig_sr=loaded_sr, target_sr=sr)
        
        # 4. Loudness normalize (RMS-based, LUFS approximation)
        y = normalize_loudness(y, target_db=target_loudness)
        
        # 5. Trim silence
        y, _ = librosa.effects.trim(y, top_db=40)
        
        # 6. Random crop to fixed-length segment
        segment_samples = int(segment_duration * sr)
        if len(y) >= segment_samples:
            # Random starting point
            max_start = len(y) - segment_samples
            start_idx = np.random.randint(0, max_start + 1)
            y = y[start_idx:start_idx + segment_samples]
        else:
            # Pad with zeros if shorter than segment duration
            pad_width = segment_samples - len(y)
            y = np.pad(y, (0, pad_width), mode='constant')
        
        # 7. High-pass filter (20 Hz)
        y = apply_highpass_filter(y, sr, cutoff_freq=hp_freq)
        
        # 8. (Optional) Apply neural codec confounder
        if codec_name is not None and codec_confounder is not None:
            codec_audio = codec_confounder.apply_codec(y, codec_name)
            if codec_audio is not None:
                y = codec_audio
        
        return y
    
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None


def normalize_loudness(y, target_db=-20.0):
    """
    RMS-based loudness normalization (approximates LUFS).
    
    Args:
        y: Audio array
        target_db: Target RMS level in dB
    
    Returns:
        Normalized audio array
    """
    # Calculate RMS
    rms = np.sqrt(np.mean(y ** 2))
    
    # Avoid log of zero
    if rms < 1e-7:
        return y
    
    # Current RMS in dB
    current_db = 20 * np.log10(rms)
    
    # Calculate gain needed
    gain_db = target_db - current_db
    gain_linear = 10 ** (gain_db / 20)
    
    return y * gain_linear


def apply_highpass_filter(y, sr, cutoff_freq=20):
    """
    Apply high-pass filter to remove low-frequency rumble.
    
    Args:
        y: Audio array
        sr: Sample rate
        cutoff_freq: Cutoff frequency in Hz
    
    Returns:
        Filtered audio array
    """
    # Design Butterworth high-pass filter
    nyquist = sr / 2
    normalized_cutoff = cutoff_freq / nyquist
    
    # Ensure normalized cutoff is in valid range
    if normalized_cutoff >= 1.0:
        normalized_cutoff = 0.99
    if normalized_cutoff <= 0:
        normalized_cutoff = 0.01
    
    sos = butter(4, normalized_cutoff, btype='high', output='sos')
    y_filtered = sosfilt(sos, y)
    return y_filtered


def extract_mel_spectrogram(file_path, n_mels=128, sr=44100):
    """Extract mel spectrogram from preprocessed audio."""
    try:
        y = load_and_prep_audio(file_path, sr=sr)
        if y is None:
            return None
        
        # 8. Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        
        # 9. Log scale (already done with power_to_db)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 10. Normalize spectrogram (mean/std normalization)
        mel_spec_db = normalize_spectrogram(mel_spec_db)
        
        return mel_spec_db
    except Exception as e:
        print(f"❌ Error extracting mel spectrogram from {file_path}: {e}")
        return None


def normalize_spectrogram(spec):
    """
    Normalize spectrogram using mean/std normalization.
    
    Args:
        spec: Mel spectrogram array
    
    Returns:
        Normalized spectrogram
    """
    mean = np.mean(spec)
    std = np.std(spec)
    
    if std < 1e-7:
        return spec
    
    return (spec - mean) / std

# Module-level function for multiprocessing (must be at module level to be pickleable)
def _process_audio_file(args_tuple):
    """
    Process a single audio file for preprocessing.
    This function must be at module level to be pickleable for multiprocessing.
    Args: tuple of (filepath, label, segment_duration, target_loudness, hp_freq, n_mels, target_shape, codec_name)
    """
    filepath, label, segment_duration, target_loudness, hp_freq, n_mels, target_shape, codec_name = args_tuple
    
    try:
        # Create codec confounder in worker process if needed
        codec_confounder = None
        if codec_name is not None:
            codec_confounder = NeuralCodecConfounder(sr=44100)
        
        # Step 1-7: Comprehensive audio preprocessing
        audio = load_and_prep_audio(
            filepath,
            sr=44100,
            segment_duration=segment_duration,
            target_loudness=target_loudness,
            hp_freq=hp_freq,
            codec_name=codec_name,
            codec_confounder=codec_confounder
        )
        
        if audio is None:
            return None, None
        
        # Step 8: Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=44100, n_mels=n_mels)
        
        # Step 9: Log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Step 10: Normalize spectrogram (mean/std)
        mel_spec_db = normalize_spectrogram(mel_spec_db)
        
        # Pad or crop to target shape
        # Frequency dimension (should already be n_mels)
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
        
        return mel_spec_db, label
    
    except Exception as e:
        print(f"⚠️  Error processing {filepath}: {e}")
        return None, None


def load_dataset_comprehensive(
    manifest_csv, 
    n_mels=128, 
    target_shape=(128, 128),
    segment_duration=5.0,
    target_loudness=-20.0,
    hp_freq=20,
    num_workers=20,
    codec_name=None
):
    """
    Load dataset with comprehensive preprocessing:
    - Full audio preprocessing pipeline
    - Mel spectrogram extraction with normalization
    - (Optional) Neural codec confounder application
    - Multiprocessing support for faster loading
    
    Args:
        manifest_csv: Path to CSV with 'filepath' and 'label' columns
        n_mels: Number of mel frequency bins (default 128)
        target_shape: Target shape for mel spectrogram (freq, time)
        segment_duration: Fixed segment duration in seconds (default 5.0s)
        target_loudness: Target RMS level in dB (default -20 dB)
        hp_freq: High-pass filter frequency in Hz (default 20 Hz)
        num_workers: Number of processes for multiprocessing (default 20)
        codec_name: Optional neural codec name to apply as confounder
    
    Returns:
        X: Array of shape (n_samples, freq, time, 1)
        y: Array of labels
    """
    df = pd.read_csv(manifest_csv)
    X, y = [], []
    
    print(f"📊 Loading dataset from {manifest_csv}")
    print(f"   Settings: segment={segment_duration}s, loudness={target_loudness}dB, hp_filter={hp_freq}Hz")
    if codec_name:
        print(f"   Neural codec confounder: {codec_name}")
    print(f"   Using {num_workers} workers for multiprocessing")
    
    # Create argument tuples for each file
    args_list = [
        (fp, lb, segment_duration, target_loudness, hp_freq, n_mels, target_shape, codec_name) 
        for fp, lb in zip(df["filepath"], df["label"])
    ]
    
    # Process files in parallel
    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(_process_audio_file, args_list), 
                           total=len(df), desc="Processing"))
    
    # Collect results
    for mel_spec_db, label in results:
        if mel_spec_db is not None and label is not None:
            X.append(mel_spec_db)
            y.append(label)
    
    X = np.array(X)[..., np.newaxis]  # Add channel dimension
    y = np.array(y)
    
    print(f"✅ Loaded dataset: X.shape={X.shape}, y.shape={y.shape}")
    print(f"   Class distribution: {np.bincount(y.astype(int))}")
    
    return X, y


def load_dataset(manifest_csv, target_shape=(128, 128)):
    """Legacy function - use load_dataset_comprehensive instead."""
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
            audio, sr = librosa.load(filepath, sr=44100, duration=15)
            
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

