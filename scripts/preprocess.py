import os
import numpy as np
import librosa
import librosa.display
from tqdm import tqdm
import pandas as pd
import argparse
import time
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
from transforms import apply_transform
from utils import (
    ROOT_DIR, 
    DATA_DIR,
    normalize_audio,
    apply_highpass_filter,
    normalize_spectrogram
)
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
        t_start = time.time()
        file_basename = os.path.basename(file_path)
        
        # 1. Load audio
        t = time.time()
        y, loaded_sr = librosa.load(file_path, sr=None, mono=True)
        t_load = time.time() - t
        
        # 2. Mono conversion (already done by librosa with mono=True)
        # y is now mono
        
        # 3. Resample to target sr if needed
        t = time.time()
        if loaded_sr != sr:
            y = librosa.resample(y, orig_sr=loaded_sr, target_sr=sr)
        t_resample = time.time() - t
        
        # 4. Loudness normalize
        t = time.time()
        y = normalize_audio(y, method='db', target=target_loudness)
        t_normalize = time.time() - t
        
        # 5. Trim silence
        t = time.time()
        y, _ = librosa.effects.trim(y, top_db=40)
        t_trim = time.time() - t
        
        # 6. Random crop to fixed-length segment
        t = time.time()
        segment_samples = int(segment_duration * sr)
        if len(y) >= segment_samples:
            max_start = len(y) - segment_samples
            start_idx = np.random.randint(0, max_start + 1)
            y = y[start_idx:start_idx + segment_samples]
        else:
            pad_width = segment_samples - len(y)
            y = np.pad(y, (0, pad_width), mode='constant')
        t_crop = time.time() - t
        
        # 7. High-pass filter
        t = time.time()
        y = apply_highpass_filter(y, sr, cutoff_freq=hp_freq)
        t_hpfilter = time.time() - t
        
        # 8. (Optional) Apply neural codec confounder
        t_codec = 0
        if codec_name is not None and codec_confounder is not None:
            t = time.time()
            if codec_name == 'random':
                codec_audio, used_codec = codec_confounder.apply_random_codec(y)
                codec_used = used_codec
            else:
                codec_audio = codec_confounder.apply_codec(y, codec_name)
                codec_used = codec_name
            t_codec = time.time() - t
            if codec_audio is not None:
                y = codec_audio
        
        t_total = time.time() - t_start
        
        return y
    
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None


def extract_mel_spectrogram_keras(file_path, n_mels=128, duration=15, sr=22050):
    try:
        y, sr = librosa.load(file_path, sr=sr, duration=duration)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

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


# Module-level variable for per-worker codec confounder (initialized via Pool initializer)
_worker_codec_confounder = None
_worker_codec_cache = {}  # Cache for codecs loaded per-file in balanced distribution

def _init_worker(codec_name):
    """Pool initializer: create codec confounder once per worker process."""
    global _worker_codec_confounder, _worker_codec_cache
    if codec_name is not None:
        _worker_codec_confounder = NeuralCodecConfounder(sr=44100, init_only=codec_name)
    else:
        _worker_codec_confounder = None
    
    _worker_codec_cache = {}  # Initialize codec cache for this worker


# Module-level function for multiprocessing (must be at module level to be pickleable)
def _process_audio_file(args_tuple):
    """
    Process a single audio file for preprocessing.
    This function must be at module level to be pickleable for multiprocessing.
    Args: tuple of (filepath, label, segment_duration, target_loudness, hp_freq, n_mels, target_shape, codec_name)
    """
    global _worker_codec_confounder, _worker_codec_cache
    
    filepath, label, segment_duration, target_loudness, hp_freq, n_mels, target_shape, codec_name = args_tuple
    
    try:
        # Create codec confounder if codec_name is specified
        # This ensures each file gets the right codec even with balanced distribution
        local_codec_confounder = None
        if codec_name is not None and codec_name != 'random':
            # Check if codec already loaded in this worker
            if codec_name in _worker_codec_cache:
                # Use cached codec
                local_codec_confounder = _worker_codec_cache[codec_name]
            elif _worker_codec_confounder is not None:
                # Use pre-initialized codec from worker (same codec for all files)
                local_codec_confounder = _worker_codec_confounder
            else:
                # Initialize codec once and cache it for this worker
                local_codec_confounder = NeuralCodecConfounder(sr=44100, init_only=codec_name)
                _worker_codec_cache[codec_name] = local_codec_confounder
        
        # Step 1-7: Comprehensive audio preprocessing
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
    num_workers=12,
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
                    If 'random', will equally distribute all available codecs
    
    Returns:
        X: Array of shape (n_samples, freq, time, 1)
        y: Array of labels
    """
    df = pd.read_csv(manifest_csv)
    X, y = [], []
    
    print(f"📊 Loading dataset: {manifest_csv}")
    print(f"   Segment: {segment_duration}s | Target loudness: {target_loudness}dB | HP filter: {hp_freq}Hz")
    
    codec_info = ""
    if codec_name:
        if codec_name == 'random':
            codec_info = "RANDOM (balanced distribution)"
        else:
            codec_info = codec_name
    else:
        codec_info = "None"
    print(f"   Codec: {codec_info} | Workers: {num_workers}")
    
    # If codec_name is 'random', create balanced codec assignments
    assigned_codecs = None
    if codec_name == 'random':
        # Initialize a temporary confounder to get available codecs
        temp_confounder = NeuralCodecConfounder(sr=44100)
        available_codecs = temp_confounder.get_available_codecs()
        
        # Create balanced assignment: repeat each codec equally
        num_files = len(df)
        num_codecs = len(available_codecs)
        repetitions = (num_files + num_codecs - 1) // num_codecs  # Ceiling division
        
        assigned_codecs = (available_codecs * repetitions)[:num_files]
        np.random.shuffle(assigned_codecs)  # Shuffle for randomness
        
        print(f"   Balanced codec distribution ({num_codecs} codecs):")
    
    # Create argument tuples for each file
    args_list = []
    for i, (fp, lb) in enumerate(zip(df["filepath"], df["label"])):
        # Use assigned codec if available, otherwise use the codec_name passed in
        assigned_codec = assigned_codecs[i] if assigned_codecs is not None else codec_name
        args_list.append((fp, lb, segment_duration, target_loudness, hp_freq, n_mels, target_shape, assigned_codec))
    
    # Process files in parallel (codec initialized once per worker, not per file)
    init_codec = None if assigned_codecs is not None else codec_name
    with Pool(num_workers, initializer=_init_worker, initargs=(init_codec,)) as pool:
        results = list(tqdm(pool.imap(_process_audio_file, args_list), 
                           total=len(df), desc="Processing"))
    
    # Collect results
    for mel_spec_db, label in results:
        if mel_spec_db is not None and label is not None:
            X.append(mel_spec_db)
            y.append(label)
    
    X = np.array(X)[..., np.newaxis]  # Add channel dimension
    y = np.array(y)
    
    print(f"✅ Loaded {len(X)} samples | Shape: {X.shape} | Classes: {np.bincount(y.astype(int))}")
    
    return X, y


def load_dataset(manifest_csv, target_shape=(128, 128), num_workers=40):
    """
    Load dataset using ThreadPoolExecutor for parallel I/O.
    Threads are better for I/O-bound operations (file loading, network).
    
    Args:
        manifest_csv: Path to CSV with 'filepath' and 'label' columns
        target_shape: Target shape for mel spectrogram (freq, time)
        num_workers: Number of threads to use (default 8)
    
    Returns:
        X: array of shape (n_samples, freq, time, 1)
        y: array of labels
    """
    df = pd.read_csv(manifest_csv)
    X, y = [], []

    def process_file(row):
        """Process a single file and return (mel_spec, label) tuple."""
        try:
            mel = extract_mel_spectrogram_keras(row["filepath"])
            if mel is None:
                return None, None

            # --- Pad or crop to match training ---
            if mel.shape[1] < target_shape[1]:
                pad_width = target_shape[1] - mel.shape[1]
                mel = np.pad(mel, ((0, 0), (0, pad_width)), mode="constant")
            else:
                mel = mel[:, :target_shape[1]]

            # --- Apply same per-sample normalization ---
            mel = (mel - mel.min()) / (mel.max() - mel.min())

            return mel, row["label"]
        except Exception as e:
            print(f"❌ Error processing {row['filepath']}: {e}")
            return None, None

    # Use ThreadPoolExecutor for parallel I/O
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(process_file, [row for _, row in df.iterrows()]),
            total=len(df),
            desc="Loading dataset"
        ))

    # Collect successful results
    for mel, label in results:
        if mel is not None and label is not None:
            X.append(mel)
            y.append(label)

    X = np.array(X)[..., np.newaxis]  # add channel dim
    y = np.array(y)
    print(f"✅ Loaded dataset: X.shape={X.shape}, y.shape={y.shape}")
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

