import os
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from utils import (
    normalize_audio,
    apply_highpass_filter,
    normalize_spectrogram
)
from neural_codec_confounders import NeuralCodecConfounder


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
    """Comprehensive audio preprocessing: load, resample, normalize, trim, crop, filter, codec."""
    try:
        # Load audio
        y, loaded_sr = librosa.load(file_path, sr=None, mono=True)
        
        # Resample if needed
        if loaded_sr != sr:
            y = librosa.resample(y, orig_sr=loaded_sr, target_sr=sr)
        
        # Loudness normalize
        y = normalize_audio(y, method='db', target=target_loudness)
        
        # Trim silence
        y, _ = librosa.effects.trim(y, top_db=40)
        
        # Random crop to fixed-length segment
        segment_samples = int(segment_duration * sr)
        if len(y) >= segment_samples:
            start_idx = np.random.randint(0, len(y) - segment_samples + 1)
            y = y[start_idx:start_idx + segment_samples]
        else:
            y = np.pad(y, (0, segment_samples - len(y)), mode='constant')
        
        # High-pass filter
        y = apply_highpass_filter(y, sr, cutoff_freq=hp_freq)
        
        # Apply neural codec if specified
        if codec_name is not None and codec_confounder is not None:
            if codec_name == 'random':
                codec_audio, _ = codec_confounder.apply_random_codec(y)
            else:
                codec_audio = codec_confounder.apply_codec(y, codec_name)
            
            if codec_audio is not None:
                y = codec_audio
        
        return y
    
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None


def extract_mel_spectrogram(audio, sr=44100, n_mels=128):
    """Extract and normalize mel spectrogram from audio array."""
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return normalize_spectrogram(mel_spec_db)


def pad_or_crop_spectrogram(mel_spec_db, target_shape=(128, 128)):
    """Pad or crop spectrogram to target shape."""
    # Frequency dimension
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
    
    return mel_spec_db


# Module-level codec management for multiprocessing workers
_worker_codec_confounder = None
_worker_codec_cache = {}
_worker_device_type = 'gpu'


def _init_worker(codec_name, device_type='gpu'):
    """Initialize codec confounder once per worker process."""
    global _worker_codec_confounder, _worker_codec_cache, _worker_device_type
    _worker_device_type = device_type
    
    if codec_name is None:
        _worker_codec_confounder = None
        return
    
    # Initialize codec based on device type
    if device_type == 'both':
        _worker_codec_confounder = NeuralCodecConfounder(sr=44100, device_type='gpu')
        try:
            cpu_confounder = NeuralCodecConfounder(sr=44100, device_type='cpu')
            for codec_key, codec_obj in cpu_confounder.codecs.items():
                if codec_key not in _worker_codec_confounder.codecs:
                    _worker_codec_confounder.codecs[codec_key] = codec_obj
        except:
            pass  # Use GPU codecs only if CPU init fails
    else:
        _worker_codec_confounder = NeuralCodecConfounder(sr=44100, init_only=codec_name, device_type=device_type)
    
    _worker_codec_cache = {}

def _process_audio_file(args_tuple):
    """Process a single audio file: preprocess, extract mel spectrogram, pad/crop."""
    global _worker_codec_confounder, _worker_codec_cache, _worker_device_type
    
    filepath, label, segment_duration, target_loudness, hp_freq, n_mels, target_shape, codec_name = args_tuple
    
    try:
        # Initialize codec confounder if needed
        local_codec_confounder = None
        if codec_name is not None and codec_name != 'random':
            if codec_name in _worker_codec_cache:
                local_codec_confounder = _worker_codec_cache[codec_name]
            elif _worker_codec_confounder is not None:
                local_codec_confounder = _worker_codec_confounder
            else:
                local_codec_confounder = NeuralCodecConfounder(sr=44100, init_only=codec_name, device_type=_worker_device_type)
                _worker_codec_cache[codec_name] = local_codec_confounder
        
        # Preprocess audio
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
        
        # Extract mel spectrogram and normalize
        mel_spec_db = extract_mel_spectrogram(audio, sr=44100, n_mels=n_mels)
        
        # Pad or crop to target shape
        mel_spec_db = pad_or_crop_spectrogram(mel_spec_db, target_shape)
        
        return mel_spec_db, label
    
    except Exception as e:
        print(f"⚠️  Error processing {filepath}: {e}")
        return None, None


def load_precomputed_latents(manifest_csv, latent_dir, target_shape=(128, 128)):
    """Load pre-encoded latents (.npy files) from disk."""
    df = pd.read_csv(manifest_csv)
    X, y = [], []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading precomputed latents"):
        try:
            filepath = row["filepath"]
            label = row["label"]
            
            # Load latent file (.npy) based on audio filename
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            latent_path = os.path.join(latent_dir, f"{base_name}.npy")
            
            if not os.path.exists(latent_path):
                print(f"⚠️ Latent file not found: {latent_path}")
                continue
            
            latent = np.load(latent_path)
            latent = pad_or_crop_spectrogram(latent, target_shape)
            
            X.append(latent)
            y.append(label)
        
        except Exception as e:
            print(f"❌ Error loading latent for {filepath}: {e}")
    
    if not X:
        print(f"❌ No latents loaded from {latent_dir}")
        return np.array([]), np.array([])
    
    X = np.array(X)[..., np.newaxis]  # Add channel dimension
    y = np.array(y)
    
    print(f"✅ Loaded {len(X)} precomputed latents")
    return X, y


def load_dataset_comprehensive(
    manifest_csv, 
    n_mels=128, 
    target_shape=(128, 128),
    segment_duration=5.0,
    target_loudness=-20.0,
    hp_freq=20,
    num_workers=12,
    codec_name=None,
    device_type='gpu',
    latent_mode='cpu_codecs',
    latent_dir=None
):
    """
    Load dataset with full preprocessing pipeline.
    
    Args:
        manifest_csv: Path to CSV with 'filepath' and 'label' columns
        n_mels: Number of mel frequency bins (default 128)
        target_shape: Target shape for mel spectrogram (freq, time)
        segment_duration: Fixed segment duration in seconds (default 5.0s)
        target_loudness: Target RMS level in dB (default -20 dB)
        hp_freq: High-pass filter frequency in Hz (default 20 Hz)
        num_workers: Number of processes for multiprocessing
        codec_name: Neural codec name or 'random' (only for cpu_codecs mode)
        device_type: 'cpu', 'gpu', or 'both'
        latent_mode: 'cpu_codecs' or 'precomputed'
        latent_dir: Directory with precomputed latents (required if latent_mode='precomputed')
    
    Returns:
        X: Array of shape (n_samples, freq, time, 1)
        y: Array of labels
    """
    # Use precomputed latents if requested
    if latent_mode == 'precomputed':
        if latent_dir is None:
            raise ValueError("latent_dir must be provided when latent_mode='precomputed'")
        return load_precomputed_latents(manifest_csv, latent_dir, target_shape)
    
    # Otherwise, process audio with optional codec augmentation
    df = pd.read_csv(manifest_csv)
    X, y = [], []
    
    # Create balanced codec assignments if using random codecs
    assigned_codecs = None
    if codec_name == 'random':
        # Get available codecs
        if device_type == 'both':
            gpu_confounder = NeuralCodecConfounder(sr=44100, device_type='gpu')
            cpu_confounder = NeuralCodecConfounder(sr=44100, device_type='cpu')
            available_codecs = list(set(
                gpu_confounder.get_available_codecs() + 
                cpu_confounder.get_available_codecs()
            ))
        else:
            temp_confounder = NeuralCodecConfounder(sr=44100, device_type=device_type)
            available_codecs = temp_confounder.get_available_codecs()
        
        # Create balanced codec assignments
        num_files = len(df)
        num_codecs = len(available_codecs)
        repetitions = (num_files + num_codecs - 1) // num_codecs
        
        assigned_codecs = (available_codecs * repetitions)[:num_files]
        np.random.shuffle(assigned_codecs)
    
    # Build argument tuples for each file
    args_list = []
    for i, (filepath, label) in enumerate(zip(df["filepath"], df["label"])):
        assigned_codec = assigned_codecs[i] if assigned_codecs is not None else codec_name
        args_list.append((filepath, label, segment_duration, target_loudness, hp_freq, n_mels, target_shape, assigned_codec))
    
    # Process files in parallel
    init_codec = None if assigned_codecs is not None else codec_name
    with Pool(num_workers, initializer=_init_worker, initargs=(init_codec, device_type)) as pool:
        results = list(tqdm(pool.imap(_process_audio_file, args_list), 
                           total=len(df), desc="Processing"))
    
    # Collect results
    for mel_spec_db, label in results:
        if mel_spec_db is not None and label is not None:
            X.append(mel_spec_db)
            y.append(label)
    
    X = np.array(X)[..., np.newaxis]  # Add channel dimension
    y = np.array(y)
    
    return X, y

