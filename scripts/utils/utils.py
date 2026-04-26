"""Utility functions for dataset manifest and audio helpers."""
import os
import json
from typing import List
import numpy as np
import librosa
from scipy.signal import butter, sosfilt

# Global path configuration
ROOT_DIR = "/share/users/student/s/ssahu/aigm-classifier"
DATA_DIR = "/share/users/student/s/ssahu/aigm-classifier/data"


# ===== Audio Processing Utilities =====

def normalize_audio(audio: np.ndarray, method: str = 'db', target: float = -20.0) -> np.ndarray:
    """
    Normalize audio to target level using specified method.
    
    Args:
        audio: Input audio array
        method: Normalization method
            - 'db': dB-based loudness normalization (LUFS approximation, perceptual)
            - 'linear': Linear RMS normalization (mathematical scale)
        target: Target level
            - If method='db': target in dB (default -20.0)
            - If method='linear': target RMS value (default -20.0 dB ≈ 0.1 linear)
    
    Returns:
        Normalized audio array clipped to [-1.0, 1.0]
    """
    rms = np.sqrt(np.mean(audio ** 2))
    
    if rms < 1e-7:
        return audio
    
    if method == 'db':
        # dB-based normalization (perceptual, LUFS-like)
        current_db = 20 * np.log10(rms)
        gain_db = target - current_db
        gain_linear = 10 ** (gain_db / 20)
        normalized = audio * gain_linear
    
    elif method == 'linear':
        # Linear RMS normalization (mathematical scale)
        # target is the desired RMS value
        gain_linear = target / rms
        normalized = audio * gain_linear
    
    else:
        raise ValueError(f"Unknown normalization method: {method}. Use 'db' or 'linear'.")
    
    return np.clip(normalized, -1.0, 1.0) 


def apply_highpass_filter(y: np.ndarray, sr: int, cutoff_freq: float = 20) -> np.ndarray:
    """
    Apply high-pass filter to remove low-frequency rumble.
    
    Args:
        y: Audio array
        sr: Sample rate
        cutoff_freq: Cutoff frequency in Hz (default 20)
    
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


def normalize_spectrogram(spec: np.ndarray) -> np.ndarray:
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


# ===== JSON and File Utilities =====

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def list_audio_files(root: str, exts=None) -> List[str]:
    if exts is None:
        exts = ['.wav', '.flac', '.mp3', '.ogg']
    out = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if any(fn.lower().endswith(e) for e in exts):
                out.append(os.path.join(dirpath, fn))
    return out
