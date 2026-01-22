"""
transforms.py
-------------
Audio transformation utilities for robustness testing.

Each transform takes an audio array (numpy) + sample rate and returns a modified version.
"""

import os
import io
import random
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from scipy.signal import butter, lfilter, convolve
from scipy.io import wavfile

# -----------------------------
# Basic I/O
# -----------------------------
def load_audio(path, sr=22050, duration=None):
    """Load audio file and resample."""
    y, sr = librosa.load(path, sr=sr, duration=duration, mono=False)
    return y, sr

def save_audio(path, y, sr=22050):
    """Save numpy audio array to a file."""
    sf.write(path, y, sr)

# -----------------------------
# Transformations
# -----------------------------

def resample(y, sr, target_sr):
    """Resample audio to target sample rate."""
    return librosa.resample(y, orig_sr=sr, target_sr=target_sr), target_sr

def downmix_to_mono(y):
    """Convert stereo → mono."""
    if y.ndim > 1:
        y = np.mean(y, axis=0)
    return y

def upmix_to_stereo(y):
    """Convert mono → stereo (duplicate channels)."""
    if y.ndim == 1:
        y = np.stack([y, y])
    return y

def reencode(y, sr, codec="mp3", bitrate="96k"):
    """Re-encode audio through lossy codec using pydub (MP3, OGG, OPUS)."""
    tmp_wav = io.BytesIO()
    sf.write(tmp_wav, y, sr, format="wav")
    tmp_wav.seek(0)
    seg = AudioSegment.from_file(tmp_wav, format="wav")

    tmp_out = io.BytesIO()
    seg.export(tmp_out, format=codec, bitrate=bitrate)
    tmp_out.seek(0)
    seg_reenc = AudioSegment.from_file(tmp_out, format=codec)
    y_reenc = np.array(seg_reenc.get_array_of_samples()).astype(np.float32) / 32768.0
    return y_reenc, seg_reenc.frame_rate

def add_noise(y, snr_db=10):
    """Add white noise at given SNR (dB)."""
    rms = np.sqrt(np.mean(y**2))
    snr_linear = 10 ** (snr_db / 10)
    noise_rms = rms / np.sqrt(snr_linear)
    noise = np.random.normal(0, noise_rms, size=y.shape)
    return y + noise

def add_music_noise(y, sr, noise_path, snr_db=10):
    """Overlay another track (music noise) at target SNR."""
    noise, _ = librosa.load(noise_path, sr=sr)
    if len(noise) < len(y):
        noise = np.tile(noise, int(np.ceil(len(y)/len(noise))))[:len(y)]
    rms_y = np.sqrt(np.mean(y**2))
    rms_n = np.sqrt(np.mean(noise**2))
    desired_rms_n = rms_y / (10 ** (snr_db / 20))
    noise = noise * (desired_rms_n / rms_n)
    return y + noise

def time_stretch(y, rate):
    """Time-stretch audio (0.7–1.3 typical range)."""
    return librosa.effects.time_stretch(y, rate)

def pitch_shift(y, sr, n_steps):
    """Pitch-shift audio ± semitones."""
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def reverb(y, decay=0.5):
    """Add a simple reverberation via convolution."""
    ir = np.zeros(int(0.3 * 22050))
    ir[0] = 1.0
    ir[int(0.02 * 22050)] = decay
    return convolve(y, ir, mode='full')[:len(y)]

def equalize(y, gain_db=6, freq=1000, sr=22050):
    """Simple parametric EQ boost around freq."""
    Q = 1.0
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0)/(2*Q)
    A = 10 ** (gain_db/40)
    b0 = 1 + alpha*A
    b1 = -2*np.cos(w0)
    b2 = 1 - alpha*A
    a0 = 1 + alpha/A
    a1 = -2*np.cos(w0)
    a2 = 1 - alpha/A
    b = np.array([b0, b1, b2]) / a0
    a = np.array([1, a1/a0, a2/a0])
    return lfilter(b, a, y)

def bandpass(y, sr, lowcut, highcut):
    """Apply bandpass filter."""
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(4, [low, high], btype='band')
    return lfilter(b, a, y)

def crop(y, sr, offset_sec=0, duration_sec=30):
    """Crop or offset audio segment."""
    start = int(offset_sec * sr)
    end = start + int(duration_sec * sr)
    return y[start:end]

def normalize(y):
    """Normalize audio to -1..1."""
    return y / np.max(np.abs(y))

def embed_with_silence(y, sr, pre_silence=2.0, post_silence=2.0):
    """Embed in silence padding."""
    pre = np.zeros(int(pre_silence * sr))
    post = np.zeros(int(post_silence * sr))
    return np.concatenate([pre, y, post])

# -----------------------------
# Transform pipeline (for batch testing)
# -----------------------------

def apply_transform(y, sr, transform="random"):
    """Apply the give transform for stress testing."""
    transforms = {
        "resample": lambda: librosa.resample(y, orig_sr=sr, target_sr=random.choice([8000, 16000, 44100])),
        "downmix": lambda: downmix_to_mono(y),
        "upmix": lambda: upmix_to_stereo(y),
        "reencode": lambda: reencode(y, sr, codec=random.choice(["mp3", "ogg"]), bitrate=random.choice(["64k", "96k", "128k"]))[0],
        "equalize": lambda: equalize(y, gain_db=random.uniform(-6, 6), freq=random.uniform(500, 5000), sr=sr),
        "time_stretch": lambda: librosa.effects.time_stretch(y, rate=random.uniform(0.8, 1.2)),
        "pitch_shift": lambda: librosa.effects.pitch_shift(y, sr=sr, n_steps=random.uniform(-2, 2)),
        "noise": lambda: add_noise(y, snr_db=random.choice([5, 10, 20])),
        "reverb": lambda: reverb(y, decay=random.uniform(0.3, 0.7)),
        "bandpass": lambda: bandpass(y, sr, 300, 3400),
        "crop": lambda: crop(y, sr, offset_sec=random.uniform(0, 10), duration_sec=10),
        "normalize": lambda: normalize(y),
        "embed_with_silence": lambda: embed_with_silence(y, sr, pre_silence=1.0, post_silence=1.0)
    }
    
    if transform == "random":
        selected = random.sample(list(transforms.items()), random.randint(2, 4))
        for name, transform_fn in selected:
            y = transform_fn()
            print(f"Applying transform: {name}")
    elif transform in transforms:
        transform_fn = transforms[transform]
        y = transform_fn()
        print(f"Applying transform: {transform}")
    else:
        print(f"Unknown transform: {transform}")
    return y
