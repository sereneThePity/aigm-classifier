"""
Encode audio through neural codecs with equal dataset split.
Splits dataset randomly and equally among multiple codecs.
"""

import torch
import librosa
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from utils import normalize_audio, apply_highpass_filter
from neural_codec_confounders import MetaEnCodecWrapper, DACWrapper, AudioLMCodecWrapper, VALLECodecWrapper, GriffinMelCodec

CODECS = {
    "encodec": MetaEnCodecWrapper,
    "dac": DACWrapper,
    "audiolm": AudioLMCodecWrapper,
    "valle": VALLECodecWrapper,
    "griffin": GriffinMelCodec
}

def preprocess_audio(
    file_path,
    sr=44100,
    segment_duration=5.0,
    target_loudness=-20.0,
    hp_freq=20
):
    """
    Follow preprocessing pipeline from preprocess.py:
    1. Load audio
    2. Resample to target sr
    3. Loudness normalize
    4. Trim silence
    5. Random crop to fixed-length segment
    6. High-pass filter
    """
    try:
        # 1. Load audio
        y, loaded_sr = librosa.load(file_path, sr=None, mono=True)
        
        # 2. Resample
        if loaded_sr != sr:
            y = librosa.resample(y, orig_sr=loaded_sr, target_sr=sr)
        
        # 3. Loudness normalize
        y = normalize_audio(y, method='db', target=target_loudness)
        
        # 4. Trim silence
        y, _ = librosa.effects.trim(y, top_db=40)
        
        # 5. Random crop to fixed-length segment
        segment_samples = int(segment_duration * sr)
        if len(y) >= segment_samples:
            max_start = len(y) - segment_samples
            start_idx = np.random.randint(0, max_start + 1)
            y = y[start_idx:start_idx + segment_samples]
        else:
            pad_width = segment_samples - len(y)
            y = np.pad(y, (0, pad_width), mode='constant')
        
        # 6. High-pass filter
        y = apply_highpass_filter(y, sr, cutoff_freq=hp_freq)
        
        return y
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def encode_decode_and_save(audio_file, output_path, codec):
    """Encode and decode audio through codec, save decoded audio."""
    try:
        # Preprocess
        audio = preprocess_audio(audio_file)
        if audio is None:
            return False
        
        # Encode-decode cycle (inference mode - no gradients)
        with torch.no_grad():
            encoded_frames = codec.encode(audio)
            decoded_audio = codec.decode(encoded_frames)
        
        # Save decoded audio as numpy
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, decoded_audio)
        
        return True
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return False

def assign_codecs_equally(df, codec_names):
    """
    Assign codecs equally and randomly to files.
    
    Args:
        df: DataFrame with file list
        codec_names: List of codec names
    
    Returns:
        List of assigned codec names (one per file)
    """
    num_files = len(df)
    num_codecs = len(codec_names)
    
    # Repeat codec list and shuffle
    codec_assignments = (codec_names * ((num_files // num_codecs) + 1))[:num_files]
    np.random.shuffle(codec_assignments)
    
    return codec_assignments

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Path to CSV manifest")
    parser.add_argument("--output_dir", required=True, help="Output directory for latents")
    parser.add_argument("--codecs", nargs="+", default=["encodec", "dac", "audiolm", "valle", "griffin"], 
                        help="List of codecs to use")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()
    
    # Load manifest
    df = pd.read_csv(args.manifest)
    output_dir = Path(args.output_dir)
    
    # Assign codecs randomly and equally
    codec_assignments = assign_codecs_equally(df, args.codecs)
    df["assigned_codec"] = codec_assignments
    
    print(f"📊 Dataset split:")
    for codec in args.codecs:
        count = (df["assigned_codec"] == codec).sum()
        print(f"   {codec}: {count} files")
    
    # Process by codec
    for codec_name in args.codecs:
        print(f"\n🎯 Processing with {codec_name}...")
        
        # Load codec in inference mode
        codec_class = CODECS[codec_name]
        codec = codec_class(sr=44100)
        
        # Only call eval() if codec has a model attribute (PyTorch models)
        if hasattr(codec, 'model'):
            codec.model.eval()
        
        # Filter files for this codec
        codec_files = df[df["assigned_codec"] == codec_name]
        
        for _, row in tqdm(codec_files.iterrows(), total=len(codec_files), desc=codec_name):
            file_path = row["filepath"]
            label = row["label"]
            
            output_path = output_dir / codec_name / str(label) / f"{Path(file_path).stem}.npy"
            
            success = encode_decode_and_save(file_path, output_path, codec)
    
    print("\n✅ Done!")
