"""Preprocessing: load audio and compute mel-spectrograms."""
import argparse
import librosa
import numpy as np
import soundfile as sf


def load_audio(path, sr=22050, mono=True):
    x, _ = librosa.load(path, sr=sr, mono=mono)
    return x


def compute_mel(x, sr=22050, n_mels=128, n_fft=2048, hop_length=512):
    S = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('infile')
    args = parser.parse_args()
    x = load_audio(args.infile)
    m = compute_mel(x)
    print('mel shape:', m.shape)
