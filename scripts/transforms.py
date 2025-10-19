"""Data augmentation and audio transforms."""
import numpy as np

def add_white_noise(x, noise_level=0.005):
    noise = np.random.randn(len(x))
    return x + noise_level * noise


def time_shift(x, shift_max=0.2):
    shift = int(np.random.uniform(-shift_max, shift_max) * len(x))
    return np.roll(x, shift)
