"""Utility functions for dataset manifest and audio helpers."""
import os
import json
from typing import List


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
