"""Build a JSON manifest of audio files and labels for training/evaluation."""
import argparse
import os
from scripts.utils import list_audio_files, save_json


def build_manifest(root_dir, out_path):
    manifest = []
    for split in os.listdir(root_dir):
        split_dir = os.path.join(root_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for class_name in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            files = list_audio_files(class_dir)
            for p in files:
                manifest.append({'path': p, 'label': class_name, 'split': split})
    save_json(manifest, out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='root data directory')
    parser.add_argument('--out', required=True, help='output manifest JSON')
    args = parser.parse_args()
    build_manifest(args.root, args.out)
