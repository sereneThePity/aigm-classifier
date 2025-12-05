import os
import csv
import soundfile as sf
import argparse
import random
from tqdm import tqdm

def scan_audio_files(root_dir, exts=(".wav", ".mp3", ".flac", ".ogg", ".m4a"), sample_type="both"):
    """Return dictionary mapping top-level source dirs to list of files."""
    folder_files = {}
    for dirpath, _, filenames in os.walk(root_dir):
        valid = [os.path.join(dirpath, f) for f in filenames if f.lower().endswith(exts)]
        if not valid:
            continue

        parts = dirpath.replace("\\", "/").split("/")
        if len(parts) < 4:
            continue

        # skip files not matching sample_type
        if sample_type == "real" and parts[2] != "real":
            continue
        if sample_type == "fake" and parts[2] != "fake":
            continue

        source_key = "/".join(parts[:4])
        folder_files.setdefault(source_key, []).extend(valid)
    return folder_files

def analyze_audio(path):
    try:
        info = sf.info(path)
        return info.samplerate, info.duration
    except Exception:
        return None, None

def build_manifest(root_dir, output_csv, max_per_folder=50, sample_type="both"):
    header = ["filepath","label","source","generator","format","sample_rate","duration","notes"]
    rows = []

    if sample_type == "real":
        print("📂 Scanning for REAL samples only...")
    elif sample_type == "fake":
        print("📂 Scanning for FAKE samples only...")

    print(f"📂 Scanning {root_dir} and sampling up to {max_per_folder} per top-level source...")

    folder_files = scan_audio_files(root_dir, sample_type=sample_type)

    for source_dir, files in folder_files.items():
        sampled = random.sample(files, min(len(files), max_per_folder))
        print(f"• {os.path.basename(source_dir)} → {len(sampled)} selected (of {len(files)})")

        for path in tqdm(sampled, desc=f"Processing {os.path.basename(source_dir)}", leave=False):
            parts = path.replace("\\","/").split("/")
            if len(parts) < 4:
                continue
            label_str = parts[2]
            source_str = parts[-2]
            label = 0 if label_str == "real" else 1
            generator = source_str if label else ""
            source = source_str if not label else ""
            fmt = os.path.splitext(path)[1].replace(".", "").lower()
            sr, dur = analyze_audio(path)
            rows.append([path, label, source, generator, fmt, sr, dur, ""])

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"✅ Manifest written to {output_csv} ({len(rows)} total entries)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/testset", help="Root folder to scan")
    parser.add_argument("--out", default="data/testset/manifest.csv", help="Output CSV path")
    parser.add_argument("--max", type=int, default=50, help="Maximum files per top-level source")
    parser.add_argument("--type", choices=["real", "fake", "both"], default="both", help="Type of samples to include")
    args = parser.parse_args()
    build_manifest(args.root, args.out, args.max, args.type)