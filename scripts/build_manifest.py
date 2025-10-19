import os
import csv
import soundfile as sf
import argparse
from tqdm import tqdm

def scan_audio_files(root_dir, exts=(".wav", ".mp3", ".flac", ".ogg", ".m4a")):
    audio_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith(exts):
                audio_paths.append(os.path.join(dirpath, f))
    return audio_paths

def analyze_audio(path):
    try:
        info = sf.info(path)
        return info.samplerate, info.duration
    except Exception:
        return None, None

def build_manifest(root_dir, output_csv):
    header = ["filepath","label","source","generator","format","sample_rate","duration","notes"]
    rows = []

    print(f"Scanning {root_dir} ...")
    for path in tqdm(scan_audio_files(root_dir)):
        parts = path.replace("\\","/").split("/")
        # assume structure: data/testset/{real|fake}/{source}/filename
        label_str = parts[-3]
        source_str = parts[-2]

        label = 0 if label_str == "real" else 1
        generator = source_str if label else ""
        source = source_str if not label else ""

        fmt = os.path.splitext(path)[1].replace(".", "").lower()
        sr, dur = analyze_audio(path)
        rows.append([path,label,source,generator,fmt,sr,dur,""])

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"✅ Manifest written to {output_csv} ({len(rows)} entries)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/testset", help="Root folder to scan")
    parser.add_argument("--out", default="data/testset/manifest.csv", help="Output CSV path")
    args = parser.parse_args()
    build_manifest(args.root, args.out)
