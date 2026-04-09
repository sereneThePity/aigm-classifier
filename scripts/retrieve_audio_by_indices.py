"""
Retrieve audio files from manifest based on spectrogram indices.

This script maps spectrogram indices (from X_spectrograms) back to their 
corresponding audio file paths in the manifest.

Usage:
    python retrieve_audio_by_indices.py --indices 594 21 39 605 575 845 905 228 346 691
    python retrieve_audio_by_indices.py --indices 594 21 39 605 575 845 905 228 346 691 --output mapping.json
    python retrieve_audio_by_indices.py --indices 594 21 39 605 575 845 905 228 346 691 --copy-to ./audio_subset
"""

import csv
import json
import os
import shutil
import argparse
from pathlib import Path
from typing import List, Dict
from utils import ROOT_DIR, DATA_DIR


def retrieve_audio_files(
    manifest_path: str, 
    indices: List[int],
    output_format: str = "list"
) -> Dict:
    """
    Retrieve audio file information based on spectrogram indices.
    
    Args:
        manifest_path: Path to the manifest CSV file
        indices: List of indices corresponding to spectrogram order
        output_format: Either "list" or "mapping"
    
    Returns:
        Dictionary containing audio file information
    """
    
    # Read the manifest
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"✓ Loaded manifest with {len(rows)} entries")
    
    # Validate indices
    invalid_indices = [i for i in indices if i < 0 or i >= len(rows)]
    if invalid_indices:
        print(f"⚠ Invalid indices (out of range): {invalid_indices}")
        print(f"   Valid range is 0-{len(rows)-1}")
    
    # Retrieve audio files
    audio_files = []
    mapping = {}
    
    for idx in indices:
        if 0 <= idx < len(rows):
            row = rows[idx]
            filepath = row['filepath']
            label = row['label']
            generator = row['generator']
            source = row['source']
            
            audio_files.append(filepath)
            mapping[int(idx)] = {
                'filepath': filepath,
                'label': int(label),
                'is_fake': bool(int(label)),
                'generator': generator if generator else None,
                'source': source if source else None
            }
    
    print(f"✓ Retrieved {len(audio_files)} audio files")
    
    return {"files": audio_files, "mapping": mapping}


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve audio files based on spectrogram indices"
    )
    parser.add_argument(
        '--indices',
        type=int,
        nargs='+',
        required=True,
        help='Indices to retrieve (space-separated)'
    )
    parser.add_argument(
        '--manifest',
        type=str,
        default=os.path.join(DATA_DIR, '/trainset/manifest.csv'),
        help='Path to manifest CSV file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Optional output JSON file to save results'
    )
    parser.add_argument(
        '--copy-to',
        type=str,
        default=None,
        help='Optional directory to copy all audio files to'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['list', 'mapping'],
        default='mapping',
        help='Output format: "list" (filepaths only) or "mapping" (with metadata)'
    )
    
    args = parser.parse_args()
    
    # Retrieve audio files
    result = retrieve_audio_files(
        manifest_path=args.manifest,
        indices=args.indices,
        output_format=args.format
    )
    
    # Print results
    print(f"\n{'='*60}")
    print("RETRIEVED AUDIO FILES")
    print(f"{'='*60}")
    
    if args.format == "mapping":
        for idx, info in sorted(result['mapping'].items()):
            label_str = "FAKE" if info['is_fake'] else "REAL"
            gen_str = f" ({info['generator']})" if info['generator'] else ""
            print(f"Index {idx:3d}: {label_str}{gen_str}")
            print(f"           {info['filepath']}")
    else:
        for i, filepath in enumerate(result['files'], 1):
            print(f"{i:2d}. {filepath}")
    
    # Save to file if requested
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(args.output, 'w') as f:
            json.dump(result['mapping'], f, indent=2)
        print(f"\n✓ Results saved to {args.output}")
    
    # Copy audio files if requested
    if args.copy_to:
        copy_audio_files(result['mapping'], args.copy_to)


def copy_audio_files(mapping: Dict, target_dir: str) -> None:
    """
    Copy audio files to a target directory with simple naming.
    
    Args:
        mapping: Dictionary with index -> file info
        target_dir: Target directory to copy files to
    """
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("COPYING AUDIO FILES")
    print(f"{'='*60}")
    
    copied_count = 0
    failed_count = 0
    
    for idx, info in sorted(mapping.items()):
        src_path = info['filepath']
        
        # Get file extension
        _, ext = os.path.splitext(src_path)
        
        # Simple naming: audio_594.wav, audio_21.mp3, etc.
        dest_filename = f"audio_{idx}{ext}"
        dest_path = os.path.join(target_dir, dest_filename)
        
        try:
            if os.path.exists(src_path):
                shutil.copy2(src_path, dest_path)
                label_str = "FAKE" if info['is_fake'] else "REAL"
                print(f"✓ [{copied_count+1}] {label_str:4s} - {dest_filename}")
                copied_count += 1
            else:
                print(f"✗ [{failed_count+1}] NOT FOUND - {src_path}")
                failed_count += 1
        except Exception as e:
            print(f"✗ ERROR - {src_path}: {e}")
            failed_count += 1
    
    print(f"\n{'='*60}")
    print(f"✓ Copied: {copied_count} files")
    if failed_count > 0:
        print(f"✗ Failed: {failed_count} files")
    print(f"✓ Location: {os.path.abspath(target_dir)}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
