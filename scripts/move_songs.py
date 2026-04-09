"""
Move 10000 random songs from trainset/fake to testset/fake
"""

import os
import shutil
import random
from pathlib import Path

# Paths
trainset_fake = Path("data/trainset/fake")
testset_fake = Path("data/testset/fake")

# Create testset/fake if it doesn't exist
testset_fake.mkdir(parents=True, exist_ok=True)

# Get all files from trainset/fake
all_files = list(trainset_fake.glob("**/*"))
audio_files = [f for f in all_files if f.is_file() and f.suffix.lower() in ['.mp3', '.wav', '.flac', '.ogg']]

print(f"Found {len(audio_files)} audio files in trainset/fake")

if len(audio_files) < 10000:
    print(f"Warning: Only {len(audio_files)} files available, moving all of them")
    to_move = audio_files
else:
    to_move = random.sample(audio_files, 10000)

print(f"Moving {len(to_move)} files...")

for i, src_file in enumerate(to_move, 1):
    # Preserve directory structure
    rel_path = src_file.relative_to(trainset_fake)
    dst_file = testset_fake / rel_path
    
    # Create parent directories
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Move file
    shutil.move(str(src_file), str(dst_file))
    
    if i % 1000 == 0:
        print(f"  Moved {i}/{len(to_move)}")

print(f"✅ Done! Moved {len(to_move)} files")
