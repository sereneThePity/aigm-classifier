#!/bin/bash

#SBATCH --job-name="Encode-Latents"
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:A100:1
#SBATCH --mail-type=NONE
#SBATCH --output=/home/student/s/ssahu/share/aigm-classifier/logs/slurm_%j.out
#SBATCH --error=/home/student/s/ssahu/share/aigm-classifier/logs/slurm_%j.err

# SLURM Script to encode audio through neural codecs
# Submit with: sbatch encode_latents.sh
# Monitor with: squeue -u $USER

set -e  # Exit on any error

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default parameters
MANIFEST="${1:-/home/student/s/ssahu/share/aigm-classifier/data/trainset/manifest.csv}"
OUTPUT_DIR="${2:-/home/student/s/ssahu/share/aigm-classifier/data/encoded_latents}"
CODECS="${3:-encodec dac audiolm valle griffin}"
DEVICE="cuda"

echo -e "${BLUE}🎯 Starting EnCodec latent encoding job...${NC}"
echo -e "${BLUE}Manifest: $MANIFEST${NC}"
echo -e "${BLUE}Output: $OUTPUT_DIR${NC}"
echo -e "${BLUE}Codecs: $CODECS${NC}"

# Navigate to project directory
cd /home/student/s/ssahu/share/aigm-classifier

# Run encoding script
python scripts/encode_latents.py \
    --manifest "$MANIFEST" \
    --output_dir "$OUTPUT_DIR" \
    --codecs $CODECS \
    --device "$DEVICE"

echo -e "${GREEN}✅ Encoding complete!${NC}"
