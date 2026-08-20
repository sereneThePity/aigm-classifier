#!/bin/bash

#SBATCH --job-name="Decode-Latents"
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mail-type=NONE
#SBATCH --output=/home/student/s/ssahu/share/aigm-classifier/logs/slurm_%j.out
#SBATCH --error=/home/student/s/ssahu/share/aigm-classifier/logs/slurm_%j.err

# SLURM Script to decode latents to spectrograms for 2D CNN training
# Submit with: sbatch decode_latents_parallel.sh
# Monitor with: squeue -u $USER
# 
# With 40 CPU cores and ~40k files, processing takes ~1 hour (1-2 sec/file)
# Adjust NUM_WORKERS (default 40) if needed for resource constraints

set -e  # Exit on any error

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default parameters
INPUT_DIR="${1:-/home/student/s/ssahu/share/aigm-classifier/data/encoded_latents_16khz}"
OUTPUT_DIR="${2:-/home/student/s/ssahu/share/aigm-classifier/data/encoded_trainset_16khz}"
NUM_WORKERS="${3:-40}"

# Default decode parameters
SR=16000
N_MELS=128

echo -e "${BLUE}🎯 Starting latent decoding job...${NC}"
echo -e "${BLUE}Input: $INPUT_DIR${NC}"
echo -e "${BLUE}Output: $OUTPUT_DIR${NC}"
echo -e "${BLUE}Sample rate: $SR Hz${NC}"
echo -e "${BLUE}Mel bins: $N_MELS${NC}"
echo -e "${BLUE}Using $NUM_WORKERS parallel workers${NC}"

# Navigate to project directory
cd /home/student/s/ssahu/share/aigm-classifier

# Run decoding script
python scripts/decode_latents_to_audio.py \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    --sr "$SR" \
    --n-mels "$N_MELS" \
    --workers "$NUM_WORKERS"

echo -e "${GREEN}✅ Decoding job completed!${NC}"

