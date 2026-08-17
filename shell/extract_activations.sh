#!/bin/bash

#SBATCH --job-name="Extract-Activations"
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=20
#SBATCH --mail-type=NONE
#SBATCH --output=/home/student/s/ssahu/share/aigm-classifier/logs/slurm_%j.out
#SBATCH --error=/home/student/s/ssahu/share/aigm-classifier/logs/slurm_%j.err

# SLURM Script to extract intermediate activations from neural network models
# Supports single layer or all-layers extraction with GPU acceleration
#
# Usage:
#   sbatch extract_activations.sh [extract|extract_all] [model_path] [manifest] [layer_name] [device]
#
# Examples:
#   sbatch extract_activations.sh extract models/cnn_model.pt data/trainset/manifest.csv fc
#   sbatch extract_activations.sh extract_all models/cnn_model.pt data/trainset/manifest.csv "" cuda
#   sbatch extract_activations.sh extract models/CNN.pt data/trainset/manifest.csv "" auto
#
# Defaults:
#   Command: extract_all (all layers)
#   Model: models/cnn_model.pt
#   Manifest: data/trainset/manifest.csv
#   Layer: auto-detect last conv/dense layer (for extract)
#   Device: auto-detect (cuda if available, else cpu)

set -e  # Exit on any error

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Default parameters
COMMAND="${1:-extract_all}"
MODEL_PATH="${2:-/home/student/s/ssahu/share/aigm-classifier/models/22khz.pt}"
MANIFEST="${3:-/home/student/s/ssahu/share/aigm-classifier/data/testset/manifest.csv}"
LAYER_NAME="${4:-}"
DEVICE="${5:-}"  # Empty string means auto-detect

# Resolve device
if [[ "$DEVICE" == "" ]] || [[ "$DEVICE" == "auto" ]]; then
    DEVICE=""  # Let Python auto-detect
    DEVICE_NAME="auto-detect"
else
    DEVICE_NAME="$DEVICE"
fi

echo -e "${BLUE}🎯 Starting activation extraction job...${NC}"
echo -e "${BLUE}Command: $COMMAND${NC}"
echo -e "${BLUE}Model: $MODEL_PATH${NC}"
echo -e "${BLUE}Manifest: $MANIFEST${NC}"
if [[ -n "$LAYER_NAME" ]]; then
    echo -e "${BLUE}Layer: $LAYER_NAME${NC}"
else
    echo -e "${BLUE}Layer: auto-detect${NC}"
fi
echo -e "${BLUE}Device: $DEVICE_NAME${NC}"

# Validate files
if [[ ! -f "$MODEL_PATH" ]]; then
    echo -e "${YELLOW}⚠️  Model not found: $MODEL_PATH${NC}"
    exit 1
fi

if [[ ! -f "$MANIFEST" ]]; then
    echo -e "${YELLOW}⚠️  Manifest not found: $MANIFEST${NC}"
    exit 1
fi

# Navigate to project directory
cd /home/student/s/ssahu/share/aigm-classifier

# Show GPU info if available
if command -v nvidia-smi &> /dev/null; then
    echo -e "${BLUE}GPU Status:${NC}"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader | head -n1 || true
fi

# Build command
if [[ "$COMMAND" == "extract_all" ]]; then
    echo -e "${BLUE}Extracting from ALL layers...${NC}"
    CMD="python scripts/evaluate_model.py extract_all \
        --model_path \"$MODEL_PATH\" \
        --manifest_path \"$MANIFEST\" \
        --sample_rate 22050"
    if [[ -n "$DEVICE" ]]; then
        CMD="$CMD --device $DEVICE"
    fi
elif [[ "$COMMAND" == "extract" ]]; then
    echo -e "${BLUE}Extracting from single layer...${NC}"
    CMD="python scripts/evaluate_model.py extract \
        --model_path \"$MODEL_PATH\" \
        --manifest_path \"$MANIFEST\" \
        --sample_rate 22050"
    if [[ -n "$LAYER_NAME" ]]; then
        CMD="$CMD --layer_name $LAYER_NAME"
    fi
    if [[ -n "$DEVICE" ]]; then
        CMD="$CMD --device $DEVICE"
    fi
else
    echo -e "${YELLOW}Unknown command: $COMMAND${NC}"
    echo -e "${YELLOW}Use 'extract' or 'extract_all'${NC}"
    exit 1
fi

# Run extraction
eval "$CMD"

echo -e "${GREEN}✅ Extraction complete!${NC}"
