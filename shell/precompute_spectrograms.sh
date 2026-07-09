#!/bin/bash

#SBATCH --job-name="Precompute-Spectrograms"
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mail-type=NONE
#SBATCH --output=/home/student/s/ssahu/share/aigm-classifier/logs/slurm_%j.out
#SBATCH --error=/home/student/s/ssahu/share/aigm-classifier/logs/slurm_%j.err

# SLURM Script to pre-compute and cache mel-spectrograms
# Usage: sbatch precompute_spectrograms.sh [--workers N] [--output_dir PATH]
#
# Pre-computes spectrograms without neural codec bias for fast training

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default parameters
WORKERS=8
MANIFEST="/home/student/s/ssahu/share/aigm-classifier/data/trainset/manifest.csv"
OUTPUT_DIR="/home/student/s/ssahu/share/aigm-classifier/data/cached_spectrograms"

while [[ $# -gt 0 ]]; do
    case $1 in
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --manifest)
            MANIFEST="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Get project root - use SLURM_SUBMIT_DIR if available (most reliable)
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    # Fallback for non-SLURM execution
    SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null || echo "$0")"
    SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
fi

# Ensure logs directory exists (required for SLURM output redirection)
mkdir -p "$PROJECT_ROOT/logs" 2>/dev/null || true

# Source bashrc to load environment
echo -e "${YELLOW}[*] Sourcing bashrc and activating conda environment...${NC}"
source ~/.bashrc

# Activate conda environment
conda activate music
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Error: Failed to activate conda environment 'music'${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Conda environment 'music' activated${NC}"

echo -e "\n${BLUE}============================================================${NC}"
echo -e "${BLUE}    Pre-computing Mel-Spectrograms (CPU-only)${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo -e "${YELLOW}SLURM Job Information:${NC}"
echo "    Job ID:            $SLURM_JOB_ID"
echo "    Job Name:          $SLURM_JOB_NAME"
echo "    Partition:         $SLURM_JOB_PARTITION"
echo "    Nodes:             $SLURM_NNODES"
echo "    CPUs per Task:     $SLURM_CPUS_PER_TASK"
echo "    Memory:            $SLURM_MEM_PER_NODE"
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# Create logs directory
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/precompute_$(date +%Y%m%d_%H%M%S).log"

echo -e "\n${YELLOW}[*] Precomputation Parameters:${NC}"
echo "    Input Manifest:    $MANIFEST"
echo "    Output Directory:  $OUTPUT_DIR"
echo "    Workers:           $WORKERS"
echo -e "    Log File:          $LOG_FILE"

# Verify input
if [ ! -f "$MANIFEST" ]; then
    echo -e "${RED}❌ Error: Manifest file not found at $MANIFEST${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Input manifest found${NC}\n"

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}    Starting Spectrogram Pre-computation${NC}"
echo -e "${BLUE}============================================================${NC}"

# Run preprocessing
python3 "$PROJECT_ROOT/scripts/precompute_spectrograms.py" \
    --manifest "$MANIFEST" \
    --output_dir "$OUTPUT_DIR" \
    --workers "$WORKERS" 2>&1 | tee "$LOG_FILE"

EXIT_CODE=$?

echo -e "\n${BLUE}============================================================${NC}"
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Spectrogram pre-computation completed successfully!${NC}"
    echo -e "${GREEN}✓ Log saved to: $LOG_FILE${NC}"
    echo -e "${GREEN}✓ Cached specs saved to: $OUTPUT_DIR${NC}"
    echo -e "${GREEN}✓ Next: sbatch train_cnn_gpu.sh --use_cached${NC}"
else
    echo -e "${RED}❌ Pre-computation failed with exit code $EXIT_CODE${NC}"
    echo -e "${RED}Check log file for details: $LOG_FILE${NC}"
fi
echo -e "${BLUE}============================================================${NC}"

exit $EXIT_CODE
