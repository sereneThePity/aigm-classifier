#!/bin/bash

#SBATCH --job-name="CNN2D-Spectrogram-Classifier"
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --dependency=afterok:11706322
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:A100:1
#SBATCH --mail-type=NONE
#SBATCH --output=/home/student/s/ssahu/share/aigm-classifier/logs/slurm_%j.out
#SBATCH --error=/home/student/s/ssahu/share/aigm-classifier/logs/slurm_%j.err

# SLURM Script to train 2D CNN classifier
# Usage: sbatch train_cnn_gpu.sh [--use_cached] [--epochs N] [--batch_size N] [--lr LR]
#
# Approaches:
#   Default: Neural codec latents (encodec, dac, etc.)
#   --use_cached: Pre-computed spectrograms (fastest, no codec bias)

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default parameters
EPOCHS=50
BATCH_SIZE=32
LEARNING_RATE=0.001
USE_CACHED=false
LATENT_DIR="/home/student/s/ssahu/share/aigm-classifier/data/encoded_trainset_16khz"
CACHED_MANIFEST="/home/student/s/ssahu/share/aigm-classifier/data/cached_spectrograms/manifest.csv"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr|--learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --use_cached)
            USE_CACHED=false
            shift
            ;;
        --latent_dir)
            LATENT_DIR="$2"
            shift 2
            ;;
        --cached_manifest)
            CACHED_MANIFEST="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Get project root - use SLURM_SUBMIT_DIR if available (most reliable)
# SLURM_SUBMIT_DIR is the directory where sbatch was run
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
echo -e "${BLUE}    HPC 2D CNN Classifier Training${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo -e "${YELLOW}SLURM Job Information:${NC}"
echo "    Job ID:            $SLURM_JOB_ID"
echo "    Job Name:          $SLURM_JOB_NAME"
echo "    Partition:         $SLURM_JOB_PARTITION"
echo "    Nodes:             $SLURM_NNODES"
echo "    CPUs per Task:     $SLURM_CPUS_PER_TASK"
echo "    Memory:            $SLURM_MEM_PER_NODE"
echo "    GPUs:              $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)"
echo ""

# Load CUDA module (adjust version if needed)
echo -e "${YELLOW}[*] Loading CUDA module...${NC}"
module load cuda/11.8 2>/dev/null || echo "    Note: CUDA module pre-loaded or container-based"

# Check CUDA availability
echo -e "\n${YELLOW}[*] Checking GPU availability...${NC}"
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✓ GPU Found: {torch.cuda.get_device_name(0)}')
    print(f'  Device Count: {torch.cuda.device_count()}')
    print(f'  CUDA Version: {torch.version.cuda}')
    print(f'  cuDNN Version: {torch.backends.cudnn.version()}')
    print(f'  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('⚠ No GPU found - training will run on CPU (slower)')
" || {
    echo -e "${RED}Failed to check GPU${NC}"
    exit 1
}

# Verify data source path exists
if [ "$USE_CACHED" = true ]; then
    if [ ! -f "$CACHED_MANIFEST" ]; then
        echo -e "${RED}❌ Error: Cached manifest not found at $CACHED_MANIFEST${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Cached spectrogram manifest found${NC}"
else
    if [ ! -d "$LATENT_DIR" ]; then
        echo -e "${RED}❌ Error: Latent directory not found at $LATENT_DIR${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Latent directory found${NC}"
fi

# Change to project root
cd "$PROJECT_ROOT"

# Create logs directory
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/training_2d_$(date +%Y%m%d_%H%M%S).log"

echo -e "\n${YELLOW}[*] Training Parameters:${NC}"
echo "    Epochs:            $EPOCHS"
echo "    Batch Size:        $BATCH_SIZE"
echo "    Learning Rate:     $LEARNING_RATE"
if [ "$USE_CACHED" = true ]; then
    echo "    Mode:              Cached Spectrograms (no codec bias)"
    echo "    Manifest:          $CACHED_MANIFEST"
else
    echo "    Mode:              Neural Codec Latents"
    echo "    Latent Dir:        $LATENT_DIR"
fi
echo -e "    Log File:          $LOG_FILE"

# Set PyTorch environment variables for GPU optimization
export CUDA_LAUNCH_BLOCKING=1  # For better error messages
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Prevent OOM issues
export CUBLAS_WORKSPACE_CONFIG=:16:8  # For cuBLAS performance

echo -e "\n${BLUE}============================================================${NC}"
echo -e "${BLUE}    Starting 2D CNN Training${NC}"
echo -e "${BLUE}============================================================${NC}"

# Run training
CMD="python3 \"$PROJECT_ROOT/scripts/train_cnn_2d.py\""

# Add data source arguments
if [ "$USE_CACHED" = true ]; then
    CMD="$CMD --use_cached --cached_manifest \"$CACHED_MANIFEST\""
else
    CMD="$CMD --latent_dir \"$LATENT_DIR\""
fi

# Add training parameters
CMD="$CMD --epochs \"$EPOCHS\" --batch_size \"$BATCH_SIZE\" --lr \"$LEARNING_RATE\""

eval "$CMD" 2>&1 | tee "$LOG_FILE"

EXIT_CODE=$?

echo -e "\n${BLUE}============================================================${NC}"
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Training completed successfully!${NC}"
    echo -e "${GREEN}✓ Log saved to: $LOG_FILE${NC}"
    echo -e "${GREEN}✓ Model saved to: $PROJECT_ROOT/models/${NC}"
else
    echo -e "${RED}❌ Training failed with exit code $EXIT_CODE${NC}"
    echo -e "${RED}Check log file for details: $LOG_FILE${NC}"
fi
echo -e "${BLUE}============================================================${NC}"

exit $EXIT_CODE
