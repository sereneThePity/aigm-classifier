#!/bin/bash

#SBATCH --job-name="SAE-Training"
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:A100:1
#SBATCH --mail-type=NONE
#SBATCH --output=/home/student/s/ssahu/share/aigm-classifier/logs/slurm_%j.out
#SBATCH --error=/home/student/s/ssahu/share/aigm-classifier/logs/slurm_%j.err

# SLURM Script to train Sparse Autoencoder (SAE) on intermediate activations using GPU on HPC
# Submit with: sbatch train_sae.sh [options]
# Monitor with: squeue -u $USER

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default parameters (SAE-specific)
PATCH_SIZE=1
NB_CONCEPTS=128
TOP_K=32
EPOCHS=50
BATCH_SIZE=32
LEARNING_RATE=0.001
ACTIVATION_LAYER="conv6"
USE_CUDA=false


# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --patch_size)
            PATCH_SIZE="$2"
            shift 2
            ;;
        --nb_concepts)
            NB_CONCEPTS="$2"
            shift 2
            ;;
        --top_k)
            TOP_K="$2"
            shift 2
            ;;
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
        --activation_layer)
            ACTIVATION_LAYER="$2"
            shift 2
            ;;
        --use_cuda)
            USE_CUDA=true
            shift
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
echo -e "${BLUE}    HPC Sparse Autoencoder (SAE) Training with SLURM + GPU${NC}"
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

# Check CUDA availability and auto-enable if GPU detected
echo -e "\n${YELLOW}[*] Checking GPU availability...${NC}"
python -c "
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

# Auto-enable CUDA if GPU is available
if python -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    if [ "$USE_CUDA" = "false" ]; then
        echo "    → Auto-enabling CUDA"
        USE_CUDA=true
    fi
fi

# Verify data directory exists
DATA_DIR="$PROJECT_ROOT/data/processed"
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}❌ Error: Data directory not found at $DATA_DIR${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Data directory found${NC}"

# Change to project root
cd "$PROJECT_ROOT"

# Create logs directory
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/training_sae_$(date +%Y%m%d_%H%M%S).log"

echo -e "\n${YELLOW}[*] SAE Training Parameters:${NC}"
echo "    Patch Size:        $PATCH_SIZE"
echo "    Nb Concepts:       $NB_CONCEPTS"
echo "    Top-K:             $TOP_K"
echo "    Epochs:            $EPOCHS"
echo "    Batch Size:        $BATCH_SIZE"
echo "    Learning Rate:     $LEARNING_RATE"
echo "    Activation Layer:  $ACTIVATION_LAYER"
echo "    Use CUDA:          $USE_CUDA"
echo -e "    Log File:          $LOG_FILE"

# Set PyTorch environment variables for GPU optimization
export CUDA_LAUNCH_BLOCKING=1  # For better error messages
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Prevent OOM issues
export CUBLAS_WORKSPACE_CONFIG=:16:8  # For cuBLAS performance

echo -e "\n${BLUE}============================================================${NC}"
echo -e "${BLUE}    Starting SAE Training${NC}"
echo -e "${BLUE}============================================================${NC}"

# Run training
CMD="python \"$PROJECT_ROOT/scripts/train_topk_sae.py\" \
    --patch_size \"$PATCH_SIZE\" \
    --nb_concepts \"$NB_CONCEPTS\" \
    --top_k \"$TOP_K\" \
    --epochs \"$EPOCHS\" \
    --batch_size \"$BATCH_SIZE\" \
    --lr \"$LEARNING_RATE\" \
    --activation_layer \"$ACTIVATION_LAYER\""

# Add CUDA flag if requested
if [ "$USE_CUDA" = true ]; then
    CMD="$CMD --use_cuda"
fi

eval "$CMD" 2>&1 | tee "$LOG_FILE"

EXIT_CODE=$?

echo -e "\n${BLUE}============================================================${NC}"
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Training completed successfully!${NC}"
    echo -e "${GREEN}✓ Log saved to: $LOG_FILE${NC}"
    echo -e "${GREEN}✓ Model saved to: $PROJECT_ROOT/models/topk_sae/${NC}"
else
    echo -e "${RED}❌ Training failed with exit code $EXIT_CODE${NC}"
    echo -e "${RED}Check log file for details: $LOG_FILE${NC}"
fi
echo -e "${BLUE}============================================================${NC}"

exit $EXIT_CODE
