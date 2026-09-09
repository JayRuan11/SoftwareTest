#!/bin/bash

#############################################################
# OVI Fusion Model Test Script with LoRA Support
#############################################################

set -e  # Exit on error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default configuration
CONFIG_FILE="${CONFIG_FILE:-ovi/configs/test/test.yaml}"
LORA_PATH="${LORA_PATH:-}"
PROMPTS_FILE="${PROMPTS_FILE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-./test_outputs}"
MODE="${MODE:-t2v}"
SEED="${SEED:-100}"
MODEL_NAME="${MODEL_NAME:-}"

# Distributed settings
export GPU_NUM=${GPU_NUM:-1}
export NUM_NODES=${NUM_NODES:-1}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-9998}
NODE_RANK=${1:-0}

# Performance settings
export NCCL_TIMEOUT=7200
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Enable detailed error tracking
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-INFO}

#############################################################
# Functions
#############################################################

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_usage() {
    cat << EOF
Usage: $0 [NODE_RANK]

Environment Variables:
    CONFIG_FILE       Path to test config YAML (default: ovi/configs/test/test.yaml)
    LORA_PATH         Path to LoRA checkpoint (optional)
    PROMPTS_FILE      Path to CSV file with prompts (optional)
    OUTPUT_DIR        Output directory (default: ./test_outputs)
    MODE              Generation mode: t2v, i2v, t2i2v (default: t2v)
    SEED              Random seed (default: 100)
    MODEL_NAME        Model variant: 720x720_5s, 960x960_5s, 960x960_10s (optional)

    GPU_NUM           Number of GPUs per node (default: 1)
    NUM_NODES         Number of nodes (default: 1)
    MASTER_ADDR       Master node address (default: 127.0.0.1)
    MASTER_PORT       Master port (default: 9998)
    CUDA_VISIBLE_DEVICES  GPU devices to use (default: 0)

Examples:
    # Single GPU test
    bash test.sh

    # Multi-GPU test with LoRA
    GPU_NUM=4 LORA_PATH=./outputs/checkpoints/lora.safetensors bash test.sh

    # Batch test from CSV
    PROMPTS_FILE=./test_prompts.csv OUTPUT_DIR=./results bash test.sh

    # Image-to-video test
    MODE=i2v CONFIG_FILE=./configs/test_i2v.yaml bash test.sh

EOF
}

check_requirements() {
    print_info "Checking requirements..."

    # Check if config file exists
    if [ ! -f "$CONFIG_FILE" ]; then
        print_error "Config file not found: $CONFIG_FILE"
        exit 1
    fi

    # Check if LoRA path exists (if specified)
    if [ -n "$LORA_PATH" ] && [ ! -f "$LORA_PATH" ]; then
        print_warning "LoRA checkpoint not found: $LORA_PATH"
        print_warning "Will use base model without LoRA"
        LORA_PATH=""
    fi

    # Check if prompts file exists (if specified)
    if [ -n "$PROMPTS_FILE" ] && [ ! -f "$PROMPTS_FILE" ]; then
        print_error "Prompts file not found: $PROMPTS_FILE"
        exit 1
    fi

    # Create output directory
    mkdir -p "$OUTPUT_DIR"

    print_info "All requirements satisfied"
}

print_config() {
    print_info "Test Configuration:"
    echo "  Config File:    $CONFIG_FILE"
    echo "  Model Name:     ${MODEL_NAME:-Default}"
    echo "  LoRA Path:      ${LORA_PATH:-None}"
    echo "  Prompts File:   ${PROMPTS_FILE:-None}"
    echo "  Output Dir:     $OUTPUT_DIR"
    echo "  Mode:           $MODE"
    echo "  Seed:           $SEED"
    echo ""
    echo "  GPU_NUM:        $GPU_NUM"
    echo "  NUM_NODES:      $NUM_NODES"
    echo "  NODE_RANK:      $NODE_RANK"
    echo "  MASTER_ADDR:    $MASTER_ADDR"
    echo "  MASTER_PORT:    $MASTER_PORT"
    echo "  CUDA_DEVICES:   $CUDA_VISIBLE_DEVICES"
    echo ""
}

#############################################################
# Main
#############################################################

# Show help
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    print_usage
    exit 0
fi

# Check requirements
check_requirements

# Print configuration
print_config

# Build command arguments
CMD_ARGS=(
    --config "$CONFIG_FILE"
    --output_dir "$OUTPUT_DIR"
    --mode "$MODE"
    --seed "$SEED"
)

# Add optional arguments
if [ -n "$MODEL_NAME" ]; then
    CMD_ARGS+=(--model_name "$MODEL_NAME")
    print_info "Using model: $MODEL_NAME"
fi

if [ -n "$LORA_PATH" ]; then
    CMD_ARGS+=(--lora_path "$LORA_PATH")
    print_info "Using LoRA checkpoint: $LORA_PATH"
fi

if [ -n "$PROMPTS_FILE" ]; then
    CMD_ARGS+=(--prompts_file "$PROMPTS_FILE")
    print_info "Using prompts from: $PROMPTS_FILE"
fi

# Launch test
print_info "Starting test..."
print_info "Results will be saved to: $OUTPUT_DIR"
echo ""

if [ "$GPU_NUM" -eq 1 ] && [ "$NUM_NODES" -eq 1 ]; then
    # Single GPU mode
    print_info "Running in single GPU mode"
    python3 test.py "${CMD_ARGS[@]}"
else
    # Multi-GPU/Multi-node mode
    print_info "Running in distributed mode (GPUs: $GPU_NUM, Nodes: $NUM_NODES)"
    torchrun \
        --nproc_per_node="$GPU_NUM" \
        --nnodes="$NUM_NODES" \
        --node_rank="$NODE_RANK" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        test.py "${CMD_ARGS[@]}"
fi

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    print_info "Test completed successfully!"
    print_info "Check outputs in: $OUTPUT_DIR"
else
    print_error "Test failed with exit code: $EXIT_CODE"
    exit $EXIT_CODE
fi
