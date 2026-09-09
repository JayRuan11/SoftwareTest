#!/bin/bash
#############################################################
NODE_RANK=${RANK-0}

# 设置基本环境变量
export NCCL_TIMEOUT=7200
export OMP_NUM_THREADS=4

# 设置分布式训练参数
export GPU_NUM=${RESOURCE_GPU:-1}
export NUM_NODES=${WORLD_SIZE:-1}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-29500}

export CONFIG="ovi/configs/train/train.yaml"

# Dataset paths: space-separated string
# Example: DATASET="/path1 /path2 /path3"
DATASET="/mnt/vision-gen-ks3/Video_Generation/ruanjunjie/ovi_add/dataset/final.json"
# DATASET="/mnt/vision-gen-ks3/Video_Generation/ruanjunjie/seed-vc/dataset/final_copy.json"


# Split DATASET into an array (handles empty or single/multi paths)
if [[ -z "$DATASET" ]]; then
    echo "Error: DATASET is empty" >&2
    exit 1
fi

# Read into array; respects spaces as delimiters (paths must NOT contain spaces)
read -r -a DATA_PATHS <<< "$DATASET"

# Launch training
torchrun \
    --nproc_per_node="$GPU_NUM" \
    --nnodes="$NUM_NODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    train_lightning.py \
    --config "$CONFIG" \
    --num_nodes "$NUM_NODES" \
    --data_paths "${DATA_PATHS[@]}"