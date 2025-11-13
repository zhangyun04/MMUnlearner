#!/bin/bash

# Distributed training script for CLEAR_finetune.py with 8 H100 GPUs
# Usage: bash CLEAR_finetune_distributed.sh

# Set number of GPUs
NUM_GPUS=8

# Training parameters
MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct"
VANILLA_DIR="Qwen/Qwen2.5-VL-7B-Instruct"
DATA_SPLIT_DIR="data/CLEAR"
BATCH_SIZE=1  # Per GPU batch size, total batch size will be BATCH_SIZE * NUM_GPUS
LR=1e-5
NUM_EPOCHS=1
SAVE_DIR="./saved_model"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ACCELERATE_FIND_UNUSED_PARAMETERS=true

# Launch distributed training using torchrun
torchrun --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    CLEAR_finetune.py \
    --model_id $MODEL_ID \
    --vanilla_dir $VANILLA_DIR \
    --data_split_dir $DATA_SPLIT_DIR \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --num_epochs $NUM_EPOCHS \
    --save_dir $SAVE_DIR

