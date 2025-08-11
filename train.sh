#!/bin/bash

# =============================================================================
# OCR Error Correction Model Training Script
# =============================================================================
# 
# This script trains language models for OCR error correction using different
# base models and datasets. It supports distributed training with multiple GPUs
# and integrates with Weights & Biases for experiment tracking.
#
# Usage:
#   bash train.sh
#
# Requirements:
#   - CUDA-capable GPUs
#   - PyTorch with CUDA support
#   - transformers, datasets, accelerate libraries
#   - DeepSpeed for distributed training
#
# Author: Gyuho Shim
# =============================================================================

set -e  # Exit on any error

# =============================================================================
# CONFIGURATION SECTION
# =============================================================================

# Global settings
export CUDA_VISIBLE_DEVICES=0,1,2,3
NPROC_PER_NODE=4
DATA_PATH="data/meta_final.jsonl"
CHECKPOINT_BASE_DIR="./ckpt"

# Training hyperparameters
CUTOFF_LEN=2048
WARMUP_RATIO=0.3
LOGGING_STEPS=1
SAVE_STEPS=50
SAVE_TOTAL_LIMIT=3
NUM_EPOCHS=2
GRADIENT_ACCUMULATION_STEPS=2

# =============================================================================
# EXPERIMENT 1: Gemma-2-2B-IT with 5e-05 Learning Rate
# =============================================================================

echo "Starting Experiment 1: Gemma-2-2B-IT (LR: 5e-05)"
echo "=================================================="

export WANDB_PROJECT="document"
export WANDB_NAME="0320_meta_gemma_2"
LR=5e-05

echo "Configuration:"
echo "  Model: google/gemma-2-2b-it"
echo "  Learning Rate: $LR"
echo "  Batch Size: 2 per device"
echo "  Data: $DATA_PATH"
echo "  Output: $CHECKPOINT_BASE_DIR/$WANDB_NAME"
echo ""

torchrun --nproc_per_node=$NPROC_PER_NODE train.py \
    --base_model="google/gemma-2-2b-it" \
    --data_path=$DATA_PATH \
    --wandb_project="$WANDB_PROJECT" \
    --per_device_train_batch_size=2 \
    --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
    --num_epochs=$NUM_EPOCHS \
    --learning_rate $LR \
    --cutoff_len=$CUTOFF_LEN \
    --warmup_ratio=$WARMUP_RATIO \
    --logging_steps=$LOGGING_STEPS \
    --save_steps=$SAVE_STEPS \
    --save_total_limit=$SAVE_TOTAL_LIMIT \
    --bf16 \
    --add_eos_token True \
    --output_dir "$CHECKPOINT_BASE_DIR/$WANDB_NAME" \
    > "$WANDB_NAME.log" 2>&1 

wait
echo "Experiment 1 completed. Log saved to: $WANDB_NAME.log"
echo ""

# =============================================================================
# EXPERIMENT 2: Gemma-2-2B-IT with 6e-05 Learning Rate
# =============================================================================

echo "Starting Experiment 2: Gemma-2-2B-IT (LR: 6e-05)"
echo "=================================================="

export WANDB_PROJECT="document"
export WANDB_NAME="0320_meta_gemma"
LR=6e-05

echo "Configuration:"
echo "  Model: google/gemma-2-2b-it"
echo "  Learning Rate: $LR"
echo "  Batch Size: 2 per device"
echo "  Data: $DATA_PATH"
echo "  Output: $CHECKPOINT_BASE_DIR/$WANDB_NAME"
echo ""

torchrun --nproc_per_node=$NPROC_PER_NODE train.py \
    --base_model="google/gemma-2-2b-it" \
    --data_path=$DATA_PATH \
    --wandb_project="$WANDB_PROJECT" \
    --per_device_train_batch_size=2 \
    --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
    --num_epochs=$NUM_EPOCHS \
    --learning_rate $LR \
    --cutoff_len=$CUTOFF_LEN \
    --warmup_ratio=$WARMUP_RATIO \
    --logging_steps=$LOGGING_STEPS \
    --save_steps=$SAVE_STEPS \
    --save_total_limit=$SAVE_TOTAL_LIMIT \
    --bf16 \
    --add_eos_token True \
    --output_dir "$CHECKPOINT_BASE_DIR/$WANDB_NAME" \
    > "$WANDB_NAME.log" 2>&1 

wait
echo "Experiment 2 completed. Log saved to: $WANDB_NAME.log"
echo ""

echo "All training experiments completed!"
echo "Check individual log files for detailed training information."
echo "Checkpoints saved in: $CHECKPOINT_BASE_DIR/"


