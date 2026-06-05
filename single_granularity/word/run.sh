#!/bin/bash
set -e

# --- 1. Environment Setup ---
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

HF_CACHE="/vol/tensusers8/aparikh/qwen_simpo_apa_wildfire/cache_dir"
mkdir -p "${HF_CACHE}"
export HF_HOME="${HF_CACHE}"
export HF_DATASETS_CACHE="${HF_CACHE}/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE}/hub"

# --- 2. Paths ---
# Using the V5 Inline-Separated dataset
TRAIN_CSV="/vol/tensusers8/aparikh/dpoftllm/Words/data/train_dpo_inline_word_accuracy.csv"

# You should probably use the same file for testing to verify it learns the format, 
# or a hold-out set formatted identically.

OUT_DIR="/vol/tensusers8/aparikh/dpoftllm/Words/runs/word_level_bdpo_v1"

# --- 3. Hyperparameters ---
NUM_EPOCHS=6
BATCH_SIZE=1      
GRAD_ACCUM=16     
LR=5e-6
LORA_R=64

# DPO & SFT
BETA=0.1          
ALPHA=0.5         # BDPO Alpha
SFT_WEIGHT=1.0    # Crucial for preventing mode collapse in word sequences

# --- 4. Execution ---
echo "Starting Word-Level BDPO Training..."
echo "Train CSV: $TRAIN_CSV"
echo "Output: $OUT_DIR"

python3 /vol/tensusers8/aparikh/dpoftllm/Words/words_bdpo_ft.py \
    --train_csv "$TRAIN_CSV" \
    --out_dir "$OUT_DIR" \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --lr $LR \
    --lora_r $LORA_R \
    --beta $BETA \
    --alpha $ALPHA \
    --sft_weight $SFT_WEIGHT \
    --seed 42