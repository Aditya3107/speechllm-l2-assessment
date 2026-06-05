#!/bin/bash
set -e

# --- 1. Environment Setup ---
export CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

HF_CACHE="/vol/tensusers8/aparikh/qwen_simpo_apa_wildfire/cache_dir"
mkdir -p "${HF_CACHE}"
export HF_HOME="${HF_CACHE}"
export HF_DATASETS_CACHE="${HF_CACHE}/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE}/hub"

# --- 2. Paths ---
# Point this to your FIXED dataset (avoid the Reverse Preference issue!)
TRAIN_CSV="/vol/tensusers8/aparikh/dpoftllm/Sentence/data/train_prosody_strategic_v1.csv"
# TRAIN_CSV="train_sentence_bins_fixed_v3.csv" 

OUT_DIR="/vol/tensusers8/aparikh/dpoftllm/Sentence/runs/bdpo_sentence_prosody"

# --- 3. Hyperparameters ---
NUM_EPOCHS=5
BATCH_SIZE=1      
GRAD_ACCUM=16     
LR=5e-6
LORA_R=64

# BDPO Specifics
BETA=0.1          
ALPHA=0.5         # 0.5 is standard for BDPO

# --- 4. Execution ---
echo "Starting BDPO Training with Per-Epoch Evaluation..."
echo "Output Dir: $OUT_DIR"

# Assumes python script is named train_bdpo.py
python3 /vol/tensusers8/aparikh/dpoftllm/Sentence/sentence_ft_bdpo_gai_prosody.py \
    --train_csv "$TRAIN_CSV" \
    --out_dir "$OUT_DIR" \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --lr $LR \
    --lora_r $LORA_R \
    --beta $BETA \
    --alpha $ALPHA \
    --seed 42