#!/bin/bash
set -e

# --- Environment ---
export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

HF_CACHE="/vol/tensusers8/aparikh/qwen_simpo_apa_wildfire/cache_dir"
mkdir -p "${HF_CACHE}"
export HF_HOME="${HF_CACHE}"

# --- Paths ---
# Use the V2 dataset (Strategic Rejection of Good>Avg AND Good>Exc)
TRAIN_CSV="/vol/tensusers8/aparikh/dpoftllm/Phoneme/data/train_phoneme_strategic_v2.csv"

OUT_DIR="/vol/tensusers8/aparikh/dpoftllm/Phoneme/runs/phoneme_bdpo_v1"

# --- Hyperparameters ---
NUM_EPOCHS=8          
BATCH_SIZE=1          
GRAD_ACCUM=16         
LR=5e-6               
LORA_R=64

# --- DPO Config ---
BETA=0.1
ALPHA=0.5             
SFT_WEIGHT=1.0        

# --- Run ---
echo "Starting Phoneme-Level BDPO Training..."
echo "Train: $TRAIN_CSV"

python3 /vol/tensusers8/aparikh/dpoftllm/Phoneme/phoneme_bdpo_ft.py \
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