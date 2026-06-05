#!/bin/bash
set -e

# Make GPUs 4, 5, 6, 7 visible
export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Adjust to your path
HF_CACHE="/vol/tensusers8/aparikh/qwen_simpo_apa_wildfire/cache_dir"
export HF_HOME="${HF_CACHE}"

TRAIN_CSV="/vol/tensusers8/aparikh/dpoftllm/Multi-Granular/data/train_multitask_comprehensive_v2.csv"
OUT_DIR="/vol/tensusers8/aparikh/dpoftllm/Multi-Granular/runs/comprehensive_bdpo_v2"
RESUME_CKPT="${OUT_DIR}/epoch_8"

echo "Resuming training (Epochs 9-12) on GPU 4..."
echo "Auto-evaluation is DISABLED to allow simultaneous eval of Epoch 8 on other GPUs."

python3 train_resume.py \
    --train_csv "$TRAIN_CSV" \
    --out_dir "$OUT_DIR" \
    --resume_from "$RESUME_CKPT" \
    --num_epochs 5 \
    --start_epoch 9 \
    --batch_size 1 \
    --grad_accum 16 \
    --lr 5e-6 \
    --beta 0.1 \
    --alpha 0.5 \
    --sft_weight 1.0 \
