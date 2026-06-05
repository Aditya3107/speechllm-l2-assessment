#!/bin/bash
set -e

# ONLY expose the empty GPUs (4, 5, 6, 7)
export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

HF_CACHE="/vol/tensusers8/aparikh/qwen_simpo_apa_wildfire/cache_dir"
export HF_HOME="${HF_CACHE}"

TRAIN_CSV="/vol/tensusers8/aparikh/dpoftllm/Multi-Granular/data/train_multitask_comprehensive_v2.csv"
OUT_DIR="/vol/tensusers8/aparikh/dpoftllm/Multi-Granular/runs/comprehensive_bdpo_v2"

echo "Starting Comprehensive Training..."
echo "Training on Physical GPU 4 (Visible 0)"
echo "Eval on Physical GPUs 5,6,7 (Visible 1,2,3)"

python3 train_mg_bdpo.py \
    --train_csv "$TRAIN_CSV" \
    --out_dir "$OUT_DIR" \
    --num_epochs 8 \
    --batch_size 1 \
    --grad_accum 16 \
    --lr 5e-6 \
    --lora_r 64 \
    --beta 0.1 \
    --alpha 0.5 \
    --sft_weight 1.0 \
    --seed 42