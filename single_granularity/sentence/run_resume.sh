#!/bin/bash
set -e

# --- 1. Environment Setup ---
export CUDA_VISIBLE_DEVICES=7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

HF_CACHE="/vol/tensusers8/aparikh/qwen_simpo_apa_wildfire/cache_dir"
mkdir -p "${HF_CACHE}"
export HF_HOME="${HF_CACHE}"
export HF_DATASETS_CACHE="${HF_CACHE}/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE}/hub"


python /vol/tensusers8/aparikh/dpoftllm/Sentence/train_resume.py \
  --train_csv /vol/tensusers8/aparikh/dpoftllm/Sentence/data/train_sentence_accuracy_strategic_v4.csv \
  --out_dir outputs_continued_accuracy \
  --resume_checkpoint /vol/tensusers8/aparikh/dpoftllm/Sentence/runs/bdpo_sentence_run_final_accuracy/epoch_5 \
  --num_epochs 3 \
  --batch_size 1 \
  --grad_accum 16 \
  --lr 5e-6