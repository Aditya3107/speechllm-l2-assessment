#!/bin/bash
set -e

# --- Configuration ---
# GPUs to use for evaluation (Adjust as needed)
EVAL_GPUS="2,3,4,5,6,7"

# Epoch to evaluate (Change this to the epoch you want)
EPOCH_TO_EVAL=9

HF_CACHE="/vol/tensusers8/aparikh/qwen_simpo_apa_wildfire/cache_dir"
export HF_HOME="${HF_CACHE}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Paths (Ensure these match your setup)
TEST_CSV="/vol/tensusers8/aparikh/dpoftllm/Phoneme/data/test_preprocess_with_audio_labels.csv"
OUT_DIR="/vol/tensusers8/aparikh/dpoftllm/Phoneme/runs/phoneme_bdpo_v1"
CKPT_PATH="${OUT_DIR}/epoch_${EPOCH_TO_EVAL}"
LOG_FILE="${OUT_DIR}/training_log.csv"

# --- Launch Evaluation ---
echo "=================================================="
echo "STARTING MANUAL EVALUATION FOR EPOCH ${EPOCH_TO_EVAL}"
echo "GPUs: ${EVAL_GPUS}"
echo "Checkpoint: ${CKPT_PATH}"
echo "=================================================="

# Check if checkpoint exists
if [ ! -f "${CKPT_PATH}/adapter_config.json" ]; then
    echo "[ERROR] Checkpoint not found at ${CKPT_PATH}"
    exit 1
fi

# Launch Eval
# We use 'env' to set CUDA_VISIBLE_DEVICES only for this command
env CUDA_VISIBLE_DEVICES=${EVAL_GPUS} python3 standalone_eval_dynamic.py \
    --model_path "$CKPT_PATH" \
    --test_csv "$TEST_CSV" \
    --out_dir "$OUT_DIR" \
    --epoch "$EPOCH_TO_EVAL" \
    --log_file "$LOG_FILE" \
    --batch_size 16

echo "Finished Eval for Epoch ${EPOCH_TO_EVAL}. Results saved."