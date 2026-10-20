#!/bin/bash
#
# Step 3 entrypoint — evaluate base vs SFT vs SFT+GRPO (single process, one GPU).
#
set -euo pipefail

echo "=================================================================="
echo " STEP 3 — EVALUATE"
echo "=================================================================="

: "${DATA_S3:?DATA_S3 not set}"
: "${SFT_MODEL_S3:?SFT_MODEL_S3 not set}"
: "${GRPO_MODEL_S3:?GRPO_MODEL_S3 not set}"
: "${OUTPUT_RESULTS_S3:?OUTPUT_RESULTS_S3 not set}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-4B}"
SEQ_LEN="${SEQ_LEN:-2048}"
SEED="${SEED:-42}"
REWARD_WEIGHTS="${REWARD_WEIGHTS:-0.1,0.3,0.2,0.4}"
EVAL_BASE="${EVAL_BASE:-1}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-256}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"

cd "$(dirname "$0")"
echo "data       : ${DATA_S3}"
echo "sft model  : ${SFT_MODEL_S3}"
echo "grpo model : ${GRPO_MODEL_S3}"
echo "eval base  : ${EVAL_BASE}"
echo "output     : ${OUTPUT_RESULTS_S3}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

python evaluate_job.py \
  --base-model         "${BASE_MODEL}" \
  --data-s3            "${DATA_S3}" \
  --sft-model-s3       "${SFT_MODEL_S3}" \
  --grpo-model-s3      "${GRPO_MODEL_S3}" \
  --seq-len            "${SEQ_LEN}" \
  --seed               "${SEED}" \
  --reward-weights     "${REWARD_WEIGHTS}" \
  --eval-base          "${EVAL_BASE}" \
  --eval-max-new-tokens "${EVAL_MAX_NEW_TOKENS}" \
  --eval-batch-size    "${EVAL_BATCH_SIZE}" \
  --output-results-s3  "${OUTPUT_RESULTS_S3}"

echo "STEP 3 complete."
