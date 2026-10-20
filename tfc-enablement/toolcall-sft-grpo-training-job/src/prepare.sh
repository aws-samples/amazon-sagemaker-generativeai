#!/bin/bash
#
# Step 0 entrypoint — prepare data (CPU, no torchrun).
# Config arrives as ENVIRONMENT VARIABLES set by the launcher's environment={...}.
#
set -euo pipefail

echo "=================================================================="
echo " STEP 0 — PREPARE DATA"
echo "=================================================================="

: "${OUTPUT_DATA_S3:?OUTPUT_DATA_S3 not set}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-4B}"
XLAM_DATASET="${XLAM_DATASET:-Salesforce/xlam-function-calling-60k}"
N_TRAIN="${N_TRAIN:-8000}"
N_GRPO="${N_GRPO:-3000}"
N_EVAL="${N_EVAL:-500}"
SEQ_LEN="${SEQ_LEN:-2048}"
SEED="${SEED:-42}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"

cd "$(dirname "$0")"
echo "base model : ${BASE_MODEL}"
echo "dataset    : ${XLAM_DATASET}"
echo "splits     : train=${N_TRAIN} grpo=${N_GRPO} eval=${N_EVAL}"
echo "output     : ${OUTPUT_DATA_S3}"

python prepare_job.py \
  --base-model      "${BASE_MODEL}" \
  --xlam-dataset    "${XLAM_DATASET}" \
  --n-train         "${N_TRAIN}" \
  --n-grpo          "${N_GRPO}" \
  --n-eval          "${N_EVAL}" \
  --seq-len         "${SEQ_LEN}" \
  --seed            "${SEED}" \
  --enable-thinking "${ENABLE_THINKING}" \
  --output-data-s3  "${OUTPUT_DATA_S3}"

echo "STEP 0 complete."
