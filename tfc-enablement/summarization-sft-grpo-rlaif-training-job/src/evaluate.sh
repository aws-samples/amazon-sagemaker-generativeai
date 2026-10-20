#!/bin/bash
#
# Step 3 entrypoint — evaluate base vs SFT vs SFT+GRPO (single process, one GPU).
# Uses the EVAL judge, which must differ from the TRAIN judge.
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
SEQ_LEN="${SEQ_LEN:-1024}"
SEED="${SEED:-42}"
N_PAIRWISE="${N_PAIRWISE:-200}"
JUDGE_BACKEND="${JUDGE_BACKEND:-bedrock}"
EVAL_JUDGE_MODEL="${EVAL_JUDGE_MODEL:-openai.gpt-oss-120b-1:0}"
TRAIN_JUDGE_MODEL="${TRAIN_JUDGE_MODEL:-us.anthropic.claude-sonnet-4-5-20250929-v1:0}"
JUDGE_REGION="${JUDGE_REGION:-}"
JUDGE_MAX_CONCURRENCY="${JUDGE_MAX_CONCURRENCY:-8}"
REWARD_WEIGHTS="${REWARD_WEIGHTS:-0.30,0.40,0.15,0.15}"
EVAL_BASE="${EVAL_BASE:-1}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-128}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"

cd "$(dirname "$0")"
echo "data        : ${DATA_S3}"
echo "sft model   : ${SFT_MODEL_S3}"
echo "grpo model  : ${GRPO_MODEL_S3}"
echo "eval judge  : ${JUDGE_BACKEND} / ${EVAL_JUDGE_MODEL}"
echo "train judge : ${TRAIN_JUDGE_MODEL}  (must differ from eval judge)"
echo "output      : ${OUTPUT_RESULTS_S3}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

ARGS=(
  --base-model          "${BASE_MODEL}"
  --data-s3             "${DATA_S3}"
  --sft-model-s3        "${SFT_MODEL_S3}"
  --grpo-model-s3       "${GRPO_MODEL_S3}"
  --seq-len             "${SEQ_LEN}"
  --seed                "${SEED}"
  --n-pairwise          "${N_PAIRWISE}"
  --judge-backend       "${JUDGE_BACKEND}"
  --eval-judge-model    "${EVAL_JUDGE_MODEL}"
  --train-judge-model   "${TRAIN_JUDGE_MODEL}"
  --judge-max-concurrency "${JUDGE_MAX_CONCURRENCY}"
  --reward-weights      "${REWARD_WEIGHTS}"
  --eval-base           "${EVAL_BASE}"
  --eval-max-new-tokens "${EVAL_MAX_NEW_TOKENS}"
  --eval-batch-size     "${EVAL_BATCH_SIZE}"
  --output-results-s3   "${OUTPUT_RESULTS_S3}"
)
if [ -n "${JUDGE_REGION}" ]; then
  ARGS+=( --judge-region "${JUDGE_REGION}" )
fi

python evaluate_job.py "${ARGS[@]}"

echo "STEP 3 complete."
