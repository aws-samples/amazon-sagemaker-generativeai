#!/bin/bash
#
# Step 2 entrypoint — GRPO (RLAIF) with multi-GPU DDP via torchrun.
#
# Each rank is a full replica and generates rollouts for its shard of prompts;
# each rollout summary is scored by the TRAINING judge (judge_reward.py). With LoRA
# the KL reference is free (TRL disables the adapter for ref logprobs).
#
set -euo pipefail

echo "=================================================================="
echo " STEP 2 — GRPO (RLAIF)"
echo "=================================================================="

: "${SFT_MODEL_S3:?SFT_MODEL_S3 not set}"
: "${DATA_S3:?DATA_S3 not set}"
: "${OUTPUT_GRPO_S3:?OUTPUT_GRPO_S3 not set}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-4B}"
SEQ_LEN="${SEQ_LEN:-1024}"
SEED="${SEED:-42}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
USE_LORA="${USE_LORA:-1}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
GRPO_EPOCHS="${GRPO_EPOCHS:-1}"
GRPO_LR="${GRPO_LR:-5e-6}"
GRPO_BS="${GRPO_BS:-8}"
GRPO_GRAD_ACCUM="${GRPO_GRAD_ACCUM:-4}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
GRPO_BETA="${GRPO_BETA:-0.01}"
GRPO_TEMPERATURE="${GRPO_TEMPERATURE:-1.0}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-128}"
JUDGE_BACKEND="${JUDGE_BACKEND:-bedrock}"
TRAIN_JUDGE_MODEL="${TRAIN_JUDGE_MODEL:-us.anthropic.claude-sonnet-4-5-20250929-v1:0}"
JUDGE_REGION="${JUDGE_REGION:-}"
JUDGE_MAX_CONCURRENCY="${JUDGE_MAX_CONCURRENCY:-8}"
REWARD_WEIGHTS="${REWARD_WEIGHTS:-0.30,0.40,0.15,0.15}"
REWARD_MODE="${REWARD_MODE:-guarded}"
GUARD_WEIGHTS="${GUARD_WEIGHTS:-0.45,0.30,0.25}"
FAITH_GATE="${FAITH_GATE:-0.40}"
GATE_MULT="${GATE_MULT:-0.30}"
COMPARE_AGAINST="${COMPARE_AGAINST:-sft}"

if [ -z "${NUM_GPUS:-}" ]; then
  NUM_GPUS="$(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 1)"
fi

cd "$(dirname "$0")"
echo "sft model      : ${SFT_MODEL_S3}"
echo "data           : ${DATA_S3}"
echo "judge          : ${JUDGE_BACKEND} / ${TRAIN_JUDGE_MODEL}"
echo "reward mode    : ${REWARD_MODE}"
echo "rubric weights : ${REWARD_WEIGHTS}  (coverage,faithfulness,coherence,conciseness)"
echo "guard weights  : ${GUARD_WEIGHTS}  (preference,faithfulness,coverage)  gate=${FAITH_GATE}x${GATE_MULT}"
echo "compare against: ${COMPARE_AGAINST}  (sft = judge rollouts vs the SFT model's own output)"
echo "num_generations: ${NUM_GENERATIONS}   beta: ${GRPO_BETA}   lr: ${GRPO_LR}"
echo "GPUs (DDP)     : ${NUM_GPUS}"
echo "output         : ${OUTPUT_GRPO_S3}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

ARGS=(
  --base-model            "${BASE_MODEL}"
  --sft-model-s3          "${SFT_MODEL_S3}"
  --data-s3               "${DATA_S3}"
  --seq-len               "${SEQ_LEN}"
  --seed                  "${SEED}"
  --attn-impl             "${ATTN_IMPL}"
  --use-lora              "${USE_LORA}"
  --lora-r                "${LORA_R}"
  --lora-alpha            "${LORA_ALPHA}"
  --grpo-epochs           "${GRPO_EPOCHS}"
  --grpo-lr               "${GRPO_LR}"
  --grpo-bs               "${GRPO_BS}"
  --grpo-grad-accum       "${GRPO_GRAD_ACCUM}"
  --num-generations       "${NUM_GENERATIONS}"
  --grpo-beta             "${GRPO_BETA}"
  --grpo-temperature      "${GRPO_TEMPERATURE}"
  --max-completion-length "${MAX_COMPLETION_LENGTH}"
  --judge-backend         "${JUDGE_BACKEND}"
  --train-judge-model     "${TRAIN_JUDGE_MODEL}"
  --judge-max-concurrency "${JUDGE_MAX_CONCURRENCY}"
  --reward-weights        "${REWARD_WEIGHTS}"
  --reward-mode           "${REWARD_MODE}"
  --guard-weights         "${GUARD_WEIGHTS}"
  --faith-gate            "${FAITH_GATE}"
  --gate-mult             "${GATE_MULT}"
  --compare-against       "${COMPARE_AGAINST}"
  --output-grpo-s3        "${OUTPUT_GRPO_S3}"
)
if [ -n "${JUDGE_REGION}" ]; then
  ARGS+=( --judge-region "${JUDGE_REGION}" )
fi

if [ "${NUM_GPUS}" -gt 1 ]; then
  echo "launching torchrun with ${NUM_GPUS} processes (DDP)"
  torchrun --standalone --nnodes 1 --nproc_per_node "${NUM_GPUS}" grpo_job.py "${ARGS[@]}"
else
  echo "single GPU — running without torchrun"
  python grpo_job.py "${ARGS[@]}"
fi

echo "STEP 2 complete."
