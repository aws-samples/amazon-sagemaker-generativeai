#!/bin/bash
#
# Step 2 entrypoint — GRPO (RLVR) with multi-GPU DDP via torchrun.
#
# Each rank is a full replica and generates rollouts for its shard of prompts.
# With LoRA the KL reference is free (TRL disables the adapter for ref logprobs).
# Set USE_VLLM=1 (needs vLLM in the image) to offload rollouts — much faster.
#
set -euo pipefail

echo "=================================================================="
echo " STEP 2 — GRPO (RLVR)"
echo "=================================================================="

: "${SFT_MODEL_S3:?SFT_MODEL_S3 not set}"
: "${DATA_S3:?DATA_S3 not set}"
: "${OUTPUT_GRPO_S3:?OUTPUT_GRPO_S3 not set}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-4B}"
SEQ_LEN="${SEQ_LEN:-2048}"
SEED="${SEED:-42}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
USE_LORA="${USE_LORA:-1}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
GRPO_EPOCHS="${GRPO_EPOCHS:-1}"
GRPO_LR="${GRPO_LR:-1e-6}"
GRPO_BS="${GRPO_BS:-8}"
GRPO_GRAD_ACCUM="${GRPO_GRAD_ACCUM:-4}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
GRPO_BETA="${GRPO_BETA:-0.04}"
GRPO_TEMPERATURE="${GRPO_TEMPERATURE:-1.0}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-256}"
REWARD_WEIGHTS="${REWARD_WEIGHTS:-0.1,0.3,0.2,0.4}"
USE_VLLM="${USE_VLLM:-0}"

if [ -z "${NUM_GPUS:-}" ]; then
  NUM_GPUS="$(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 1)"
fi

cd "$(dirname "$0")"
echo "sft model      : ${SFT_MODEL_S3}"
echo "data           : ${DATA_S3}"
echo "reward weights : ${REWARD_WEIGHTS}  (format,name,param_name,param_value)"
echo "num_generations: ${NUM_GENERATIONS}   beta: ${GRPO_BETA}"
echo "grpo bs/accum  : ${GRPO_BS}/${GRPO_GRAD_ACCUM}   use_vllm: ${USE_VLLM}"
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
  --reward-weights        "${REWARD_WEIGHTS}"
  --use-vllm              "${USE_VLLM}"
  --output-grpo-s3        "${OUTPUT_GRPO_S3}"
)

if [ "${NUM_GPUS}" -gt 1 ]; then
  echo "launching torchrun with ${NUM_GPUS} processes (DDP)"
  torchrun --standalone --nnodes 1 --nproc_per_node "${NUM_GPUS}" grpo_job.py "${ARGS[@]}"
else
  echo "single GPU — running without torchrun"
  python grpo_job.py "${ARGS[@]}"
fi

echo "STEP 2 complete."
