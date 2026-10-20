#!/bin/bash
#
# Step 1 entrypoint — SFT with real multi-GPU DDP via torchrun.
#
# A shell entrypoint (not a bare .py) is what lets us launch torchrun explicitly.
# Without it SageMaker starts a SINGLE process and HuggingFace Trainer silently
# falls back to legacy DataParallel (replicating from cuda:0 every step) — slow.
#
set -euo pipefail

echo "=================================================================="
echo " STEP 1 — SFT"
echo "=================================================================="

: "${DATA_S3:?DATA_S3 not set}"
: "${OUTPUT_SFT_S3:?OUTPUT_SFT_S3 not set}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-4B}"
SEQ_LEN="${SEQ_LEN:-1024}"
SEED="${SEED:-42}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
USE_LORA="${USE_LORA:-1}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
SFT_EPOCHS="${SFT_EPOCHS:-1}"
SFT_LR="${SFT_LR:-2e-4}"
SFT_BS="${SFT_BS:-4}"
SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-4}"

if [ -z "${NUM_GPUS:-}" ]; then
  NUM_GPUS="$(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 1)"
fi

cd "$(dirname "$0")"
echo "base model : ${BASE_MODEL}"
echo "data       : ${DATA_S3}"
echo "seq_len    : ${SEQ_LEN}   epochs: ${SFT_EPOCHS}   lr: ${SFT_LR}"
echo "lora r/a   : ${LORA_R}/${LORA_ALPHA} (use_lora=${USE_LORA})"
echo "GPUs (DDP) : ${NUM_GPUS}"
echo "output     : ${OUTPUT_SFT_S3}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

ARGS=(
  --base-model     "${BASE_MODEL}"
  --data-s3        "${DATA_S3}"
  --seq-len        "${SEQ_LEN}"
  --seed           "${SEED}"
  --attn-impl      "${ATTN_IMPL}"
  --use-lora       "${USE_LORA}"
  --lora-r         "${LORA_R}"
  --lora-alpha     "${LORA_ALPHA}"
  --sft-epochs     "${SFT_EPOCHS}"
  --sft-lr         "${SFT_LR}"
  --sft-bs         "${SFT_BS}"
  --sft-grad-accum "${SFT_GRAD_ACCUM}"
  --output-sft-s3  "${OUTPUT_SFT_S3}"
)

if [ "${NUM_GPUS}" -gt 1 ]; then
  echo "launching torchrun with ${NUM_GPUS} processes (DDP)"
  torchrun --standalone --nnodes 1 --nproc_per_node "${NUM_GPUS}" sft_job.py "${ARGS[@]}"
else
  echo "single GPU — running without torchrun"
  python sft_job.py "${ARGS[@]}"
fi

echo "STEP 1 complete."
