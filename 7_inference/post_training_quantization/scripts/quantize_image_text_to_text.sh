#!/usr/bin/env bash
# quantization_all.sh
# Runs quantization for GPTQ and AWQ schemes sequentially on a specified GPU,
# allowing MODEL_ID, DATASET_ID, DATASET_SPLIT and IGNORE_LAYERS to be passed in.

set -euo pipefail

# Usage check
if [ "$#" -lt 5 ]; then
  echo "Usage: $0 <GPU_ID> <MODEL_ID> <DATASET_ID> <DATASET_SPLIT> <IGNORE_LAYERS>"
  echo "Example:"
  echo "  $0 0 \\"
  echo "    meta-llama/Llama-3.1-8B-Instruct \\"
  echo "    HuggingFaceH4/ultrachat_200k \\"
  echo "    train_sft \\"
  echo "    \"lm_head,re:.*block_sparse_moe.gate\""
  exit 1
fi

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPU_ID="$1"
MODEL_ID="$2"
DATASET_ID="$3"
DATASET_SPLIT="$4"
IGNORE_LAYERS="$5"

echo "Using GPU ID:     $GPU_ID"
echo "Model ID:         $MODEL_ID"
echo "Dataset ID:       $DATASET_ID"
echo "Split:            $DATASET_SPLIT"
echo "Ignore Layers:    $IGNORE_LAYERS"

# Sanitize MODEL_ID into directory-friendly name
SANITIZED_MODEL_ID="${MODEL_ID//\//--}"
SM_MODEL_DIR="/home/sagemaker-user/Inference-Optimization/EXPERIMENTS/${SANITIZED_MODEL_ID}"

# Other fixed configuration
DATASET_SEED=42
NUM_CAL_SAMPLES=256
MAX_SEQ_LEN=1024
INCLUDE_TARGETS="Linear"

# Scheme lists
GPTQ_SCHEMES=("W4A16" "W4A16_ASYM" "W8A8" "W8A16")
AWQ_SCHEMES=("W4A16" "W4A16_ASYM")

# Run GPTQ quantization for each scheme
for SCHEME in "${GPTQ_SCHEMES[@]}"; do
  echo "============================================================"
  echo "Running GPTQ with scheme: $SCHEME on GPU $GPU_ID"
  CUDA_VISIBLE_DEVICES="$GPU_ID" python3 post_training_sagemaker_quantizer.py \
    --model-id                   "$MODEL_ID" \
    --dataset-id                 "$DATASET_ID" \
    --dataset-split              "$DATASET_SPLIT" \
    --dataset-seed               "$DATASET_SEED" \
    --algorithm                  gptq \
    --num-calibration-samples   "$NUM_CAL_SAMPLES" \
    --max-sequence-length       "$MAX_SEQ_LEN" \
    --ignore-layers             "$IGNORE_LAYERS" \
    --include-targets           "$INCLUDE_TARGETS" \
    --gptq-quantization-scheme   "$SCHEME" \
    --sm-model-dir              "$SM_MODEL_DIR" \
    --vision-enabled \
    --transformer-model-name Qwen2_5_VLForConditionalGeneration \
    --vision-sequential-targets Qwen2_5_VLDecoderLayer
  echo
done

# Run AWQ quantization for each scheme
for SCHEME in "${AWQ_SCHEMES[@]}"; do
  echo "============================================================"
  echo "Running AWQ with scheme: $SCHEME on GPU $GPU_ID"
  CUDA_VISIBLE_DEVICES="$GPU_ID" python3 post_training_sagemaker_quantizer.py \
    --model-id                   "$MODEL_ID" \
    --dataset-id                 "$DATASET_ID" \
    --dataset-split              "$DATASET_SPLIT" \
    --dataset-seed               "$DATASET_SEED" \
    --algorithm                  awq \
    --num-calibration-samples   "$NUM_CAL_SAMPLES" \
    --max-sequence-length       "$MAX_SEQ_LEN" \
    --ignore-layers             "$IGNORE_LAYERS" \
    --include-targets           "$INCLUDE_TARGETS" \
    --awq-quantization-scheme    "$SCHEME" \
    --sm-model-dir              "$SM_MODEL_DIR" \
    --vision-enabled \
    --transformer-model-name Qwen2_5_VLForConditionalGeneration \
    --vision-sequential-targets Qwen2_5_VLDecoderLayer
  echo
done
