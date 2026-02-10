#!/bin/bash
# Chronos-2 Fine-Tuning Launch Script for SageMaker
# Usage: ./train.sh <recipe_path>
# Example: ./train.sh hf_recipes/amazon/chronos-2--full.yaml

set -e

RECIPE_PATH=${1:-"hf_recipes/amazon/chronos-2--full.yaml"}

echo "=============================================="
echo "Chronos-2 Fine-Tuning"
echo "Recipe: ${RECIPE_PATH}"
echo "=============================================="

# install uv
pip install uv

# Install dependencies if needed
uv pip install -r requirements.txt --system

# Run training with accelerate
accelerate launch \
    --mixed_precision bf16 \
    chronos2_finetune.py \
    "${RECIPE_PATH}"

echo "=============================================="
echo "Training completed!"
echo "=============================================="
