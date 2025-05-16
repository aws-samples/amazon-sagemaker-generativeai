#!/bin/bash

echo "Running Unsloth training script..."

export TRAIN_FILE="/opt/ml/input/data/train/data.json"
export MODEL_OUTPUT_DIR="/opt/ml/model"
export PER_DEVICE_TRAIN_BATCH_SIZE=4
export GRADIENT_ACCUMULATION_STEPS=2
export NUM_TRAIN_EPOCHS=3

python sagemaker_unsloth_qwen2_5_trainer.py --train_file /opt/ml/input/data/train/data.json
