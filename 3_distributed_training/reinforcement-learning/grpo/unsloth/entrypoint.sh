#!/bin/bash

echo "Running GRPO Unsloth training script..."

python sagemaker_grpo_training_wb_tracing.py \
    --model_output_dir /opt/ml/model \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE:-1} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS:-1} \
    --num_train_epochs ${NUM_TRAIN_EPOCHS:-1} \
    --hf_model_name meta-llama/Llama-3.2-1B-Instruct \
