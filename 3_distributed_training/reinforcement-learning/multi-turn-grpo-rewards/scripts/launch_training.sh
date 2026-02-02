#!/bin/bash
# Quick launch script for SageMaker training

# Configuration
INSTANCE_TYPE="ml.g5.24xlarge"  # Change to ml.p4d.24xlarge for 8 GPUs
NUM_GPUS=4
MODEL_NAME="Qwen/Qwen2.5-7B"

# Launch training via AWS CLI
aws sagemaker create-training-job \
    --training-job-name "mt-grpo-$(date +%Y%m%d-%H%M%S)" \
    --role-arn "arn:aws:iam::YOUR_ACCOUNT:role/SageMakerRole" \
    --algorithm-specification TrainingImage="763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.5.1-gpu-py311" \
        TrainingInputMode="File" \
    --resource-config InstanceType="${INSTANCE_TYPE}" \
        InstanceCount=1 \
        VolumeSizeInGB=200 \
    --hyper-parameters \
        model_name="${MODEL_NAME}" \
        num_gpus="${NUM_GPUS}" \
        learning_rate="1e-6" \
        num_generations="8" \
        per_device_train_batch_size="2" \
        grad_accum_steps="4" \
        num_iterations="2" \
        max_steps="300" \
        beta="0" \
        trainer="mt_grpo" \
        turn_advantage_coef="1" \
    --output-data-config S3OutputPath="s3://YOUR_BUCKET/mt-grpo-training/output" \
    --stopping-condition MaxRuntimeInSeconds=86400

echo "Training job launched. Check SageMaker console for status."
