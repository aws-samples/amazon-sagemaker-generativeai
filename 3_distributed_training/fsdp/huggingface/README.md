# SageMaker FSDP Model Trainer

This module provides a framework for fine-tuning large language models on Amazon SageMaker using PyTorch's Fully Sharded Data Parallel (FSDP) and LoRA/QLoRA techniques.

## Overview

The model trainer implements efficient fine-tuning for language models using the following key components:

- FSDP (Fully Sharded Data Parallel) for distributed training
- LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
- 4-bit quantization with bitsandbytes
- MLflow for experiment tracking
- Weights & Biases for monitoring

## Directory Structure

```
scripts/
├── requirements.txt           # Python dependencies
└── train.py                   # Main training script
utils/
├── data_format.py             # Data format utilities
├── function_extraction_utils.py # Function call extraction utilities
└── preprocessing.py           # Dataset preprocessing utilities
model-trainer-notebook.ipynb   # Example notebook
```

## Prerequisites

Install the required dependencies:

```bash
pip install -r scripts/requirements.txt
```

Key dependencies include:

- transformers==4.51.3
- peft==0.15.2
- accelerate==1.6.0
- bitsandbytes==0.45.5
- trl==0.17.0
- sagemaker==2.244.0
- mlflow
- wandb

## Features

### Training Approach

The trainer uses Hugging Face's Trainer with FSDP for efficient distributed training:

1. Supports fine-tuning with LoRA for parameter efficiency
2. Enables 4-bit quantization for memory optimization
3. Provides function calling optimization for tool use

### Training Optimizations

- 4-bit quantization using bitsandbytes
- Gradient checkpointing
- Flash Attention 2 support
- Distributed training with FSDP
- CPU offloading for memory efficiency

### Monitoring & Tracking

- MLflow integration for experiment tracking
- Weights & Biases integration for training monitoring
- Custom GPU metrics logging

## Usage

1. Configure your training parameters in `args.yaml`
2. Prepare your dataset in JSON format
3. Run the training script through SageMaker ModelTrainer

The notebook `model-trainer-notebook.ipynb` provides a complete example workflow:

1. Loading and preprocessing a dataset (function calling dataset example)
2. Configuring training parameters
3. Launching a SageMaker training job
4. Deploying the fine-tuned model

### Key Configuration Parameters

- `model_id`: Hugging Face model identifier
- `lora_r`: LoRA rank
- `lora_alpha`: LoRA alpha parameter
- `lora_dropout`: LoRA dropout rate
- `learning_rate`: Learning rate for optimization
- `num_train_epochs`: Number of training epochs
- `per_device_train_batch_size`: Batch size per device
- `gradient_accumulation_steps`: Steps before parameter update
- `fsdp`: FSDP configuration string
- `fsdp_config`: Detailed FSDP configuration parameters
- `mlflow_uri`: MLflow tracking server URI
- `mlflow_experiment_name`: MLflow experiment name

## Dataset Format

The trainer expects datasets in JSON format with a `text` field containing the formatted conversation. The notebook includes utilities for preprocessing datasets from various formats.

## Model Saving

The trainer supports two saving modes:

1. Saving adapter weights separately (default)
2. Merging adapter weights with the base model (when `merge_weights=True`)

## Monitoring

### MLflow Integration

When enabled, the following is tracked:

- Training metrics
- Model parameters
- Dataset versions
- System metrics

### Weights & Biases Integration

When configured, monitors:

- Training progress
- GPU utilization
- Loss metrics
- Per-GPU metrics

## Error Handling

The trainer includes comprehensive error handling and logging:

- Detailed error messages
- Training state recovery
- Checkpoint management

## Deployment

The notebook includes a complete workflow for deploying the fine-tuned model to a SageMaker endpoint using DJL Inference with:

- FP8 quantization for inference
- vLLM for rolling batch processing
- Streaming response support
- Tool calling capabilities
