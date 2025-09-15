# SageMaker DDP Model Trainer

This module provides a framework for fine-tuning large language models on Amazon SageMaker using PyTorch's Distributed Data Parallel (DDP) and LoRA/QLoRA techniques.

## Overview

The model trainer implements efficient fine-tuning for language models using the following key components:

- DDP (Distributed Data Parallel) for distributed training
- LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
- 4-bit quantization with bitsandbytes and MXFP4
- MLflow for experiment tracking

## Directory Structure

```
scripts/
├── requirements.txt           # Python dependencies
└── train.py                   # Main training script
model-trainer-ddp.ipynb        # Example notebook
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

The trainer uses Hugging Face's Trainer with DDP for efficient distributed training:

1. Supports fine-tuning with LoRA for parameter efficiency
2. Enables 4-bit quantization (BitsAndBytes) for memory optimization
3. Provides flexible attention implementations (Flash Attention 2)

### Training Optimizations

- 4-bit quantization using bitsandbytes
- Gradient checkpointing with non-reentrant mode
- Flash Attention 2 support
- Distributed training with DDP
- Configurable torch dtype (auto, bfloat16, float16, float32)

### Monitoring & Tracking

- MLflow integration for experiment tracking
- Weights & Biases integration for training monitoring
- Custom GPU metrics logging

## Usage

1. Configure your training parameters in `args.yaml`
2. Prepare your dataset in JSON/JSONL format
3. Run the training script through SageMaker ModelTrainer

The notebook `model-trainer-ddp.ipynb` provides a complete example workflow:

1. Loading and preprocessing a dataset
2. Configuring training parameters for DDP
3. Launching a SageMaker training job
4. Deploying the fine-tuned model

### Key Configuration Parameters

- `model_id`: Hugging Face model identifier
- `torch_dtype`: Torch data type (auto, bfloat16, float16, float32)
- `attn_implementation`: Attention implementation (flash_attention_2, sdpa, eager)
- `load_in_4bit`: Enable 4-bit quantization
- `lora_r`: LoRA rank
- `lora_alpha`: LoRA alpha parameter
- `lora_dropout`: LoRA dropout rate
- `learning_rate`: Learning rate for optimization
- `num_train_epochs`: Number of training epochs
- `per_device_train_batch_size`: Batch size per device
- `gradient_accumulation_steps`: Steps before parameter update
- `gradient_checkpointing`: Enable gradient checkpointing
- `mlflow_uri`: MLflow tracking server URI
- `mlflow_experiment_name`: MLflow experiment name

## Dataset Format

The trainer supports datasets in JSON and JSONL formats with a `text` field containing the formatted conversation. The trainer automatically detects file format and handles both single files and directory structures.

## Model Saving

The trainer supports two saving modes:

1. Saving adapter weights separately (default)
2. Merging adapter weights with the base model (when `merge_weights=True`)

## Distribution Strategy Detection

The trainer automatically detects the distribution strategy:

- **DDP**: Default distributed training approach

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
- Automatic report_to configuration based on enabled services

## Deployment

The notebook includes a complete workflow for deploying the fine-tuned model to a SageMaker endpoint with optimized inference configurations for production use.
