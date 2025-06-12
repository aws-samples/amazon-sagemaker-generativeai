# SageMaker Training Job with DPO Model Trainer

This module provides a framework for training language models using Direct Preference Optimization (DPO) on Amazon SageMaker AI.
It supports distributed training, quantization, and various optimization techniques including LoRA (Low-Rank Adaptation).

## Overview

The model trainer implements preference-based learning for language models using the following key components:

- DPO (Direct Preference Optimization) for training
- LoRA for efficient fine-tuning
- MLflow for experiment tracking
- Weights & Biases for monitoring
- Support for function calling optimization

## Directory Structure

```
trl/
├── scripts/
│   ├── requirements.txt           # Python dependencies
│   └── train.py                  # Main training script
├── args.yaml                     # Training configuration
└── model-trainer-notebook.ipynb  # Example notebook
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

The trainer uses Direct Preference Optimization (DPO), which:

1. Learns from human preferences between pairs of model outputs
2. Optimizes the model to align with preferred responses
3. Supports function calling optimization for tool use

### Training Optimizations

- 4-bit quantization using bitsandbytes
- Gradient checkpointing
- Flash Attention 2 support
- Distributed training with FSDP
- LoRA fine-tuning

### Monitoring & Tracking

- MLflow integration for experiment tracking
- Weights & Biases integration for training monitoring
- Custom GPU metrics logging

## Usage

1. Configure your training parameters in `args.yaml`
2. Prepare your dataset in JSON format with chosen and rejected responses
3. Run the training script:

```bash
python scripts/train.py --config args.yaml
```

### Key Configuration Parameters

- `model_id`: Hugging Face model identifier
- `lora_r`: LoRA rank
- `lora_alpha`: LoRA alpha parameter
- `lora_dropout`: LoRA dropout rate
- `learning_rate`: Learning rate for optimization
- `num_train_epochs`: Number of training epochs
- `per_device_train_batch_size`: Batch size per device
- `gradient_accumulation_steps`: Steps before parameter update
- `mlflow_uri`: MLflow tracking server URI
- `mlflow_experiment_name`: MLflow experiment name

## Dataset Format

The DPO trainer expects a dataset with the following structure:

- `prompt`: The input prompt/context
- `chosen`: The preferred model response
- `rejected`: The less preferred model response

This format enables the model to learn from human preferences between response pairs.

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

## Contributing

When contributing to this project:

1. Follow the existing code style
2. Add appropriate error handling
3. Update documentation as needed
4. Add tests for new features
