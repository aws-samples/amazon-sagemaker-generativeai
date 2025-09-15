# Distributed Data Parallel (DDP) Training on Amazon SageMaker AI

This directory contains implementations of various Distributed Data Parallel (DDP) approaches for training Large Language Models (LLMs) on Amazon SageMaker AI. Each implementation demonstrates different techniques and frameworks for efficient distributed training of LLMs using DDP.

## Overview

Distributed Data Parallel (DDP) is a distributed training technique that enables efficient training of large language models by replicating the model across multiple GPUs and synchronizing gradients during backpropagation. This repository includes implementations of several DDP approaches:

- **Hugging Face DDP**: Implementation using Hugging Face's Transformers library with PyTorch DDP.

## Implementations

### Hugging Face DDP

- [**Hugging Face DDP**](huggingface/README.md): Implementation using Hugging Face's Trainer with DDP.
  - Supports LoRA/QLoRA for parameter-efficient fine-tuning
  - Includes 4-bit quantization with bitsandbytes
  - Provides MLflow integration for experiment tracking
  - Supports Weights & Biases for monitoring
  - Includes gradient accumulation for effective large batch sizes

## Key Features

- **Scalability**: DDP enables training across multiple GPUs and nodes with linear scaling.
- **Distributed Training**: All implementations support various forms of distributed training to handle large models efficiently.
- **Quantization**: Support for different quantization techniques to reduce memory requirements.
- **Parameter-Efficient Fine-Tuning**: Support for LoRA and QLoRA for efficient fine-tuning.
- **Experiment Tracking**: Integration with MLflow and Weights & Biases for comprehensive experiment tracking.
- **Checkpointing**: Support for saving and loading checkpoints during training.

## Getting Started

Each implementation directory contains detailed documentation and instructions for setting up and running DDP training jobs on Amazon SageMaker. Refer to the specific README files in each directory for more information.

## Requirements

Requirements vary by implementation, but generally include:

- PyTorch (version 1.8 or later for DDP support)
- Transformers
- Accelerate
- SageMaker SDK
- PEFT (for parameter-efficient fine-tuning)
- bitsandbytes (for quantization)

## Performance Considerations

When using DDP, consider the following factors that can impact training performance:

- **Batch Size**: Larger batch sizes generally improve DDP efficiency by reducing communication overhead relative to computation.
- **Gradient Accumulation**: Use gradient accumulation to achieve effective large batch sizes when memory is limited.
- **Mixed Precision**: Using mixed precision (BF16/FP16) can significantly improve training speed and reduce memory usage.
- **Gradient Checkpointing**: Can be used to reduce memory usage at the cost of recomputation.
- **Communication Backend**: NCCL is typically the most efficient backend for GPU-to-GPU communication in DDP.
- **Network Bandwidth**: High-speed interconnects (like NVLink or InfiniBand) are crucial for efficient gradient synchronization across nodes.
