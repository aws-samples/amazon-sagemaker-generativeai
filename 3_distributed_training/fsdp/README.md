# Fully Sharded Data Parallel (FSDP) Training on Amazon SageMaker

This directory contains implementations of various Fully Sharded Data Parallel (FSDP) approaches for training Large Language Models (LLMs) on Amazon SageMaker. Each implementation demonstrates different techniques and frameworks for efficient distributed training of LLMs using FSDP.

## Overview

Fully Sharded Data Parallel (FSDP) is a distributed training technique that enables efficient training of large language models by sharding model parameters, gradients, and optimizer states across multiple GPUs. This repository includes implementations of several FSDP approaches:

- **Hugging Face FSDP**: Implementation using Hugging Face's Transformers library with PyTorch FSDP.

## Implementations

### Hugging Face FSDP

- [**Hugging Face FSDP**](huggingface/README.md): Implementation using Hugging Face's Trainer with FSDP.
  - Supports LoRA/QLoRA for parameter-efficient fine-tuning
  - Includes 4-bit quantization with bitsandbytes
  - Provides MLflow integration for experiment tracking
  - Supports Weights & Biases for monitoring
  - Includes CPU offloading for memory efficiency

## Key Features

- **Memory Efficiency**: FSDP significantly reduces memory usage by sharding model parameters across devices.
- **Distributed Training**: All implementations support various forms of distributed training to handle large models efficiently.
- **Quantization**: Support for different quantization techniques to reduce memory requirements.
- **Parameter-Efficient Fine-Tuning**: Support for LoRA and QLoRA for efficient fine-tuning.
- **Experiment Tracking**: Integration with MLflow and Weights & Biases for comprehensive experiment tracking.
- **Checkpointing**: Support for saving and loading checkpoints during training.

## Getting Started

Each implementation directory contains detailed documentation and instructions for setting up and running FSDP training jobs on Amazon SageMaker. Refer to the specific README files in each directory for more information.

## Requirements

Requirements vary by implementation, but generally include:

- PyTorch (version 2.0 or later for best FSDP support)
- Transformers
- Accelerate
- SageMaker SDK
- PEFT (for parameter-efficient fine-tuning)
- bitsandbytes (for quantization)

## Performance Considerations

When using FSDP, consider the following factors that can impact training performance:

- **Sharding Strategy**: Different sharding strategies (FULL_SHARD, SHARD_GRAD_OP, etc.) offer different trade-offs between memory efficiency and communication overhead.
- **CPU Offloading**: Offloading to CPU can reduce GPU memory usage but may increase training time.
- **Mixed Precision**: Using mixed precision (BF16/FP16) can significantly improve training speed and reduce memory usage.
- **Gradient Checkpointing**: Can be combined with FSDP to further reduce memory usage at the cost of recomputation.
- **Communication Bandwidth**: High-speed interconnects (like NVLink or InfiniBand) are crucial for efficient FSDP training.
