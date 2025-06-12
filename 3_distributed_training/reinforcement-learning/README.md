# Reinforcement Learning for LLMs on Amazon SageMaker

This directory contains implementations of various reinforcement learning approaches for training Large Language Models (LLMs) on Amazon SageMaker. Each implementation demonstrates different techniques and frameworks for efficient distributed training of LLMs using reinforcement learning.

## Overview

Reinforcement Learning (RL) has become a critical component in aligning Large Language Models with human preferences and improving their capabilities. This repository includes implementations of several state-of-the-art RL techniques:

- **Direct Preference Optimization (DPO)**: A method that directly optimizes language models based on human preferences without requiring a separate reward model.
- **Group Relative Policy Optimization (GRPO)**: An approach that uses reward signals to guide model optimization while maintaining coherence with the original model.

## Implementations

### Direct Preference Optimization (DPO)

- [**DPO with TRL**](dpo/trl/README.md): Implementation of Direct Preference Optimization using Hugging Face's TRL library.
  - Supports distributed training with FSDP
  - Includes LoRA fine-tuning capabilities
  - Provides MLflow integration for experiment tracking

### Group Relative Policy Optimization (GRPO)

- [**GRPO with TRL**](grpo/trl/README.md): Implementation of GRPO using Hugging Face's TRL library with different distributed training approaches:

  - Accelerate-based implementation for simplified distributed training
  - Torchrun-based implementation with FSDP support
  - Comprehensive monitoring and experiment tracking

- [**GRPO with Unsloth**](grpo/unsloth/README.md): GRPO implementation using the Unsloth framework for efficient training.

  - 4-bit quantization support
  - Single-GPU training focus
  - Optimized for memory efficiency

- [**GRPO with veRL**](grpo/veRL/README.md): Advanced GRPO implementation using the veRL framework.
  - Ray-based distribution for scalable training
  - Support for Megatron-style model parallelism
  - FSDP (Fully Sharded Data Parallel) integration
  - Sequence parallelism support
  - Multiple reward function implementations
  - Tool-augmented training capabilities

## Key Features

- **Distributed Training**: All implementations support various forms of distributed training to handle large models efficiently.
- **Quantization**: Support for different quantization techniques to reduce memory requirements.
- **Model Parallelism**: Advanced implementations support tensor and pipeline parallelism for very large models.
- **Experiment Tracking**: Integration with MLflow and Weights & Biases for comprehensive experiment tracking.
- **Reward Functions**: Flexible reward function implementations for different use cases.

## Getting Started

Each implementation directory contains detailed documentation and instructions for setting up and running reinforcement learning training jobs on Amazon SageMaker. Refer to the specific README files in each directory for more information.

## Requirements

Requirements vary by implementation, but generally include:

- PyTorch
- Transformers
- TRL (Transformer Reinforcement Learning)
- SageMaker SDK
- Accelerate or other distributed training libraries
