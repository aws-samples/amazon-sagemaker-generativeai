# Distributed Training on Amazon SageMaker

This directory contains examples of distributed training implementations for Large Language Models (LLMs) on Amazon SageMaker. Each example demonstrates different approaches and frameworks for efficient distributed training.

## Examples

### 1. SageMaker Unified Studio

- [Distributed Training in SageMaker Studio](distributed_training_sm_unified_studio/README.md) - Example showing how to use SageMaker's distributed training capabilities directly from SageMaker Unified Studio.

### 2. Fully-Sharded Data Parallel (FSDP)

- [FSDP](fsdp/README.md) - Examples showing how to use Hugging Face FSDP distributed training capabilities with SageMaker Training Job

### 3. Reinforcement Learning

#### Direct Preference Optimization (DPO)

- [DPO with TRL](reinforcement-learning/dpo/trl/README.md) - Implementation of Direct Preference Optimization using Hugging Face's TRL library.

#### Generative Reward Penalized Optimization (GRPO)

- [GRPO with TRL](reinforcement-learning/grpo/trl/README.md) - Implementation of GRPO using Hugging Face's TRL library with different distributed training approaches:
  - Accelerate-based implementation for simplified distributed training
  - Torchrun-based implementation with FSDP support
- [GRPO with Unsloth](reinforcement-learning/grpo/unsloth/README.md) - GRPO implementation using the Unsloth framework for efficient training with 4-bit quantization.
- [GRPO with veRL](reinforcement-learning/grpo/veRL/README.md) - Advanced GRPO implementation using the veRL framework with support for:
  - Ray-based distribution
  - Megatron model parallelism
  - FSDP (Fully Sharded Data Parallel)
  - Multiple reward functions

### 4. Unsloth Fine-tuning Examples

- [Instruction Fine-tuning Example 1](unsloth/instruct-fine-tuning-example-1/README.md) - Example of instruction fine-tuning using Unsloth.
- [Instruction Fine-tuning Example 2](unsloth/instruct-fine-tuning-example-2/README.md) - Additional example demonstrating Unsloth's capabilities for instruction fine-tuning.

Each example includes detailed documentation and instructions for setting up and running distributed training jobs on Amazon SageMaker.
