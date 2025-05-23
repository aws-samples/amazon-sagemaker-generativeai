# Distributed Training on Amazon SageMaker

This directory contains examples of distributed training implementations for Large Language Models (LLMs) on Amazon SageMaker. Each example demonstrates different approaches and frameworks for efficient distributed training.

## Examples

### 1. Distributed Training in SageMaker Studio

- [Distributed Training in SageMaker Studio](distributed_training_sm_unified_studio/README.md) - Example showing how to use SageMaker's distributed training capabilities directly from Studio.

### 2. Reinforcement Learning

- [GRPO with TRL](reinforcement-learning/grpo/trl/README.md) - Implementation of Generative Reward Penalized Optimization (GRPO) using Hugging Face's TRL library.
- [GRPO with Unsloth](reinforcement-learning/grpo/unsloth/README.md) - GRPO implementation using the Unsloth framework for efficient training.
- [GRPO with veRL](reinforcement-learning/grpo/veRL/README.md) - Advanced GRPO implementation using the veRL framework.

### 3. Unsloth Fine-tuning Examples

- [Instruction Fine-tuning Example 1](unsloth/instruct-fine-tuning-example-1/README.md) - Example of instruction fine-tuning using Unsloth.
- [Instruction Fine-tuning Example 2](unsloth/instruct-fine-tuning-example-2/README.md) - Additional example demonstrating Unsloth's capabilities for instruction fine-tuning.

Each example includes detailed documentation and instructions for setting up and running distributed training jobs on Amazon SageMaker.
