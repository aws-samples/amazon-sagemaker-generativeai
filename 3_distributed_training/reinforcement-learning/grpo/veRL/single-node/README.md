# veRL: Distributed Reinforcement Learning Framework for Large Language Models

veRL is a distributed reinforcement learning framework designed specifically for training and fine-tuning large language models (LLMs). It supports various training approaches including PPO (Proximal Policy Optimization), GRPO, and other RL algorithms.

## Key Features

- Distributed training support with Ray
- Multiple model architectures (Qwen, LLaMA, etc.)
- Integration with popular LLM frameworks (vLLM, SGLang)
- Support for Megatron-style model parallelism
- FSDP (Fully Sharded Data Parallel) training
- Flexible reward modeling and management
- Multi-turn conversation support
- Various dataset preprocessing utilities

## Project Structure

```
veRL/
├── container/            # Docker container configuration
├── models/              # Model implementations and adapters
├── scripts/             # Training and example scripts
│   ├── data_preprocess/ # Data preprocessing utilities
│   ├── generation/      # Generation scripts
│   ├── grpo_trainer/    # GRPO training scripts
│   ├── ppo_trainer/     # PPO training scripts
│   └── tuning/         # Model tuning configurations
├── trainer/             # Core training implementations
├── utils/              # Utility functions and helpers
└── workers/            # Worker implementations for distributed training
```

## Supported Models

- Qwen (2.0, 2.5, 3.0)
- LLaMA-based models
- DeepSeek
- Gemma

## Training Methods

1. GRPO (Generalized Reinforcement Policy Optimization)
2. PPO (Proximal Policy Optimization)
3. SFT (Supervised Fine-Tuning)
4. Reinforce++
5. RLOO (Reinforcement Learning with Own-Opponent)
6. ReMax

## Getting Started

1. Refer to `verl-on-sagemaker.ipynb`.

## Features

### Distributed Training

- Ray-based distribution
- Megatron model parallelism
- FSDP (Fully Sharded Data Parallel)
- Sequence parallelism
- Pipeline parallelism

### Model Support

- Checkpoint loading/saving
- Weight sharding
- Mixed precision training
- Tensor parallelism

### Training Features

- Multi-turn conversation
- Tool-augmented training
- Custom reward functions
- Sequence length balancing
- Memory-efficient training
