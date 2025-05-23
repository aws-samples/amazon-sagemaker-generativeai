# GRPO Training on Amazon SageMaker

This directory provides implementations of Generative Reward Penalized Optimization (GRPO) training using different distributed training approaches on Amazon SageMaker.

## Overview

The repository offers two distinct implementations for GRPO training:

1. **Accelerate-based Implementation** ([accelerate/](accelerate/)): Uses Hugging Face's Accelerate library for distributed training with:

   - Basic and advanced implementation options
   - Built-in support for PEFT and external reward models
   - Simplified configuration using YAML files

2. **Torchrun-based Implementation** ([torchrun/](torchrun/)): Uses PyTorch's distributed training with:
   - FSDP (Fully Sharded Data Parallel) support
   - Comprehensive monitoring and experiment tracking
   - Multiple reward function implementations

## Features

### Common Features

- GRPO training implementation
- Support for various model architectures
- Distributed training capabilities
- Integration with Amazon SageMaker
- PEFT (Parameter-Efficient Fine-Tuning) support
- Custom reward functions
- Model checkpointing

### Implementation-specific Features

#### Accelerate Implementation

- Mixed precision training (FP16)
- Simple configuration via YAML
- Hub model pushing capabilities
- Training evaluation during runtime

#### Torchrun Implementation

- 4-bit quantization support
- Flash Attention 2 integration
- MLflow experiment tracking
- GPU metrics logging

## Directory Structure

```
.
├── accelerate/                     # Accelerate-based implementation
│   ├── scripts/
│   │   ├── default_config.yaml    # Accelerate configuration
│   │   ├── grpo.py               # Basic implementation
│   │   ├── grpo_advanced.py      # Advanced implementation
│   │   └── requirements.txt      # Dependencies
│   └── launch-training-job.ipynb  # SageMaker launcher
└── torchrun/                      # Torchrun-based implementation
    ├── scripts/
    │   ├── utils/
    │   │   └── reward_functions.py # Reward implementations
    │   ├── train.py              # Training script
    │   └── requirements.txt      # Dependencies
    └── model-trainer-notebook.ipynb # Example notebook
```

## Getting Started

1. Choose the implementation that best suits your needs:

   - Use `accelerate/` for a simpler setup with built-in distributed training
   - Use `torchrun/` for advanced features and comprehensive monitoring

2. Follow the README in the respective directory for specific setup and usage instructions

3. Use the provided notebooks to launch training jobs on Amazon SageMaker

## Requirements

Each implementation has its own requirements.txt file with specific dependencies. Common requirements include:

- transformers
- trl
- accelerate/torch
- sagemaker
- datasets
- peft

## Usage

Refer to the specific implementation directories for detailed usage instructions:

- [Accelerate Implementation](accelerate/README.md)
- [Torchrun Implementation](torchrun/README.md)

## Contributing

When contributing to this project:

1. Follow the existing code style
2. Add appropriate error handling
3. Update documentation as needed
4. Add tests for new features
