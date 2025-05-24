# GRPO Training with Accelerate on Amazon SageMaker

This directory contains an implementation of Generative Reward Penalized Optimization (GRPO) training using Hugging Face's TRL library with Accelerate for distributed training on Amazon SageMaker.

## Overview

The implementation provides two approaches for GRPO training:

1. **Basic Implementation** ([grpo.py](scripts/grpo.py)): A simple implementation that demonstrates GRPO training with a custom reward function.
2. **Advanced Implementation** ([grpo_advanced.py](scripts/grpo_advanced.py)): A more comprehensive implementation that supports command-line arguments, PEFT configurations, and external reward models.

## Requirements

The project requires the following main dependencies:

- accelerate==1.6.0
- datasets==3.5.0
- transformers==4.51.3
- trl==0.16.1
- sagemaker==2.243.2
- wandb==0.19.9

## Accelerate Configuration

The distributed training is configured using [default_config.yaml](scripts/default_config.yaml) with the following settings:

- Multi-GPU training enabled
- Mixed precision training (FP16)
- 8 processes for distributed training
- Local machine compute environment

## Usage

### Basic Implementation

The basic implementation in `grpo.py` provides a straightforward example using:

- The TRL-lib/tldr dataset
- A simple length-based reward function
- Qwen2-0.5B-Instruct model

### Advanced Implementation

The advanced implementation in `grpo_advanced.py` offers more flexibility with:

- Command-line argument support
- Integration with external reward models
- PEFT (Parameter-Efficient Fine-Tuning) configuration
- Model checkpointing and Hub pushing capabilities
- Evaluation during training

### Training on SageMaker

Use the provided `launch-training-job.ipynb` notebook to:

1. Configure the training environment
2. Set up distributed training parameters
3. Launch training jobs on SageMaker

## Project Structure

```
.
├── scripts/
│   ├── default_config.yaml    # Accelerate configuration
│   ├── grpo.py               # Basic GRPO implementation
│   ├── grpo_advanced.py      # Advanced GRPO implementation
│   └── requirements.txt      # Project dependencies
├── launch-training-job.ipynb  # SageMaker training launcher
└── README.md                 # This file
```
