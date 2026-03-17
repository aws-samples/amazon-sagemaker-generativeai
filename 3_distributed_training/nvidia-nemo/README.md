# NVIDIA NeMo on Amazon SageMaker

This directory contains examples for training and fine-tuning models using NVIDIA NeMo frameworks on Amazon SageMaker.

## Overview

[NVIDIA NeMo](https://github.com/NVIDIA/NeMo) is a scalable framework for building, training, and fine-tuning GPU-accelerated generative AI models. These examples demonstrate how to run NeMo workloads on Amazon SageMaker Training Jobs using custom containers based on the official NGC images.

## Examples

### NeMo RL — GRPO Training

| Model        | Algorithm | Dataset            | Instance Type            | Link                 |
| ------------ | --------- | ------------------ | ------------------------ | -------------------- |
| Qwen2.5-1.5B | GRPO      | OpenMathInstruct-2 | ml.p5.48xlarge (8x H100) | [nemo-rl/](nemo-rl/) |

GRPO (Group Relative Policy Optimization) training using [NeMo RL](https://github.com/NVIDIA/NeMo-RL) v0.5.0. Uses Ray for distributed orchestration, vLLM for generation, and DTensor for policy training. Includes:

- Custom Docker container with EFA and AWS OFI NCCL support
- Ray Dashboard with Prometheus metrics
- TensorBoard logging
- SageMaker checkpointing

### NeMo AutoModel — LLM Fine-Tuning

| Model          | Technique           | Instance Type   | Link                               |
| -------------- | ------------------- | --------------- | ---------------------------------- |
| Mistral-7B-v01 | SFT with LoRA/PEFT  | ml.p5.48xlarge  | [nemo-automodel/](nemo-automodel/) |

Fine-tuning using [NVIDIA NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel) for simplified distributed training with automatic parallelism configuration. Uses FSDP2 with DTensor, supports tensor/pipeline/context/sequence parallelism, FP8 mixed precision, and sequence packing.

## Architecture

Both examples share a common architecture for running on SageMaker:

```
NGC Base Image (NeMo RL / NeMo AutoModel)
    + EFA drivers (for multi-node networking)
    + AWS OFI NCCL plugin (for optimized collective communications)
    + SageMaker Training Toolkit
    = Custom SageMaker Training Container
```

## Directory Structure

```
nvidia-nemo/
├── nemo-rl/
│   ├── docker/
│   │   ├── Dockerfile              # Custom container with EFA + NeMo RL
│   │   └── create-image.sh         # Build and push to ECR
│   ├── scripts/
│   │   ├── launcher.py             # Ray cluster setup for SageMaker
│   │   ├── setup.sh                # Runtime environment setup
│   │   ├── train_grpo.py           # GRPO training entry point
│   │   └── requirements.txt        # Client-side dependencies
│   ├── 1-grpo-training.ipynb       # Main notebook (launch training job)
│   └── grpo.yaml                   # GRPO training configuration
└── nemo-automodel/
    ├── scripts/
    │   ├── train.py                # NeMo AutoModel training entry point
    │   └── requirements.txt        # Python dependencies
    └── 1-llm-fine-tuning.ipynb     # Main notebook (launch training job)
```

## Prerequisites

- AWS account with access to GPU instances (p5.48xlarge for NeMo RL)
- Amazon ECR repository for the custom container
- SageMaker execution role with appropriate permissions
- Python 3.10+ with SageMaker SDK installed

## Getting Started

1. Choose the example that fits your use case
2. Follow the README in the respective directory for container setup and configuration
3. Use the provided notebook to launch the training job on SageMaker
