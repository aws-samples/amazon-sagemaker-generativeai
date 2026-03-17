# Fine-tune Mistral 7B v0.3 on Amazon SageMaker AI

This example demonstrates how to fine-tune [Mistral 7B Instruct v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) on Amazon SageMaker AI using multiple distributed training strategies: DDP, FSDP, and DeepSpeed ZeRO-3.

## Overview

- **Model**: mistralai/Mistral-7B-Instruct-v0.3
- **Strategies**: DDP, FSDP with CPU offloading, DeepSpeed ZeRO-3
- **Dataset**: [HuggingFaceH4/Multilingual-Thinking](https://huggingface.co/datasets/HuggingFaceH4/Multilingual-Thinking)
- **Training**: QLoRA fine-tuning (4-bit quantization + LoRA) with merged weights
- **Framework**: PyTorch with Hugging Face Transformers and TRL

## Prerequisites

1. AWS Account with SageMaker access
2. Hugging Face account and API token (for accessing Mistral model)

## Project Structure

```
mistral-7b-v03/
├── scripts/
│   ├── train.py                        # Training script
│   ├── sm_accelerate_train.sh          # Accelerate launcher script
│   └── requirements.txt                # Python dependencies
├── model-trainer-ddp.ipynb             # DDP training notebook
├── model-trainer-fsdp.ipynb            # FSDP training notebook
├── model-trainer-deepspeed-zero3.ipynb # DeepSpeed ZeRO-3 training notebook
└── README.md                           # This file
```

## Quick Start

1. Choose a training strategy:
   - **DDP**: `model-trainer-ddp.ipynb` - Standard Distributed Data Parallel
   - **FSDP**: `model-trainer-fsdp.ipynb` - Fully Sharded Data Parallel with CPU offloading
   - **DeepSpeed ZeRO-3**: `model-trainer-deepspeed-zero3.ipynb` - DeepSpeed ZeRO Stage 3
2. Set your Hugging Face token in the configuration cell
3. Run all cells to prepare data, launch training, and deploy the model
