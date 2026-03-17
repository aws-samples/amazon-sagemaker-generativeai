# Fine-tune Qwen3 0.6B on Amazon SageMaker AI

This example demonstrates how to fine-tune [Qwen3 0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) on Amazon SageMaker AI using DDP and FSDP distributed training strategies, with a focus on function calling capabilities.

## Overview

- **Model**: Qwen/Qwen3-0.6B
- **Strategies**: DDP, FSDP with CPU offloading
- **Dataset**: [glaiveai/glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)
- **Training**: QLoRA fine-tuning (4-bit quantization + LoRA) with merged weights
- **Framework**: PyTorch with Hugging Face Transformers and TRL

## Prerequisites

1. AWS Account with SageMaker access
2. Hugging Face account (Qwen3 is publicly available)

## Project Structure

```
qwen3-0.6b/
├── scripts/
│   ├── train.py            # Training script
│   └── requirements.txt    # Python dependencies
├── utils/
│   ├── __init__.py
│   ├── data_format.py      # Data formatting utilities
│   ├── function_extraction_utils.py  # Function call extraction
│   └── preprocessing.py    # Dataset preprocessing (Glaive to OpenAI format)
├── model-trainer-ddp.ipynb  # DDP training notebook
├── model-trainer-fsdp.ipynb # FSDP training notebook
└── README.md                # This file
```

## Quick Start

1. Choose a training strategy:
   - **DDP**: `model-trainer-ddp.ipynb` - Standard Distributed Data Parallel
   - **FSDP**: `model-trainer-fsdp.ipynb` - Fully Sharded Data Parallel with CPU offloading
2. Run all cells to prepare the function-calling dataset, launch training, and deploy the model with tool-use support
