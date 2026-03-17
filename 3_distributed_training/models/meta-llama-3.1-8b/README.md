# Fine-tune Meta Llama 3.1 8B on Amazon SageMaker AI

This example demonstrates how to fine-tune [Meta Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) on Amazon SageMaker AI using PyTorch FSDP with QLoRA.

## Overview

- **Model**: meta-llama/Llama-3.1-8B-Instruct
- **Strategy**: FSDP with CPU offloading
- **Dataset**: [UCSC-VLAA/MedReason](https://huggingface.co/datasets/UCSC-VLAA/MedReason)
- **Training**: QLoRA fine-tuning (4-bit quantization + LoRA) with merged weights
- **Framework**: PyTorch with Hugging Face Transformers and TRL

## Prerequisites

1. AWS Account with SageMaker access
2. Hugging Face account and API token (for accessing Llama 3.1)
3. Accept the Llama 3.1 license on Hugging Face

## Project Structure

```
meta-llama-3.1-8b/
├── scripts/
│   ├── train.py            # Training script
│   └── requirements.txt    # Python dependencies
├── sft_llama_31_8b.ipynb   # Main notebook
└── README.md               # This file
```

## Quick Start

1. Open `sft_llama_31_8b.ipynb` in SageMaker Studio or JupyterLab
2. Set your Hugging Face token in the configuration cell
3. Run all cells to:
   - Prepare and upload the dataset to S3
   - Configure training parameters
   - Launch the SageMaker training job
   - Deploy the fine-tuned model with vLLM
   - Test inference
