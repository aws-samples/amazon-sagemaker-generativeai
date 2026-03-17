# Fine-tune OpenAI GPT-OSS on Amazon SageMaker AI

This example demonstrates how to fine-tune [OpenAI GPT-OSS](https://huggingface.co/openai/gpt-oss-20b) open-weight models on Amazon SageMaker AI using FSDP, DeepSpeed, and HyperPod Recipes.

## Overview

- **Models**: openai/gpt-oss-20b (21B params, MoE) and openai/gpt-oss-120b (117B params, MoE)
- **Strategies**: FSDP with QLoRA, DeepSpeed, HyperPod Recipes (TrainingJob and EKS)
- **Dataset**: [HuggingFaceH4/Multilingual-Thinking](https://huggingface.co/datasets/HuggingFaceH4/Multilingual-Thinking)
- **Training**: LoRA fine-tuning with Flash Attention v3 (Hopper GPUs) or eager attention
- **Framework**: PyTorch with Hugging Face Transformers and TRL

## Prerequisites

1. AWS Account with SageMaker access (p4de.24xlarge for 20B, p5en.48xlarge for 120B)
2. Hugging Face account and API token

## Project Structure

```
openai--gpt-oss/
├── code/
│   ├── gpt_oss_sft.py                        # SFT training script
│   ├── sm_accelerate_train.sh                 # Accelerate launcher script
│   ├── requirements.txt                       # Training dependencies
│   └── deploy_gpt_oss_transformers/           # Deployment artifacts
│       ├── model.py                           # Inference handler
│       ├── serving.properties                 # DJL serving config
│       └── requirements.txt                   # Inference dependencies
├── finetune_gpt_oss_fsdp.ipynb               # FSDP training notebook
├── finetune_gpt_oss_deepspeed_zero3.ipynb    # DeepSpeed ZeRO-3 notebook
├── finetune_gpt_oss_hyperpod_recipes_tj.ipynb # HyperPod Recipes (TrainingJob)
├── finetune_gpt_oss_hyperpod_recipes_eks.ipynb # HyperPod Recipes (EKS)
└── README.md                                  # This file
```

## Quick Start

1. Choose a training approach:
   - **FSDP**: `finetune_gpt_oss_fsdp.ipynb` - FSDP with QLoRA (PyTorch or HuggingFace container)
   - **DeepSpeed**: `finetune_gpt_oss_deepspeed_zero3.ipynb` - DeepSpeed ZeRO-3
   - **HyperPod Recipes (TrainingJob)**: `finetune_gpt_oss_hyperpod_recipes_tj.ipynb`
   - **HyperPod Recipes (EKS)**: `finetune_gpt_oss_hyperpod_recipes_eks.ipynb`
2. Set your Hugging Face token in the configuration cell
3. Run all cells to prepare data, launch training, and deploy the model with vLLM
