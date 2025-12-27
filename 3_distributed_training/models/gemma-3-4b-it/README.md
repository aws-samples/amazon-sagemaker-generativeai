# Fine-tune Google Gemma-3 4B with DeepSpeed ZeRO-3 on Amazon SageMaker

This example demonstrates how to fine-tune [Google Gemma-3 4B Instruct](https://huggingface.co/google/gemma-3-4b-it) on Amazon SageMaker AI using DeepSpeed ZeRO-3 distributed training strategy.

## Overview

- **Model**: google/gemma-3-4b-it (4 billion parameters)
- **Strategy**: DeepSpeed ZeRO-3 with CPU offloading
- **Dataset**: [HuggingFaceH4/Multilingual-Thinking](https://huggingface.co/datasets/HuggingFaceH4/Multilingual-Thinking) (Apache 2.0 license)
- **Training**: LoRA fine-tuning with merged weights
- **Framework**: PyTorch with Hugging Face Transformers

## Prerequisites

1. AWS Account with SageMaker access
2. Hugging Face account and API token (for accessing Gemma-3 model)
3. Accept the Gemma-3 license on Hugging Face

## Files

```
gemma-3-4b-it/
├── model-trainer-deepspeed-zero3.ipynb  # Main notebook
├── scripts/
│   ├── train.py                         # Training script
│   ├── sm_accelerate_train.sh           # Accelerate launcher script
│   └── requirements.txt                 # Python dependencies
└── README.md                            # This file
```

## Quick Start

1. Open `model-trainer-deepspeed-zero3.ipynb` in SageMaker Studio or JupyterLab
2. Set your Hugging Face token in the configuration cell
3. Run all cells to:
   - Prepare and upload the dataset to S3
   - Configure training parameters
   - Launch the SageMaker training job
   - Deploy the fine-tuned model
   - Test inference

## Training Configuration

The training uses the following key parameters:

| Parameter | Value |
|-----------|-------|
| Instance Type | ml.g5.12xlarge (4x A10G GPUs) |
| Distributed Strategy | DeepSpeed ZeRO-3 |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| Learning Rate | 2e-5 |
| Batch Size | 1 per device |
| Gradient Accumulation | 16 steps |
| Precision | bfloat16 |

## Deployment

The fine-tuned model is deployed using:
- DJL LMI container with vLLM backend
- Instance: ml.g5.2xlarge (single A10G GPU)
- Tensor parallelism for efficient inference

## License

- Model: [Gemma License](https://ai.google.dev/gemma/terms)
- Dataset: Apache 2.0
- Code: MIT-0
