# Spectrum Fine-Tuning on Amazon SageMaker AI

This example demonstrates how to use [Spectrum](https://github.com/QuixiAI/spectrum) with Amazon SageMaker fully managed training jobs to selectively fine-tune model layers based on Signal-to-Noise Ratio (SNR) analysis.

## Overview

Spectrum fine-tuning analyzes each layer in a model to determine its Signal-to-Noise Ratio (SNR), then selectively freezes or unfreezes layers during training. This approach reduces resource requirements and training time without significant impact to model quality.

- **Model**: Qwen/Qwen3-8B
- **Strategy**: FSDP with Spectrum selective layer freezing
- **Dataset**: [rajpurkar/squad](https://huggingface.co/datasets/rajpurkar/squad)
- **Instance Type**: ml.p4de.24xlarge
- **Framework**: PyTorch with Hugging Face Transformers and TRL

## Prerequisites

1. AWS Account with SageMaker access
2. Hugging Face account and API token

## Project Structure

```
spectrum_finetuning/
├── scripts/
│   ├── train_spectrum.py    # Training script with Spectrum layer handling
│   └── requirements.txt     # Python dependencies
├── helper_functions/
│   ├── __init__.py
│   └── utils.py             # Helper utilities
├── images/                  # Comparison charts
│   ├── Spectrum-CPU-Comparison.png
│   ├── Spectrum-GPU-Comparison.png
│   └── Spectrum-ValidationLoss-Comparison.png
├── spectrum_training.ipynb  # Main notebook
└── README.md                # This file
```

## Quick Start

1. Open `spectrum_training.ipynb` in SageMaker Studio or JupyterLab
2. Install dependencies and restart the kernel
3. Clone the Spectrum repository and run the SNR analysis in a terminal
4. Follow the notebook to upload data, configure training, and launch the SageMaker training job
5. Deploy and test the fine-tuned model
