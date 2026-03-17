# Supervised Fine-Tuning with NVIDIA NeMo AutoModel on Amazon SageMaker

Fine-tune LLMs using [NVIDIA NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel) on Amazon SageMaker Training Jobs.

## Overview

NeMo AutoModel is a PyTorch DTensor-native SPMD training library that provides:

- **FSDP2** with DTensor (more efficient than FSDP1)
- **Tensor, Pipeline, Context, and Sequence Parallelism** via config
- **FP8 mixed precision** support
- **Sequence packing** for better GPU utilization
- **LoRA/PEFT** out of the box
- YAML-driven configuration with CLI overrides

| Component       | Details                          |
| --------------- | -------------------------------- |
| Base model      | mistralai/Mistral-7B-v0.1       |
| Dataset         | rajpurkar/squad (HuggingFace)    |
| PEFT            | LoRA (rank 8, alpha 32)          |
| Strategy        | FSDP2 with DTensor               |
| Instance type   | ml.p5.48xlarge (8x H100 80GB)   |

## Prerequisites

- AWS account with access to `ml.p5.48xlarge` instances
- Python 3.10+ with SageMaker SDK installed

## Project Structure

```
nemo-automodel/
├── scripts/
│   ├── train.py            # NeMo AutoModel training entry point
│   └── requirements.txt    # Python dependencies (installs nemo-automodel from git)
├── 1-llm-fine-tuning.ipynb # Main notebook
└── README.md               # This file
```

## Quick Start

1. Open `1-llm-fine-tuning.ipynb` in SageMaker Studio or JupyterLab
2. Run the notebook to:
   - Write the training configuration YAML (model, dataset, optimizer, LoRA settings)
   - Upload the config to S3
   - Launch a SageMaker training job using the PyTorch DLC with NeMo AutoModel installed via requirements.txt
3. Monitor training in CloudWatch Logs

## Training Configuration

The notebook generates an `args.yaml` config file with key settings:

- **Model**: Loaded via `NeMoAutoModelForCausalLM.from_pretrained`
- **Distributed**: FSDP2 strategy with configurable TP/CP/SP
- **Optimizer**: Adam with cosine LR scheduler
- **Checkpointing**: Saved to `/opt/ml/checkpoints/` in safetensors format
- **Dataset**: Loaded directly from HuggingFace Hub (no S3 data channels needed)
