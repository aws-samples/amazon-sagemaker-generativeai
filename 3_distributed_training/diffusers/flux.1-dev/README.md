# Fine-tune FLUX.1-dev with DreamBooth LoRA on Amazon SageMaker

Fine-tune [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) with DreamBooth LoRA using Hugging Face Diffusers on Amazon SageMaker.

## Prerequisites

- AWS account with SageMaker access (p4de or p5 instance quota for SageMaker Training Job)
- Hugging Face account with API token
- Weights & Biases account with API key

## Project Structure

```
flux.1-dev/
├── scripts/
│   ├── train_dreambooth_lora_flux.py  # DreamBooth LoRA training script
│   ├── default_config.yaml            # Accelerate configuration
│   └── script.sh                      # Training launch script
├── flux-fine-tune-sagemaker.ipynb     # SageMaker training notebook
├── .env-example                       # Environment variables template
├── requirements.txt                   # Python dependencies
├── pyproject.toml                     # Project configuration
└── README.md                          # This file
```

## Quick Start

1. Install uv and restart your shell:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create a virtual environment:

```bash
uv venv --prompt flux --python 3.12
source .venv/bin/activate
```

3. Install dependencies:

```bash
uv pip install -r requirements.txt
```

4. Add your Hugging Face and Weights & Biases API keys to `.env-example` and rename it to `.env`

5. Run the `flux-fine-tune-sagemaker.ipynb` notebook to launch the training job on SageMaker.

## Training Configuration

The training uses Accelerate for distributed training with the following key parameters:

- **Model**: black-forest-labs/FLUX.1-dev
- **Method**: DreamBooth LoRA
- **Optimizer**: Prodigy
- **Precision**: bf16
- **Resolution**: 512
- **Reporting**: Weights & Biases
