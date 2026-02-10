# Distributed Training on Amazon SageMaker

This directory contains examples of distributed training implementations on Amazon SageMaker. Examples are organized by training technique, model, or domain.

## Repository Structure

```
3_distributed_training/
├── models/                          # LLM fine-tuning examples by model
├── reinforcement-learning/          # RL techniques (GRPO, DPO)
├── diffusers/                       # Image generation fine-tuning
├── spectrum_finetuning/             # Spectrum selective layer freezing
├── unsloth/                         # Unsloth fine-tuning
├── time-series-forecasting/         # Time series models
└── distributed_training_sm_unified_studio/  # SageMaker Unified Studio
```

## Contributing

When adding a new example, place it in the appropriate folder:

- **Model-specific LLM fine-tuning** → `models/<model-name>/`
- **RL technique** → `reinforcement-learning/<algorithm>/<framework>/`
- **Framework/technique callout** → top-level folder (e.g. `unsloth/`, `spectrum_finetuning/`)
- **Domain-specific** (images, time series, etc.) → dedicated folder (e.g. `diffusers/`, `time-series-forecasting/`)

Each example should include:

- A `README.md` with prerequisites, setup instructions, and description
- A notebook (`.ipynb`) or launch script as the main entry point
- Any required scripts, Dockerfiles, or configuration files

## Examples

### 1. LLM Fine-Tuning by Model

Examples demonstrating distributed fine-tuning with different parallelism strategies (DDP, FSDP, DeepSpeed).

| Model                       | Techniques                | Link                                                                      |
| --------------------------- | ------------------------- | ------------------------------------------------------------------------- |
| DeepSeek-R1-Distill-Qwen-7B | veRL + Ray GRPO           | [models/deepseek-r1-distill-qwen-7b](models/deepseek-r1-distill-qwen-7b/) |
| Gemma-3-4B-IT               | DeepSpeed Zero3           | [models/gemma-3-4b-it](models/gemma-3-4b-it/)                             |
| Meta-Llama-3.1-8B           | SFT                       | [models/meta-llama-3.1-8b](models/meta-llama-3.1-8b/)                     |
| Mistral-7B-v0.3             | DDP, FSDP, DeepSpeed      | [models/mistral-7b-v03](models/mistral-7b-v03/)                           |
| OpenAI GPT-OSS              | FSDP, DeepSpeed, HyperPod | [models/openai--gpt-oss](models/openai--gpt-oss/)                         |
| Qwen3-0.6B                  | DDP, FSDP                 | [models/qwen3-0.6b](models/qwen3-0.6b/)                                   |

### 2. Reinforcement Learning

#### GRPO (Generalized Reinforcement Policy Optimization)

| Framework | Description                                | Link                                                                        |
| --------- | ------------------------------------------ | --------------------------------------------------------------------------- |
| TRL       | Accelerate and Torchrun with FSDP          | [reinforcement-learning/grpo/trl](reinforcement-learning/grpo/trl/)         |
| Unsloth   | Efficient training with 4-bit quantization | [reinforcement-learning/grpo/unsloth](reinforcement-learning/grpo/unsloth/) |
| veRL      | Single-node and multi-node (Ray) setups    | [reinforcement-learning/grpo/veRL](reinforcement-learning/grpo/veRL/)       |

#### DPO (Direct Preference Optimization)

- [DPO with TRL](reinforcement-learning/dpo/trl/) - DPO using Hugging Face's TRL library.

### 3. Diffusers

- [FLUX.1-dev DreamBooth LoRA](diffusers/flux.1-dev/) - Fine-tune FLUX.1-dev with DreamBooth LoRA using Hugging Face Diffusers.

### 4. Spectrum Fine-Tuning

- [Spectrum Fine-Tuning](spectrum_finetuning/) - Selective layer freezing based on Signal-to-Noise Ratio analysis. Example with Qwen3-8B.

### 5. Unsloth Fine-Tuning

- [Qwen2.5-7B-Instruct](unsloth/qwen2.5-7b-instruct/) - Instruction fine-tuning with custom Docker container.
- [Gemma3-4B-IT](unsloth/gemma3-4b-it/) - Instruction fine-tuning with uv-based setup.

### 6. Time Series Forecasting

- [Amazon Chronos 2](time-series-forecasting/amazon-chronos2/) - Deploy and fine-tune Amazon Chronos 2 for time series forecasting.

### 7. SageMaker Unified Studio

- [Distributed Training in Unified Studio](distributed_training_sm_unified_studio/) - Distributed training from SageMaker Unified Studio.
