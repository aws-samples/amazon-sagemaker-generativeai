# SageMaker HuggingFace OSS Recipes

A comprehensive collection of training recipes for fine-tuning foundation models on Amazon SageMaker using HuggingFace's open-source libraries. This repository provides production-ready configurations for various model families and training methodologies.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Model Customization on Amazon SageMaker AI](#model-customization-on-amazon-sagemaker-ai)
  - [Supervised Fine-Tuning](#supervised-fine-tuning)
  - [Available Models and Recipes](#available-models-and-recipes)
- [Quick Start](#quick-start)
- [Training Methods](#training-methods)
- [Recipe Structure](#recipe-structure)
- [Advanced Features](#advanced-features)
- [Data Format](#data-format)
- [Performance Optimization](#performance-optimization)
- [Monitoring and Logging](#monitoring-and-logging)
- [Troubleshooting](#troubleshooting)
- [Running Locally on an EC2/Self-Managed Instance](#running-locally-on-an-ec2self-managed-instance)
- [Contributing](#contributing)
- [License](#license)
- [Support](#support)

## Overview

This repository provides a comprehensive framework for model customization on Amazon SageMaker AI, supporting multiple training paradigms from supervised fine-tuning to preference optimization and pre-training. Built on HuggingFace's open-source ecosystem, it offers production-ready configurations for various model families and training methodologies.

## Key Features

### Training Capabilities
- **Multiple Training Methods**: LoRA, Spectrum, and Full fine-tuning
- **Advanced Quantization**: Support for 4-bit quantization (BitsAndBytes, MXFP4)
- **Distributed Training**: Built-in support for multi-GPU and multi-node training
- **Memory Optimization**: Gradient checkpointing, Flash Attention 2, Liger Kernel
- **Flexible Data Loading**: Support for JSONL files and HuggingFace datasets
- **Checkpoint Management**: Automatic checkpoint saving and resumption

### Production Features
- **SageMaker Integration**: Optimized for SageMaker Training Jobs
- **Comprehensive Logging**: TensorBoard integration and detailed metrics
- **Model Deployment**: Automatic model saving for inference deployment
- **Recipe-based Configuration**: YAML-based configuration management

## Model Customization on Amazon SageMaker AI

![SageMaker Recipe Flow](./media/recipe_flow.jpg)

### Supervised Fine-Tuning

Supervised Fine-Tuning (SFT) is the process of adapting pre-trained foundation models to specific tasks or domains using labeled datasets. This approach leverages the rich representations learned during pre-training while specializing the model for downstream applications. Our framework supports three distinct SFT methodologies, each optimized for different resource constraints and performance requirements.

**LoRA (Low-Rank Adaptation)** represents the most resource-efficient approach, introducing trainable low-rank matrices into existing model layers while keeping the original parameters frozen. This method dramatically reduces memory requirements and training time, making it ideal for scenarios with limited computational resources or when quick experimentation is needed. LoRA is particularly effective for instruction-following tasks, domain adaptation, and scenarios where maintaining the base model's general capabilities is crucial. Choose LoRA when you need fast iteration cycles, have memory constraints, or want to create multiple specialized adapters from a single base model.

**Spectrum Training** offers a middle ground between efficiency and performance by selectively unfreezing specific parameter groups based on configurable patterns. This approach provides fine-grained control over which parts of the model to adapt, allowing practitioners to target specific layers or components that are most relevant to their task. Spectrum training is optimal when you have insights into which model components are most important for your specific use case, need better performance than LoRA but can't afford full fine-tuning, or want to experiment with different parameter selection strategies.

**Full Fine-tuning** updates all model parameters, providing maximum adaptation capability at the cost of increased computational requirements. This traditional approach offers the best performance for domain-specific tasks where significant model adaptation is required. Full fine-tuning is recommended when you have sufficient computational resources, need maximum model performance for critical applications, are working with significantly different domains from the pre-training data, or when the task requires substantial changes to the model's behavior.

The choice between these methods depends on your specific constraints: use LoRA for resource-constrained environments and rapid prototyping, Spectrum for balanced performance and efficiency with targeted adaptation, and Full fine-tuning for maximum performance when computational resources are available.

## Available Models and Recipes

| Model | QLoRA | Spectrum | Full | Notebook | Notes |
|-------|-------|----------|------|----------|-------|
| | | | | | |
| **🦙 Meta (Llama) - Text Generation** | | | | | |
| meta-llama/Llama-3.2-3B-Instruct | ✅ [QLoRA](sagemaker_code/hf_recipes/meta-llama/Llama-3.2-3B-Instruct--vanilla-peft-qlora.yaml) | ✅ [Spectrum](sagemaker_code/hf_recipes/meta-llama/Llama-3.2-3B-Instruct--vanilla-spectrum.yaml) | ✅ [Full](sagemaker_code/hf_recipes/meta-llama/Llama-3.2-3B-Instruct--vanilla-full.yaml) | 📓 [Notebook](finetune--meta-llama--Llama-3.2-3B-Instruct.ipynb) | Flash Attention 2, compact model |
| meta-llama/Llama-3.3-70B-Instruct | ✅ [QLoRA](sagemaker_code/hf_recipes/meta-llama/Llama-3.3-70B-Instruct--vanilla-peft-qlora.yaml) | ✅ [Spectrum](sagemaker_code/hf_recipes/meta-llama/Llama-3.3-70B-Instruct--vanilla-spectrum.yaml) | ✅ [Full](sagemaker_code/hf_recipes/meta-llama/Llama-3.3-70B-Instruct--vanilla-full.yaml) | 📓 [Notebook](finetune--meta-llama--Llama-3.3-70B-Instruct.ipynb) | Large scale model, enhanced capabilities |
| | | | | | |
| **🦙 Meta (Llama) - Multi-Modal** | | | | | |
| meta-llama/Llama-3.2-11B-Vision-Instruct | ✅ [QLoRA](sagemaker_code/hf_recipes/meta-llama/Llama-3.2-11B-Vision-Instruct--vanilla-peft-qlora.yaml) | ✅ [Spectrum](sagemaker_code/hf_recipes/meta-llama/Llama-3.2-11B-Vision-Instruct--vanilla-spectrum.yaml) | ✅ [Full](sagemaker_code/hf_recipes/meta-llama/Llama-3.2-11B-Vision-Instruct--vanilla-full.yaml) | 📓 [Notebook](finetune--meta-llama--Llama-3.2-11B-Vision-Instruct.ipynb) | Vision-language model |
| meta-llama/Llama-4-Maverick-17B-128E-Instruct | ✅ [QLoRA](sagemaker_code/hf_recipes/meta-llama/Llama-4-Maverick-17B-128E-Instruct--vanilla-peft-qlora.yaml) | ✅ [Spectrum](sagemaker_code/hf_recipes/meta-llama/Llama-4-Maverick-17B-128E-Instruct--vanilla-spectrum.yaml) | ✅ [Full](sagemaker_code/hf_recipes/meta-llama/Llama-4-Maverick-17B-128E-Instruct--vanilla-full.yaml) | 📓 [Notebook](finetune--meta-llama--Llama-4-Maverick-17B-128E-Instruct.ipynb) | MoE vision-language model, 128 experts |
| | | | | | |
| **🤖 OpenAI - Text Generation** | | | | | |
| openai/gpt-oss-20b | ✅ [QLoRA](sagemaker_code/hf_recipes/openai/gpt-oss-20b--vanilla-peft-qlora.yaml) | ✅ [Spectrum](sagemaker_code/hf_recipes/openai/gpt-oss-20b--vanilla-spectrum.yaml) | ✅ [Full](sagemaker_code/hf_recipes/openai/gpt-oss-20b--vanilla-full.yaml) | 📓 [Notebook](finetune--openai--gpt-oss-20b.ipynb) | 4-bit quantization, optimized for efficiency |
| openai/gpt-oss-120b | ✅ [QLoRA](sagemaker_code/hf_recipes/openai/gpt-oss-120b--vanilla-peft-qlora.yaml) | ✅ [Spectrum](sagemaker_code/hf_recipes/openai/gpt-oss-120b--vanilla-spectrum.yaml) | ✅ [Full](sagemaker_code/hf_recipes/openai/gpt-oss-120b--vanilla-full.yaml) | 📓 [Notebook](finetune--openai--gpt-oss-120b.ipynb) | Large scale model, 4-bit quantization |
| | | | | | |
| **🔮 Qwen (Alibaba) - Text Generation** | | | | | |
| Qwen/Qwen2.5-3B-Instruct | ✅ [QLoRA](sagemaker_code/hf_recipes/Qwen/Qwen2.5-3B-Instruct--vanilla-peft-qlora.yaml) | ✅ [Spectrum](sagemaker_code/hf_recipes/Qwen/Qwen2.5-3B-Instruct--vanilla-spectrum.yaml) | ✅ [Full](sagemaker_code/hf_recipes/Qwen/Qwen2.5-3B-Instruct--vanilla-full.yaml) | - | Compact, efficient model |
| Qwen/QwQ-32B | ✅ [QLoRA](sagemaker_code/hf_recipes/Qwen/QwQ-32B--vanilla-peft-qlora.yaml) | ✅ [Spectrum](sagemaker_code/hf_recipes/Qwen/QwQ-32B--vanilla-spectrum.yaml) | ✅ [Full](sagemaker_code/hf_recipes/Qwen/QwQ-32B--vanilla-full.yaml) | 📓 [Notebook](finetune--Qwen--QwQ-32B.ipynb) | Reasoning-focused model |
| | | | | | |
| **🔮 Qwen (Alibaba) - Multi-Modal** | | | | | |
| Qwen/Qwen2-Audio-7B-Instruct | ✅ [QLoRA](sagemaker_code/hf_recipes/Qwen/Qwen2-Audio-7B-Instruct-vanilla-peft-qlora.yaml) | ⏳ Coming Soon | ✅ [Full](sagemaker_code/hf_recipes/Qwen/Qwen2-Audio-7B-Instruct-vanilla-full.yaml) | 📓 [Notebook](finetune--Qwen--Qwen2-Audio-7B-Instruct.ipynb) | Audio-language model |
| | | | | | |
| **🧠 DeepSeek - Text Generation** | | | | | |
| deepseek-ai/DeepSeek-R1-0528 | ✅ [QLoRA](sagemaker_code/hf_recipes/deepseek-ai/DeepSeek-R1-0528--vanilla-peft-qlora.yaml) | ✅ [Spectrum](sagemaker_code/hf_recipes/deepseek-ai/DeepSeek-R1-0528--vanilla-spectrum.yaml) | ✅ [Full](sagemaker_code/hf_recipes/deepseek-ai/DeepSeek-R1-0528--vanilla-full.yaml) | 📓 [Notebook](finetune--deepseek-ai--DeepSeek-R1-0528.ipynb) | Advanced reasoning model |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | ✅ [QLoRA](sagemaker_code/hf_recipes/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B--vanilla-peft-qlora.yaml) | ✅ [Spectrum](sagemaker_code/hf_recipes/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B--vanilla-spectrum.yaml) | ✅ [Full](sagemaker_code/hf_recipes/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B--vanilla-full.yaml) | 📓 [Notebook](finetune--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B.ipynb) | Compact reasoning model, distilled from R1 |
| | | | | | |
| **🔬 Microsoft - Text Generation** | | | | | |
| microsoft/Phi-3-mini-128k-instruct | ✅ [QLoRA](sagemaker_code/hf_recipes/microsoft/Phi-3-mini-128k-instruct--vanilla-peft-qlora.yaml) | ✅ [Spectrum](sagemaker_code/hf_recipes/microsoft/Phi-3-mini-128k-instruct--vanilla-spectrum.yaml) | ✅ [Full](sagemaker_code/hf_recipes/microsoft/Phi-3-mini-128k-instruct--vanilla-full.yaml) | 📓 [Notebook](finetune--microsoft--Phi-3-mini-128k-instruct.ipynb) | Compact model, 128K context window |
| microsoft/phi-4 | ✅ [QLoRA](sagemaker_code/hf_recipes/microsoft/phi-4--vanilla-peft-qlora.yaml) | ✅ [Spectrum](sagemaker_code/hf_recipes/microsoft/phi-4--vanilla-spectrum.yaml) | ✅ [Full](sagemaker_code/hf_recipes/microsoft/phi-4--vanilla-full.yaml) | 📓 [Notebook](finetune--microsoft--phi-4.ipynb) | Advanced reasoning and coding capabilities |
| | | | | | |
| **🔬 Microsoft - Multi-Modal** | | | | | |
| microsoft/Florence-2-large | ✅ [QLoRA](sagemaker_code/hf_recipes/microsoft/Florence-2-large--vanilla-peft-qlora.yaml) | ✅ [Spectrum](sagemaker_code/hf_recipes/microsoft/Florence-2-large--vanilla-spectrum.yaml) | ✅ [Full](sagemaker_code/hf_recipes/microsoft/Florence-2-large--vanilla-full.yaml) | 📓 [Notebook](finetune--microsoft--Florence-2-large.ipynb) | Vision-language model, OCR and analysis |
| | | | | | |
| **🌟 Google - Text Generation** | | | | | |
| google/gemma-2b | ✅ [QLoRA](sagemaker_code/hf_recipes/google/gemma-2b--vanilla-peft-qlora.yaml) | ✅ [Spectrum](sagemaker_code/hf_recipes/google/gemma-2b--vanilla-spectrum.yaml) | ✅ [Full](sagemaker_code/hf_recipes/google/gemma-2b--vanilla-full.yaml) | 📓 [Notebook](finetune--google--gemma-2b.ipynb) | Efficient small model |
| google/gemma-3-27b-it | ✅ [QLoRA](sagemaker_code/hf_recipes/google/gemma-3-27b-it--vanilla-peft-qlora.yaml) | ✅ [Spectrum](sagemaker_code/hf_recipes/google/gemma-3-27b-it--vanilla-spectrum.yaml) | ✅ [Full](sagemaker_code/hf_recipes/google/gemma-3-27b-it--vanilla-full.yaml) | 📓 [Notebook](finetune--google--gemma-3-27b-it.ipynb) | Latest Gemma model, instruction-tuned |

### Preference Optimization

Preference Optimization represents the next frontier in model alignment, focusing on training models to generate outputs that align with human preferences and values. Unlike supervised fine-tuning which learns from demonstrations, preference optimization learns from comparative feedback, making models more helpful, harmless, and honest.

**Direct Preference Optimization (DPO)** revolutionizes the traditional RLHF pipeline by directly optimizing the model on preference data without requiring a separate reward model. This approach simplifies the training process while maintaining effectiveness, making it more stable and computationally efficient. DPO works by directly optimizing the policy to increase the likelihood of preferred responses while decreasing the likelihood of rejected ones, using a reference model to prevent over-optimization.

**Proximal Policy Optimization (PPO)** represents the traditional reinforcement learning approach to preference optimization, using a reward model trained on human preferences to guide policy updates. PPO maintains a balance between exploration and exploitation while ensuring stable training through clipped policy updates. This method excels in scenarios requiring fine-grained control over the optimization process and complex reward structures.

**Group Relative Policy Optimization (GRPO)** extends preference optimization to handle group-based preferences and multi-objective alignment. This approach is particularly valuable when dealing with diverse user groups or when optimizing for multiple, potentially conflicting objectives simultaneously. GRPO enables more nuanced preference learning that can adapt to different contexts and user populations.

These preference optimization techniques are essential for creating models that not only perform well on benchmarks but also generate outputs that users find genuinely helpful and aligned with their values and expectations.

| Model | DPO | PPO | GRPO | Notes |
|-------|-----|-----|------|-------|
| | | | | |
| **🚧 Coming Soon** | ⏳ | ⏳ | ⏳ | Preference optimization recipes in development |

### Pre-Training

Pre-training represents the foundational phase of large language model development, where models learn rich representations from vast amounts of unlabeled text data. This process creates the base knowledge and capabilities that can later be specialized through fine-tuning and alignment techniques.

**Autoregressive Language Modeling** forms the core of modern pre-training, where models learn to predict the next token in a sequence given the previous context. This seemingly simple objective enables models to develop sophisticated understanding of language, reasoning capabilities, and world knowledge. The scale of pre-training data and compute directly impacts the emergent capabilities of the resulting models.

**Distributed Pre-training** requires careful orchestration of training across multiple GPUs and nodes, involving techniques like data parallelism, model parallelism, and pipeline parallelism. Efficient pre-training also leverages advanced optimizations such as gradient accumulation, mixed precision training, and dynamic loss scaling to maximize throughput while maintaining numerical stability.

**Curriculum Learning and Data Composition** play crucial roles in pre-training effectiveness, involving strategic sequencing of training data and careful balancing of different data sources. Modern pre-training approaches also incorporate techniques like data deduplication, quality filtering, and domain-specific sampling to optimize the learning process.

Pre-training on Amazon SageMaker AI enables practitioners to create custom foundation models tailored to specific domains, languages, or use cases, providing the flexibility to build models that capture domain-specific knowledge and patterns not present in general-purpose models.

| Model | Autoregressive | Distributed | Curriculum | Notes |
|-------|---------------|-------------|------------|-------|
| | | | | |
| **🚧 Coming Soon** | ⏳ | ⏳ | ⏳ | Pre-training recipes in development |

## Quick Start

### 1. Basic Usage

```bash
# Run with a recipe configuration
python sagemaker_code/sft.py --config sagemaker_code/hf_recipes/meta-llama/Llama-3.2-3B-Instruct--vanilla-peft-qlora.yaml

# Override specific parameters
python sagemaker_code/sft.py \
    --config sagemaker_code/hf_recipes/meta-llama/Llama-3.2-3B-Instruct--vanilla-peft-qlora.yaml \
    --num_train_epochs 3 \
    --learning_rate 1e-4
```

### 2. SageMaker Training Job

```python
from sagemaker.pytorch import PyTorch

# Configure SageMaker estimator
estimator = PyTorch(
    entry_point='sft.py',
    source_dir='sagemaker_code',
    role=role,
    instance_type='ml.g5.2xlarge',
    instance_count=1,
    framework_version='2.0.0',
    py_version='py310',
    hyperparameters={
        'config': 'hf_recipes/meta-llama/Llama-3.2-3B-Instruct--vanilla-peft-qlora.yaml'
    }
)

# Start training
estimator.fit({'training': 's3://your-bucket/training-data/'})
```

## Training Methods

### LoRA (Low-Rank Adaptation)
Parameter-efficient fine-tuning that adds trainable low-rank matrices to existing layers.

**Benefits:**
- Significantly reduced memory usage
- Faster training times
- Easy to merge and deploy
- Maintains base model performance

**Configuration:**
```yaml
use_peft: true
load_in_4bit: true
lora_target_modules: "all-linear"
lora_r: 16
lora_alpha: 16
```

### Spectrum Training
Selective parameter unfreezing based on configurable patterns for targeted fine-tuning.

**Benefits:**
- Fine-grained control over parameter updates
- Balanced approach between efficiency and performance
- Customizable parameter selection

**Configuration:**
```yaml
spectrum_config_path: sagemaker_code/configs/spectrum/meta-llama/snr_results_meta-llama-Llama-3.2-3B-Instruct_unfrozenparameters_30percent.yaml
```

**Available Spectrum Configurations:**
- Meta-Llama models: `sagemaker_code/configs/spectrum/meta-llama/`
- OpenAI models: `sagemaker_code/configs/spectrum/openai/`
- Qwen models: `sagemaker_code/configs/spectrum/Qwen/`
- DeepSeek models: `sagemaker_code/configs/spectrum/deepseek-ai/`

### Full Fine-tuning
Traditional approach that updates all model parameters.

**Benefits:**
- Maximum model adaptation capability
- Best performance for domain-specific tasks
- Complete model customization

**Configuration:**
```yaml
use_peft: false
```

## Recipe Structure

Each recipe is a YAML configuration file containing:

```yaml
# Model Configuration
model_name_or_path: meta-llama/Llama-3.2-3B-Instruct
tokenizer_name_or_path: meta-llama/Llama-3.2-3B-Instruct
model_revision: main
torch_dtype: bfloat16
attn_implementation: flash_attention_2
use_liger: false
bf16: true
tf32: true
output_dir: /opt/ml/output/meta-llama/Llama-3.2-3B-Instruct/peft-qlora/

# Dataset Configuration
dataset_id_or_path: /opt/ml/input/data/training/dataset.jsonl
max_seq_length: 4096
packing: true

# Training Configuration
num_train_epochs: 1
per_device_train_batch_size: 8
gradient_accumulation_steps: 2
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: true
learning_rate: 1.0e-4
lr_scheduler_type: cosine
warmup_ratio: 0.1

# Method-specific Configuration (LoRA)
use_peft: true
load_in_4bit: true
lora_target_modules: "all-linear"
lora_modules_to_save: ["lm_head", "embed_tokens"]
lora_r: 16
lora_alpha: 16

# Logging Configuration
logging_strategy: steps
logging_steps: 5
report_to:
- tensorboard
save_strategy: "epoch"
seed: 42
```

## Advanced Features

### Quantization Options

**4-bit BitsAndBytes (Default)**
```yaml
load_in_4bit: true
mxfp4: false
```

**MXFP4 Quantization**
```yaml
load_in_4bit: true
mxfp4: true
```

### Memory Optimizations

**Gradient Checkpointing**
```yaml
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: true
```

**Flash Attention**
```yaml
attn_implementation: flash_attention_2
```

**Liger Kernel**
```yaml
use_liger: true
```

## Data Format

### JSONL Format
```json
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]}
{"messages": [{"role": "user", "content": "How are you?"}, {"role": "assistant", "content": "I'm doing well!"}]}
```

### HuggingFace Dataset
```yaml
dataset_id_or_path: "HuggingFaceH4/ultrachat_200k"
dataset_train_split: "train_sft"
dataset_test_split: "test_sft"
```

### Multimodal Data Format

**Audio Models (Qwen2-Audio)**
```yaml
# Audio dataset configuration
dataset_id_or_path: /opt/ml/input/data/training/audio_dataset.jsonl
modality_type: "audio"
processor_name_or_path: Qwen/Qwen2-Audio-7B-Instruct
```

**Vision Models (Llama-3.2-Vision)**
```yaml
# Vision dataset configuration  
dataset_id_or_path: /opt/ml/input/data/training/vision_dataset.jsonl
modality_type: "vision"
processor_name_or_path: meta-llama/Llama-3.2-11B-Vision-Instruct
```

## Performance Optimization

### Memory Usage Guidelines

| Model Size | Recommended Instance | Training Method | Batch Size | Example Models |
|------------|---------------------|-----------------|------------|----------------|
| 1.5B | ml.g5.xlarge | LoRA + 4-bit | 16 | DeepSeek-R1-Distill-Qwen-1.5B |
| 2B | ml.g5.xlarge | LoRA + 4-bit | 8 | Gemma-2B |
| 3B | ml.g5.2xlarge | LoRA + 4-bit | 8 | Llama-3.2-3B-Instruct, Qwen2.5-3B-Instruct, Phi-3-mini-128k-instruct |
| 7B | ml.g5.4xlarge | LoRA + 4-bit | 4 | Qwen2-Audio-7B-Instruct, Florence-2-large |
| 11B | ml.g5.8xlarge | LoRA + 4-bit | 4 | Llama-3.2-11B-Vision-Instruct |
| 14B | ml.g5.12xlarge | LoRA + 4-bit | 2 | Phi-4 |
| 17B (MoE) | ml.g5.12xlarge | LoRA + 4-bit | 2 | Llama-4-Maverick-17B-128E-Instruct |
| 20B | ml.g5.12xlarge | LoRA + 4-bit | 2 | GPT-OSS-20B |
| 27B | ml.g5.24xlarge | LoRA + 4-bit | 1 | Gemma-3-27B-IT |
| 32B | ml.g5.24xlarge | LoRA + 4-bit | 2 | QwQ-32B |
| 70B | ml.p4d.24xlarge | LoRA + 4-bit | 1 | Llama-3.3-70B-Instruct |
| 120B+ | ml.p4d.24xlarge | LoRA + 4-bit | 1 | GPT-OSS-120B, DeepSeek-R1-0528 |

### Training Speed Tips

1. **Use Flash Attention 2**: Reduces memory and increases speed
2. **Enable Gradient Checkpointing**: Trades compute for memory
3. **Optimize Batch Size**: Balance between memory usage and convergence
4. **Use Mixed Precision**: Enable bf16 for better performance

## Monitoring and Logging

### TensorBoard Integration
```yaml
report_to:
  - tensorboard
logging_steps: 5
```

### Metrics Tracking
- Training loss
- Learning rate schedule
- GPU memory usage
- Training throughput

## Troubleshooting

### Common Issues

**Out of Memory (OOM)**
- Reduce `per_device_train_batch_size`
- Enable `gradient_checkpointing`
- Use 4-bit quantization
- Reduce `max_seq_length`

**Slow Training**
- Increase `gradient_accumulation_steps`
- Enable Flash Attention 2
- Use Liger Kernel optimizations
- Optimize data loading

**Model Quality Issues**
- Adjust learning rate (try 5e-5 to 2e-4 range)
- Increase training epochs or adjust warmup ratio
- Check data quality and format (ensure proper JSONL structure)
- Experiment with different LoRA ranks (8, 16, 32, 64)
- For multimodal models, verify processor configuration matches model

**Configuration Issues**
- Ensure recipe paths are correct: `sagemaker_code/hf_recipes/{org}/{model}--vanilla-{method}.yaml`
- Check accelerate config paths: `sagemaker_code/configs/accelerate/{config}.yaml`
- Verify spectrum config paths: `sagemaker_code/configs/spectrum/{org}/{config}.yaml`
- Validate output directory permissions and paths

## Example Notebooks

The repository includes comprehensive Jupyter notebooks demonstrating end-to-end fine-tuning workflows:

### Meta (Llama) Models
- **[Meta Llama 3.2 3B Instruct](finetune--meta-llama--Llama-3.2-3B-Instruct.ipynb)**: Compact text generation model
- **[Meta Llama 3.2 11B Vision Instruct](finetune--meta-llama--Llama-3.2-11B-Vision-Instruct.ipynb)**: Vision-language model
- **[Meta Llama 3.3 70B Instruct](finetune--meta-llama--Llama-3.3-70B-Instruct.ipynb)**: Large-scale text generation
- **[Meta Llama 4 Maverick 17B 128E Instruct](finetune--meta-llama--Llama-4-Maverick-17B-128E-Instruct.ipynb)**: MoE vision-language model

### OpenAI Models
- **[OpenAI GPT-OSS 20B](finetune--openai--gpt-oss-20b.ipynb)**: Mid-scale efficient model
- **[OpenAI GPT-OSS 120B](finetune--openai--gpt-oss-120b.ipynb)**: Large-scale model

### Qwen (Alibaba) Models
- **[Qwen QwQ 32B](finetune--Qwen--QwQ-32B.ipynb)**: Reasoning-focused model
- **[Qwen2 Audio 7B Instruct](finetune--Qwen--Qwen2-Audio-7B-Instruct.ipynb)**: Audio-language model

### DeepSeek Models
- **[DeepSeek R1 0528](finetune--deepseek-ai--DeepSeek-R1-0528.ipynb)**: Advanced reasoning model
- **[DeepSeek R1 Distill 1.5B](finetune--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B.ipynb)**: Compact reasoning model

### Microsoft Models
- **[Microsoft Phi-3 Mini 128K Instruct](finetune--microsoft--Phi-3-mini-128k-instruct.ipynb)**: Compact model with extended context
- **[Microsoft Phi-4](finetune--microsoft--phi-4.ipynb)**: Advanced reasoning and coding model
- **[Microsoft Florence-2 Large](finetune--microsoft--Florence-2-large.ipynb)**: Vision-language model for OCR and analysis

### Google Models
- **[Google Gemma 2B](finetune--google--gemma-2b.ipynb)**: Efficient small model
- **[Google Gemma 3 27B IT](finetune--google--gemma-3-27b-it.ipynb)**: Latest instruction-tuned Gemma model

Each notebook provides:
- SageMaker setup and configuration
- Dataset preparation and formatting
- Training job execution
- Model evaluation and deployment
- Best practices and optimization tips

## Contributing

We welcome contributions! Please:

1. Add new model recipes following the naming convention: `{org}--{model}--vanilla-{method}.yaml`
2. Test configurations thoroughly on appropriate instance types
3. Update documentation and model support tables
4. Follow the existing YAML structure and include proper logging configuration
5. Add corresponding Jupyter notebooks for new model families


## Running Locally on an EC2/Self-Managed Instance

Start by updating the local instance (assuming a fresh VM),

```bash

sudo apt-get update -y && sudo apt-get install python3-pip python3-venv -y

```

Use uv (recommended to create a virtual env and install packages).

```bash
# install uv package and env manager
sudo pip install uv

# create a py3.XX environment
uv venv py311 --python 3.11

# activate venv
source py311/bin/activate
```

Clone git repo and navigate to the supervised fine-tuning repository

```bash
# clone repository
git clone https://github.com/aws-samples/amazon-sagemaker-generativeai.git

# navigate supervised fine-tuning repository
cd amazon-sagemaker-generativeai/3_distributed_training/sm_huggingface_oss_recipes/supervised_finetuning/
```

Run distributed training using Accelerate orchestrator

```bash
# Set model output directory
SM_MODEL_DIR="/home/ubuntu/amazon-sagemaker-generativeai/3_distributed_training/sm_huggingface_oss_recipes/supervised_finetuning/models"

# Run training with accelerate
accelerate launch \
    --config_file sagemaker_code/configs/accelerate/ds_zero3.yaml \
    --num_processes 1 \
    sagemaker_code/sft.py \
    --config sagemaker_code/hf_recipes/meta-llama/Llama-3.2-3B-Instruct--vanilla-peft-qlora.yaml
```

### Available Accelerate Configurations

- **DeepSpeed ZeRO Stage 1**: `sagemaker_code/configs/accelerate/ds_zero1.yaml`
- **DeepSpeed ZeRO Stage 3**: `sagemaker_code/configs/accelerate/ds_zero3.yaml`
- **FSDP**: `sagemaker_code/configs/accelerate/fsdp.yaml`
- **FSDP with QLoRA**: `sagemaker_code/configs/accelerate/fsdp_qlora.yaml`

## License

This project is licensed under the MIT License - see the [LICENSE](../../LICENSE) file for details.

## Support

For issues and questions:
- Check the troubleshooting section
- Review SageMaker documentation
- Open an issue in this repository

---

**Note**: This framework is optimized for Amazon SageMaker but can be adapted for other distributed training environments.