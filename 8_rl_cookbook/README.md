# SAMA CLI - Simple API for Machine Learning Agents

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![AWS SageMaker](https://img.shields.io/badge/AWS-SageMaker-orange.svg)](https://aws.amazon.com/sagemaker/)

SAMA CLI provides a unified interface for both **Reinforcement Learning** (GRPO, PPO) and **Standard Fine-tuning** of language models on AWS SageMaker. It includes intelligent MCP (Model Context Protocol) servers that automatically route requests to the appropriate training pipeline.

## 🚀 Quick Start

### Installation

```bash
git clone <repository-url>
cd SAMA_CLI
pip install -e .
```

### Basic Usage

#### Standard Fine-tuning (Llama 3.2)
```bash
# Prepare data and fine-tune Llama 3.2 on Dolly dataset
q chat "I want to fine-tune a Llama 3.2 model on Dolly dataset using p4d.24xlarge for 20 epochs"
```

#### Reinforcement Learning (GRPO)
```bash
# Train with custom reward functions
q chat "I want to train a Qwen model using GRPO with my custom reward functions on p5.48xlarge"
```

## 📁 Project Structure

```
8_rl_cookbook/
├── sama_rl/                          # Core RL training library
│   ├── grpo.py                       # GRPO trainer implementation
│   ├── deployment.py                 # Model deployment utilities
│   ├── inference.py                  # Inference utilities
│   └── recipes/                      # Training configurations
│       ├── GRPO/                     # GRPO-specific configs
│       ├── PPO/                      # PPO-specific configs
│       └── DPO/                      # DPO-specific configs
├── sama_cli/                         # MCP servers for RL
│   └── sama_rl_agents                # SAMA RL Agents
│       ├── model_builder.py          # Interactive GRPO setup
│       ├── model_deployment_sync.py  # RL model deployment
│       └── model_evaluation.py       # Model evaluation tools
│   └──sama-llama32-data-prep-mcp-server   # Data preparation MCP Servers
│   ├── sama-llama32-finetuning-mcp-server # Fine Tuning pipeline
│   └── sama-llama32-deployment-mcp-server # Model deployment
├── user/                             # User customizations
│   └── reward_functions.py           # Custom reward functions
└── README.md                         # This file
```

## 🎯 Core Capabilities

### 1. Reinforcement Learning (SAMA RL)

#### Supported Algorithms
- **GRPO** (Group Relative Policy Optimization)
- **PPO** (Proximal Policy Optimization) 
- **DPO** (Direct Preference Optimization)
- **RLHF** (Reinforcement Learning from Human Feedback)

#### Key Features
- **Custom Reward Functions**: Define your own reward criteria
- **Interactive Setup**: Conversational configuration through MCP
- **Multi-GPU Support**: P4d, P5 instances for large models
- **Cost Optimization**: Intelligent instance selection
- **Experiment Tracking**: WandB integration

#### Example: GRPO Training with Custom Rewards
```python
from sama_rl import GRPO
from user.reward_functions import length_reward, helpfulness_reward

trainer = GRPO(
    yaml_file="sama_rl/recipes/GRPO/qwen2-0.5b-grpo-config.yaml",
    reward_functions=[length_reward, helpfulness_reward],
    max_steps=500,
    instance_type="ml.p5.48xlarge"
)

# Train the model
trainer.train()

# Deploy to endpoint
endpoint_name = trainer.deploy()
```

### 2. Standard Fine-tuning (Llama 3.2)

#### Supported Models
- **Llama 3.2** (3B, 8B, 70B variants)
- **Qwen 2** (0.5B, 1.5B, 7B variants)
- **Custom HuggingFace models**

#### Key Features
- **Automated Data Preparation**: Dolly, custom datasets
- **SageMaker Integration**: Managed training infrastructure
- **Endpoint Deployment**: Production-ready serving
- **Cost Monitoring**: Instance recommendations and billing estimates

## 🎨 Custom Reward Functions

Create your own reward functions in `user/reward_functions.py`:

```python
def custom_reward(completions, **kwargs):
    """
    Your custom reward logic here.
    
    Args:
        completions (List[str]): Model outputs to evaluate
        **kwargs: Additional context (tokenizer, etc.)
    
    Returns:
        List[float]: Reward scores for each completion
    """
    rewards = []
    for completion in completions:
        # Your reward calculation logic
        reward = calculate_reward(completion)
        rewards.append(reward)
    return rewards

# Use in training
trainer = GRPO(
    yaml_file="sama_rl/recipes/GRPO/qwen2-0.5b-grpo-config.yaml",
    reward_functions=[custom_reward],
    max_steps=100
)
```

### Built-in Reward Functions
- `length_reward` - Target response length
- `helpfulness_reward` - Informative content
- `conciseness_reward` - Balanced brevity
- `safety_reward` - Content safety
- `combined_reward` - Multi-criteria optimization

## 💰 Cost Optimization

### Instance Recommendations
| Instance Type | Cost/Hour | Best For | GPU |
|---------------|-----------|----------|-----|
| ml.g4dn.xlarge | $0.526 | Small models (0.5B-1B) | 1x T4 |
| ml.g5.2xlarge | $1.515 | Medium models (1B-7B) | 1x A10G |
| ml.p4d.24xlarge | $32.77 | Large models (13B+) | 8x A100 |
| ml.p5.48xlarge | $98.32 | Massive models (70B+) | 8x H100 |

## 🔄 Training Job Management

### Naming Conventions
- **Standard Fine-tuning**: `sama-finetune-llama32-{timestamp}`
- **GRPO Training**: `sama-grpo-qwen205binstruct-{timestamp}`
- **Endpoints**: `sama-endpoint-llama32-{timestamp}`

## 🛠️ Configuration

### YAML Configuration Example
```yaml
model:
  name: "Qwen/Qwen2-0.5B-Instruct"
  trust_remote_code: true

data:
  dataset_name: "trl-lib/tldr"
  train_split: "train[:8000]"
  test_split: "test[:200]"

training:
  max_steps: 800
  learning_rate: 5e-5
  per_device_train_batch_size: 4
  fp16: true

grpo:
  num_generations: 2
  max_completion_length: 768
  temperature: 0.7

sagemaker:
  instance_type: "ml.p5.48xlarge"
  max_run: 3600
```

## 📄 License

This project is licensed under the MIT License.

---

**SAMA CLI**: Making reinforcement learning and fine-tuning accessible to everyone. 🚀
