# MCP Servers Guide

Setup and use SAMA MCP servers with Kiro CLI for conversational model training and deployment.

## Installing Kiro CLI

Visit **[https://docs.kiro.ai](https://docs.kiro.ai)** for installation instructions.

Quick install:
```bash
curl -fsSL https://kiro.ai/install.sh | sh
```

## Configuring MCP Servers

### 1. Edit Kiro Config File

Open `~/.kiro/mcp_config.json` and add SAMA servers:

```json
{
  "mcpServers": {
    "llama32_data_prep": {
      "command": "python",
      "args": ["-m", "sama_data_prep_mcp_server"],
      "cwd": "/path/to/SAMA/MCP_servers/sama-llama32-data-prep-mcp-server"
    },
    "llama32_finetuning": {
      "command": "python",
      "args": ["-m", "sama_finetuning_mcp_server"],
      "cwd": "/path/to/SAMA/MCP_servers/sama-llama32-finetuning-mcp-server"
    },
    "llama32_deployment": {
      "command": "python",
      "args": ["-m", "sama_deployment_mcp_server"],
      "cwd": "/path/to/SAMA/MCP_servers/sama-llama32-deployment-mcp-server"
    },
    "grpo_rl_builder": {
      "command": "python",
      "args": ["-m", "model_builder"],
      "cwd": "/path/to/SAMA/MCP_servers/sama_rl_agents"
    },
    "grpo_rl_deployment": {
      "command": "python",
      "args": ["-m", "model_deployment_sync"],
      "cwd": "/path/to/SAMA/MCP_servers/sama_rl_agents"
    }
  }
}
```

**Replace `/path/to/SAMA` with your actual installation path.**

### 2. Install MCP Servers

```bash
cd MCP_servers/sama-llama32-data-prep-mcp-server && pip install -e .
cd ../sama-llama32-finetuning-mcp-server && pip install -e .
cd ../sama-llama32-deployment-mcp-server && pip install -e .
cd ../sama_rl_agents && pip install -e .
```

### 3. Restart Kiro CLI

```bash
kiro chat
```

## Available MCP Servers

### Standard Fine-tuning

**llama32_data_prep** - Prepare datasets
```bash
> "Prepare Dolly dataset for fine-tuning"
```

**llama32_finetuning** - Train models
```bash
> "Fine-tune Llama 3.2 on Dolly with 3 epochs on ml.g5.2xlarge"
```

**llama32_deployment** - Deploy models
```bash
> "Deploy Llama 3.2 on ml.g5.xlarge"
```

### GRPO Reinforcement Learning

**grpo_rl_builder** - Configure GRPO training
```bash
> "I want to train with GRPO"
```

**grpo_rl_deployment** - Deploy GRPO models
```bash
> "Deploy my GRPO model sama-grpo-qwen205binstruct-1234567890"
```

## Recipe-Based GRPO Training

SAMA uses YAML recipes for reproducible GRPO training.

### Available Recipes

```bash
> "Show me available GRPO recipes"
```

Output:
```
- qwen2-0.5b-grpo-config: Qwen/Qwen2-0.5B-Instruct on trl-lib/tldr
- qwen2-0.5b-customer-support-config: Customer support summaries
- qwen2.5-100token-summary: 100-token summaries
```

### Recipe Structure

Recipes are in `sama_rl/recipes/GRPO/`:

```yaml
model:
  name: "Qwen/Qwen2-0.5B-Instruct"

data:
  dataset_name: "trl-lib/tldr"
  train_split: "train[:8000]"

training:
  max_steps: 800
  learning_rate: 5e-5

grpo:
  num_generations: 2
  max_completion_length: 768
  temperature: 0.7

sagemaker:
  instance_type: "ml.g5.2xlarge"
```

### Using Recipes

```bash
> "Train using qwen2-0.5b-grpo-config recipe"
```

The MCP server will:
1. Load the recipe configuration
2. Discover custom reward functions from `user/reward_functions.py`
3. Start SageMaker training job
4. Provide job name for monitoring

## HuggingFace Token Setup

For gated models (Llama 3.2):

1. Get token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Accept Llama license: [https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
3. Save token:

```bash
mkdir -p ~/.huggingface
read -sp 'Enter token: ' TOKEN
echo $TOKEN > ~/.huggingface/token
chmod 600 ~/.huggingface/token
```

MCP servers automatically detect and use the token.

## Custom Reward Functions

Add functions to `user/reward_functions.py`:

```python
def my_custom_reward(completions, **kwargs):
    """Your custom reward logic."""
    tokenizer = kwargs.get('tokenizer')
    rewards = []
    
    for completion in completions:
        # Your scoring logic
        reward = calculate_score(completion)
        rewards.append(reward)
    
    return rewards
```

MCP servers automatically discover and use your functions.

## Troubleshooting

**MCP server not found:**
- Check paths in `~/.kiro/mcp_config.json`
- Verify servers installed: `pip list | grep sama`
- Restart Kiro CLI

**HuggingFace token error:**
- Ensure token at `~/.huggingface/token`
- Check file permissions: `chmod 600 ~/.huggingface/token`
- Verify Llama 3.2 access

**Training job fails:**
- Check CloudWatch logs in AWS Console
- Verify S3 paths are correct
- Check IAM role permissions
