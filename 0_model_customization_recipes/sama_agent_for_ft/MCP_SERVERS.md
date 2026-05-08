# MCP Servers Setup Guide

Setup SAMA MCP servers with Kiro CLI for agentic model fine-tuning.

## Prerequisites

- Python 3.10+
- AWS credentials configured (for SageMaker job launch/monitor)
- HuggingFace token (for gated models like Llama)

## Setup on SageMaker

```bash
cd sama-science-agents

# One-command setup (removes Q CLI, installs Kiro CLI, installs all deps)
bash install.sh

# Copy sample config
cp -r sample.kiro .kiro

# Verify
kiro-cli chat --trust-all-tools
# Then type: /mcp
```

The install script auto-detects `/opt/conda/bin/python3` on SageMaker and configures everything.

## Setup on Local Machine (macOS/Linux)

```bash
cd sama-science-agents

# Install Kiro CLI
curl -fsSL https://cli.kiro.dev/install | bash
export PATH="$HOME/.local/bin:$PATH"

# Create a Python venv (or use an existing one)
python3 -m venv ~/my-venv
source ~/my-venv/bin/activate

# Install deps + servers
PYTHON=$(which python3) bash install.sh

# Copy and configure
cp -r sample.kiro .kiro
# Edit .kiro/settings/mcp.json — update "command" to your python path
# e.g. "/Users/you/my-venv/bin/python3"

# Verify
kiro-cli chat --trust-all-tools
# Then type: /mcp
```

## MCP Configuration

The config lives at `.kiro/settings/mcp.json`. A sample is provided at `sample.kiro/settings/mcp.json`.

```json
{
  "mcpServers": {
    "sft_model_helper": {
      "command": "/opt/conda/bin/python3",
      "args": ["-m", "sama_sft_model_helper_mcp_server.server"],
      "env": { "FASTMCP_LOG_LEVEL": "ERROR" },
      "disabled": false,
      "autoApprove": ["search_models_by_family", "search_models_by_keyword", "filter_models_by_size", "get_model_details"]
    },
    "dataset_prep": {
      "command": "/opt/conda/bin/python3",
      "args": ["-m", "sama_dataset_prep_mcp_server.server"],
      "env": { "FASTMCP_LOG_LEVEL": "ERROR" },
      "disabled": false,
      "autoApprove": ["search_datasets", "inspect_dataset", "prepare_dataset", "upload_to_s3"]
    },
    "sft_recipe_generator": {
      "command": "/opt/conda/bin/python3",
      "args": ["-m", "sama_sft_recipe_generator_mcp_server.server"],
      "env": { "FASTMCP_LOG_LEVEL": "ERROR" },
      "disabled": false,
      "autoApprove": ["generate_sft_recipe", "list_existing_recipes"]
    },
    "grpo_recipe_generator": {
      "command": "/opt/conda/bin/python3",
      "args": ["-m", "sama_grpo_recipe_generator_mcp_server.server"],
      "env": { "FASTMCP_LOG_LEVEL": "ERROR" },
      "disabled": false,
      "autoApprove": ["generate_grpo_recipe", "list_grpo_recipes"]
    },
    "sft_training_workflow": {
      "command": "/opt/conda/bin/python3",
      "args": ["-m", "sama_sft_training_workflow_mcp_server.server"],
      "env": { "FASTMCP_LOG_LEVEL": "ERROR" },
      "disabled": false,
      "autoApprove": ["generate_training_script", "get_instance_recommendation"]
    },
    "sagemaker_jobs": {
      "command": "/opt/conda/bin/python3",
      "args": ["-m", "sama_sagemaker_job_mcp_server.server"],
      "env": { "FASTMCP_LOG_LEVEL": "ERROR" },
      "disabled": false,
      "autoApprove": ["launch_sft_training_job", "launch_grpo_training_job", "monitor_training_job", "list_recent_training_jobs"]
    }
  }
}
```

Replace `/opt/conda/bin/python3` with your Python path on local machines.

## Available Servers

### Model Discovery
`sft_model_helper` — Search HuggingFace Hub, filter by family/size, get model details + license
```
> "Show me Qwen models under 5B parameters"
> "What's the license for meta-llama/Llama-3.2-3B-Instruct?"
```

### Dataset Preparation
`dataset_prep` — Search datasets, inspect schema, format to messages JSONL, upload to S3
```
> "Find financial QA datasets on HuggingFace"
> "Prepare Josephgflowers/Finance-Instruct-500k as text format"
> "Upload to s3://my-bucket/datasets/finance"
```

Supported modalities: `text`, `text_reasoning` (thinking blocks), `image`, `audio`

### SFT Recipe Generation
`sft_recipe_generator` — Generate PEFT/Spectrum/Full recipe YAMLs
```
> "Generate a PEFT recipe for Mistral-7B with flash_attention_2"
```

Writes to: `../supervised_finetuning/sagemaker_code/hf_recipes/<org>/`

### GRPO Recipe Generation
`grpo_recipe_generator` — Generate GRPO RLVR recipe YAMLs
```
> "Generate a GRPO recipe for Qwen3-4B"
```

Writes to: `../preference_optimization/grpo_rlvr/sagemaker_code/hf_recipes/<org>/`

### Training Script Generation
`sft_training_workflow` — Generate SageMaker training scripts with instance recommendations
```
> "Generate a training script for Mistral-7B with PEFT"
```

Writes to: `../supervised_finetuning/finetune--<org>--<model>.py`

### Job Launch & Monitor
`sagemaker_jobs` — Launch SFT/GRPO jobs, monitor status, list recent jobs
```
> "Launch the training job on ml.g6e.2xlarge"
> "What's the status of my training job?"
> "List my recent training jobs"
```

## Skills (Guided Workflows)

Type these in Kiro CLI chat to activate a prescribed workflow:

- `#supervised-finetuning` — Full SFT pipeline: model → dataset → recipe → script → launch
- `#grpo-rlvr` — Full GRPO pipeline: model → dataset → recipe → launch

Both support educational mode (explains everything) and fast mode (minimal questions).

## HuggingFace Token Setup

For gated models (Llama, etc.):

```bash
mkdir -p ~/.huggingface
read -sp 'Enter token: ' TOKEN
echo $TOKEN > ~/.huggingface/token
chmod 600 ~/.huggingface/token
```

## Troubleshooting

**"connection closed: initialize response"**
- Server is printing to stdout. Ensure `mcp.run(show_banner=False)` and `FASTMCP_LOG_LEVEL=ERROR` in env.
- Test: `python3 -m sama_sft_model_helper_mcp_server.server` — should hang silently.

**"Transport closed" during job launch**
- SageMaker SDK prints to stdout. The `sagemaker_jobs` server wraps all SDK calls in `sys.stdout = io.StringIO()`.
- If still failing, check: `python3 -c "import sagemaker; print('OK')"` — any output before "OK" is the problem.

**ModuleNotFoundError**
- Run `pip install -e MCP_servers/sama-<name>-mcp-server` for the failing server.
- Or run `bash install.sh` to install everything.

**Wrong Python path**
- Check `.kiro/settings/mcp.json` — the `command` field must point to the Python that has the packages installed.
- SageMaker: `/opt/conda/bin/python3`
- Local: your venv python path
