# SAMA SFT Training Workflow MCP Server

Generate SageMaker training scripts from templates for Supervised Fine-Tuning workflows.

## Tools

| Tool | Description |
|------|-------------|
| `generate_training_script` | Generate a full SageMaker training script with strategy selection |
| `get_instance_recommendation` | Get instance type recommendations based on model size and strategy |

## Installation

```bash
cd MCP_servers/sama-sft-training-workflow-mcp-server
pip install -e .
```

## MCP Configuration

Add to `~/.kiro/settings/mcp.json`:

```json
{
  "mcpServers": {
    "sft_training_workflow": {
      "command": "python",
      "args": ["-m", "sama_sft_training_workflow_mcp_server.server"],
      "cwd": "/path/to/MCP_servers/sama-sft-training-workflow-mcp-server"
    }
  }
}
```

## Example Usage

```
> "Generate a training script for meta-llama/Llama-3.2-3B-Instruct using PEFT"
> "What instance type should I use for a 7B model with full fine-tuning?"
```
