# SAMA SFT Model Helper MCP Server

Discover, filter, and select models from HuggingFace Hub for Supervised Fine-Tuning workflows.

## Tools

| Tool | Description |
|------|-------------|
| `search_models_by_family` | Search models by HF organization (e.g. `meta-llama`, `Qwen`) |
| `search_models_by_keyword` | Free-text search across HF Hub text-generation models |
| `filter_models_by_size` | Filter a model list by parameter count range |
| `get_model_details` | Get full metadata for a selected model |

## Installation

```bash
cd MCP_servers/sama-sft-model-helper-mcp-server
pip install -e .
```

## MCP Configuration

Add to `~/.kiro/settings/mcp.json`:

```json
{
  "mcpServers": {
    "sft_model_helper": {
      "command": "python",
      "args": ["-m", "sama_sft_model_helper_mcp_server.server"],
      "cwd": "/path/to/MCP_servers/sama-sft-model-helper-mcp-server"
    }
  }
}
```

## Example Usage

```
> "Show me Qwen models for fine-tuning"
> "Filter to models between 1B and 8B parameters"
> "Select Qwen/Qwen2.5-3B-Instruct"
```
