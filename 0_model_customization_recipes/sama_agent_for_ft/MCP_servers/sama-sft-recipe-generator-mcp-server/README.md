# SAMA SFT Recipe Generator MCP Server

Generate YAML training recipes for PEFT/LoRA, Spectrum, and Full fine-tuning strategies.

## Tools

| Tool | Description |
|------|-------------|
| `generate_sft_recipe` | Generate a YAML recipe for a model with configurable strategy and hyperparameters |
| `list_existing_recipes` | List all existing recipe files organized by model family |

## Supported Strategies

- **PEFT/LoRA** - Parameter-efficient fine-tuning with QLoRA quantization
- **Spectrum** - Selective layer unfreezing (between LoRA and full)
- **Full** - Update all model parameters

## Installation

```bash
cd MCP_servers/sama-sft-recipe-generator-mcp-server
pip install -e .
```

## MCP Configuration

Add to `~/.kiro/settings/mcp.json`:

```json
{
  "mcpServers": {
    "sft_recipe_generator": {
      "command": "python",
      "args": ["-m", "sama_sft_recipe_generator_mcp_server.server"],
      "cwd": "/path/to/MCP_servers/sama-sft-recipe-generator-mcp-server"
    }
  }
}
```

## Example Usage

```
> "Generate a PEFT recipe for meta-llama/Llama-3.2-3B-Instruct"
> "Create a full fine-tuning recipe for Qwen/Qwen2.5-3B-Instruct with learning rate 2e-5"
> "List all existing recipes"
```
