# MCP Servers Directory

This directory contains all Model Context Protocol (MCP) servers for the SAMA agentic fine-tuning system.

## Architecture

```
MCP_servers/
├── sama-sft-model-helper-mcp-server/          # HuggingFace Hub model discovery
├── sama-dataset-prep-mcp-server/              # Dataset search, format, S3 upload
├── sama-sft-recipe-generator-mcp-server/      # SFT recipe YAML generation
├── sama-grpo-recipe-generator-mcp-server/     # GRPO RLVR recipe YAML generation
├── sama-sft-training-workflow-mcp-server/     # SageMaker training script generation
└── sama-sagemaker-job-mcp-server/             # Job launch + monitoring
```

## Server → Tool Map

| Server | Tool | Description |
|---|---|---|
| `sft_model_helper` | `search_models_by_family` | Search HF Hub by org name |
| | `search_models_by_keyword` | Free-text model search |
| | `filter_models_by_size` | Filter by parameter count range |
| | `get_model_details` | Full metadata + license check |
| `dataset_prep` | `search_datasets` | Search HF Hub for datasets |
| | `inspect_dataset` | Preview schema and sample rows |
| | `prepare_dataset` | Format to messages JSONL (4 modalities) |
| | `upload_to_s3` | Upload to user-specified S3 path |
| `sft_recipe_generator` | `generate_sft_recipe` | Generate PEFT/Spectrum/Full YAML |
| | `list_existing_recipes` | List existing SFT recipes |
| `grpo_recipe_generator` | `generate_grpo_recipe` | Generate GRPO RLVR YAML |
| | `list_grpo_recipes` | List existing GRPO recipes |
| `sft_training_workflow` | `generate_training_script` | Generate SageMaker training script |
| | `get_instance_recommendation` | Instance sizing by model + strategy |
| `sagemaker_jobs` | `launch_sft_training_job` | Launch SFT job on SageMaker |
| | `launch_grpo_training_job` | Launch GRPO job on SageMaker |
| | `monitor_training_job` | Check job status |
| | `list_recent_training_jobs` | List recent jobs |

## Workflow Flow

```
search model → select model → license check
                                    │
              search dataset → inspect → prepare → ask S3 path → upload
                                                                   │
                    generate recipe (SFT or GRPO) ←────────────────┘
                              │
                    launch job → monitor job
```

## Configuration

Servers are configured in `.kiro/settings/mcp.json`. See `sample.kiro/settings/mcp.json` for the template.

### SageMaker Setup

```json
"command": "/opt/conda/bin/python3"
```

### Local Setup

```json
"command": "/path/to/your/python3"
```

All servers require `show_banner=False` in `mcp.run()` and `FASTMCP_LOG_LEVEL=ERROR` in env to prevent stdout pollution that breaks the MCP stdio transport.

## Installation

```bash
# From the sama-science-agents directory:
pip install -r requirements.txt
pip install -e MCP_servers/sama-sft-model-helper-mcp-server
pip install -e MCP_servers/sama-sft-recipe-generator-mcp-server
pip install -e MCP_servers/sama-sft-training-workflow-mcp-server
pip install -e MCP_servers/sama-dataset-prep-mcp-server
pip install -e MCP_servers/sama-grpo-recipe-generator-mcp-server
pip install -e MCP_servers/sama-sagemaker-job-mcp-server
```

Or just run `bash install.sh` which does everything.

## Adding a New Server

1. Create directory: `MCP_servers/sama-<name>-mcp-server/`
2. Add `pyproject.toml`, `README.md`, and `<package>/__init__.py` + `server.py`
3. Use `mcp.run(show_banner=False)` in `main()`
4. Add to `.kiro/settings/mcp.json` with `FASTMCP_LOG_LEVEL=ERROR` in env
5. Run `pip install -e MCP_servers/sama-<name>-mcp-server`
