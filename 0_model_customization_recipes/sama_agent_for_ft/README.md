# SAMA: SageMaker AI Model Agents

Natural language → SageMaker training. Use Kiro CLI to search models, prepare datasets, generate recipes, and launch fine-tuning jobs — all through conversation.

## What is SAMA?

SAMA provides 6 MCP (Model Context Protocol) servers that integrate with Kiro CLI. Describe what you want to fine-tune, and SAMA handles model discovery, dataset formatting, recipe generation, and SageMaker job orchestration.

## Architecture

```
User: "Fine-tune Mistral 7B on financial data with PEFT"
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
   Model Search    Dataset Search   Dataset Prep
   sft_model       dataset_prep     dataset_prep
   _helper         search_          prepare_
                   datasets         dataset
        │               │               │
        ▼               ▼               ▼
   Model Select    Dataset Inspect  S3 Upload
   get_model       inspect_         upload_
   _details        dataset          to_s3
        │               │               │
        └───────┬───────┘               │
                ▼                       │
         License Check                  │
         get_model_details              │
                │                       │
        ┌───────┴───────┐              │
        ▼               ▼              │
   SFT Recipe      GRPO Recipe        │
   sft_recipe      grpo_recipe        │
   _generator      _generator         │
        │               │              │
        ▼               ▼              │
   SFT Script      (uses recipe       │
   sft_training     directly)         │
   _workflow            │              │
        │               │              │
        ▼               ▼              ▼
   Launch SFT      Launch GRPO  ◄─────┘
   sagemaker       sagemaker
   _jobs           _jobs
        │               │
        ▼               ▼
   Monitor Job     Monitor Job
   sagemaker       sagemaker
   _jobs           _jobs
```

## MCP Servers (6 servers, 18 tools)

| Server | Tools | Purpose |
|---|---|---|
| `sft_model_helper` | 4 | HuggingFace Hub model search, filter, select, license check |
| `dataset_prep` | 4 | Dataset search, inspect, format (text/reasoning/image/audio), S3 upload |
| `sft_recipe_generator` | 2 | SFT YAML recipes (PEFT/Spectrum/Full) |
| `grpo_recipe_generator` | 2 | GRPO RLVR YAML recipes |
| `sft_training_workflow` | 2 | Training script generation + instance recommendations |
| `sagemaker_jobs` | 4 | Launch SFT/GRPO jobs, monitor status, list jobs |

## Quick Start

### Option A: SageMaker (recommended)

```bash
# Clone and cd into the repo
cd amazon-sagemaker-generativeai/0_model_customization_recipes/sama-science-agents

# Run the install script (removes Q CLI, installs Kiro CLI, installs all deps)
bash install.sh

# Copy sample config into place
cp -r sample.kiro .kiro

# Start
kiro-cli chat --trust-all-tools
```

The install script auto-detects `/opt/conda/bin/python3` on SageMaker.

### Option B: Local (macOS/Linux)

```bash
cd sama-science-agents

# Set your Python path and run install
PYTHON=/path/to/your/python3 bash install.sh

# Copy sample config and update the python path
cp -r sample.kiro .kiro
# Edit .kiro/settings/mcp.json — change /opt/conda/bin/python3 to your python path

# Start
kiro-cli chat --trust-all-tools
```

### Verify

```
/mcp
```

You should see 6 green servers. Then use skills to start a workflow:

```
#supervised-finetuning
#grpo-rlvr
```

## Workflow Modes

Both skills support two modes — the agent asks at the start:

- **Educational mode** — explains each step, concept, and tradeoff
- **Fast mode** — minimal questions, sensible defaults, straight to execution

## Supported Model Families

| Organization | Key Models |
|---|---|
| meta-llama | Llama 3.x |
| Qwen | Qwen3, QwQ, Qwen-VL, Qwen-Audio |
| openai | gpt-oss-20b, gpt-oss-120b |
| google | Gemma |
| microsoft | Phi-4 |
| deepseek-ai | DeepSeek-R1, DeepSeek-V3 |
| mistralai | Mistral 7B, Mixtral |
| CohereForAI | Command R, Aya |
| nvidia | Nemotron |
| tiiuae | Falcon 2 |
| NousResearch | Hermes |
| HuggingFaceTB | SmolLM |
| moonshotai | Kimi K2 |
| ibm-granite | Granite |

## Dataset Modalities

| Modality | Use Case | Example Models |
|---|---|---|
| `text` | Standard instruction-following | Most text models |
| `text_reasoning` | Chain-of-thought with `<think>` blocks | GPT-oss, DeepSeek-R1 |
| `image` | Vision-language tasks | Llama-3.2-Vision, Qwen-VL |
| `audio` | Speech/audio tasks | Qwen2-Audio |

## Example Conversations

```
> I want to fine-tune a Mistral model under 10B parameters

(Agent searches HF Hub, shows options, checks license)

> Let's go with Mistral-7B-Instruct-v0.2, PEFT strategy, Finance-Instruct-500k dataset

(Agent generates recipe, prepares dataset, asks for S3 path, uploads, launches job)

> What's the status of my training job?

(Agent checks SageMaker job status)
```

## Documentation

- [MCP_SERVERS.md](MCP_SERVERS.md) — Server setup and configuration
- [EXTENDING_SAMA.md](EXTENDING_SAMA.md) — Build custom MCP servers and reward functions
- [SAMA_RL.md](SAMA_RL.md) — GRPO RL library documentation

## License

MIT License — see LICENSE file
