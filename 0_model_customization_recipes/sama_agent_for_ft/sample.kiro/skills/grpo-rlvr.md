---
inclusion: manual
---

# SAMA GRPO RLVR (Preference Optimization) Workflow

You are orchestrating a GRPO (Group Relative Policy Optimization) with RLVR (Reinforcement Learning with Verifiable Rewards) workflow. Follow the prescribed steps IN ORDER.

## Model Discovery

Search is OPEN — any text-generation model on HuggingFace Hub is fair game. Use `search_models_by_family`, `search_models_by_keyword`, or `filter_models_by_size` to help the user find what they need. Do NOT restrict to a hardcoded list.

The user can enter the flow at any point:
- **By name**: "I want to train Qwen/Qwen3-4B with GRPO" → skip to license check
- **By family**: "Show me small Qwen models" → `search_models_by_family`
- **By size**: "I need a model under 2B for tool calling" → search + filter

## MCP Servers Used

| Server | Tools |
|---|---|
| `sft_model_helper` | search_models_by_family, search_models_by_keyword, filter_models_by_size, get_model_details |
| `dataset_prep` | upload_to_s3 |
| `grpo_recipe_generator` | generate_grpo_recipe, list_grpo_recipes |
| `sagemaker_jobs` | launch_grpo_training_job, monitor_training_job, list_recent_training_jobs |

## Workflow Modes

At the START, ask: "Would you like **educational mode** (I explain each step) or **fast mode** (minimal questions, sensible defaults)?"

### Educational Mode
- Explain what GRPO is: trains models using reward signals rather than supervised labels
- Explain RLVR: uses verifiable rewards (tool call accuracy, format compliance) instead of human preference
- Explain each recipe parameter and what it controls
- Explain instance sizing for GRPO (needs more memory than SFT due to multi-generation)

### Fast Mode
- No explanations, use defaults, only ask required questions
- Default dataset: grpo_financial_train. Default domain: financial.

## GRPO Instance Recommendations

GRPO needs more memory than SFT due to multi-generation:
- ≤1B params: ml.g5.2xlarge or ml.g6e.2xlarge
- 1B-4B params: ml.g6e.12xlarge
- 4B-8B params: ml.g6e.12xlarge or ml.p4de.24xlarge
- 8B+ params: ml.p4de.24xlarge or ml.p5.48xlarge

## Prescribed Flow

### Step 1: Model Selection

Help the user find and select a model. Once selected, call `get_model_details`. Store `model_id` and `parameters`.

**License Check (MANDATORY)**: Display the `license` field to the user. If restrictive, warn them. Confirm acceptance before proceeding.

### Step 2: Dataset

Ask the user: **"Do you have your own GRPO dataset already prepared, or would you like to use a sample dataset?"**

#### Path A: Bring Your Own Data (BYOD)

1. Ask the user for the S3 URI or local path to their dataset
2. If local, read a few lines and validate the format:
   - Must be JSONL
   - Each row must have `prompt` (list of message dicts with `role` and `content`) and `answer` (string)
   - The `prompt` should contain system and user messages
3. If valid, confirm: "Dataset looks good — GRPO format verified."
4. If invalid, explain what's wrong and how to fix it
5. Ask for the S3 URI to upload to (if local), or confirm the existing S3 URI

#### Path B: Sample Datasets

Offer ONLY these pre-approved sample datasets for GRPO. Do NOT search HuggingFace for others.

| Domain | Dataset | Reference |
|---|---|---|
| Financial tool calling | `pranavvmurthy26/synthetic-financial-tool-calling-grpo-rlvr-1k` | #[[file:preference_optimization/grpo_rlvr/finetune-tool-call--Qwen--Qwen3-4B-financial.ipynb]] |

The sample dataset has `prompt` (system + user messages) and `answer` fields. The reference notebook shows the full data prep flow including downloading tools and reward function scripts from the dataset repo.

When the user picks the sample dataset:
1. Follow the reference notebook pattern: load dataset, remove `ground_truth` column from train split
2. Save as JSONL
3. Download the tools script and reward function from the dataset repo
4. Ask the user: "Where should I upload the dataset? Provide a full S3 URI"
5. Use `upload_to_s3` with the user-provided S3 URI

### Step 3: GRPO Recipe Generation

Ask for:
1. **Domain**: financial, healthcare, or custom (affects tools/rewards scripts)
2. **Custom hyperparameters** or defaults

Call `generate_grpo_recipe` with the model_id and parameters. Writes to:
`../preference_optimization/grpo_rlvr/sagemaker_code/hf_recipes/<org>/<model>--grpo.yaml`

**Reference recipes** (use these as parameter baselines):
- #[[file:preference_optimization/grpo_rlvr/sagemaker_code/hf_recipes/Qwen/Qwen3-0.6B--grpo.yaml]]
- #[[file:preference_optimization/grpo_rlvr/sagemaker_code/hf_recipes/Qwen/Qwen3-1.7B--grpo.yaml]]
- #[[file:preference_optimization/grpo_rlvr/sagemaker_code/hf_recipes/Qwen/Qwen3-4B--grpo.yaml]]

Key differences from SFT:
- No PEFT/LoRA/Spectrum — GRPO trains the full model
- Has `num_generations`, `max_grpo_completion_length`, `mask_truncated_completions`
- Uses `save_strategy: steps` with `save_steps` instead of `save_strategy: epoch`
- Higher `num_train_epochs` (15-25 typical)
- Lower `learning_rate` (5.0e-6 typical)
- `lr_scheduler_type: constant` (no cosine decay)

### Step 4: Launch Training Job (Optional)

If the user wants to launch directly:
1. Call `launch_grpo_training_job` with:
   - recipe_path, training_data_s3_uri, instance_type
   - tools_script (default: `tools_funcs/financial_tools_complex.py`)
   - reward_fn (default: `rewards/financial_tools_reward.py`)
   - zero_stage (default: 2)
2. Show the job name and how to monitor

**Reference training notebook** (use as template):
- #[[file:preference_optimization/grpo_rlvr/finetune-tool-call--Qwen--Qwen3-4B-financial.ipynb]]

### Step 5: Summary

Summarize: recipe YAML path, instance type, tools/rewards scripts, S3 data URI, job name (if launched).

## Rules

- NEVER skip the license check
- NEVER generate a recipe without confirming model selection
- ALWAYS use GRPO-specific parameters (NOT SFT parameters) for recipes
- ALWAYS ask for S3 upload path — never use a default
- GRPO recipes go to `../preference_optimization/grpo_rlvr/sagemaker_code/hf_recipes/`, NOT the SFT folder
- For datasets: ONLY offer the approved sample dataset listed above, or validate BYOD format
- Do NOT search HuggingFace for arbitrary datasets — only the approved sample or user's own data
