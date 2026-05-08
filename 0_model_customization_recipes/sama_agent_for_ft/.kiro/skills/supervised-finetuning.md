---
inclusion: manual
---

# SAMA Supervised Fine-Tuning (SFT) Workflow

You are orchestrating an end-to-end Supervised Fine-Tuning workflow. Follow the prescribed steps IN ORDER.

## Model Discovery

Search is OPEN — any text-generation model on HuggingFace Hub is fair game. Use `search_models_by_family`, `search_models_by_keyword`, or `filter_models_by_size` to help the user find what they need. Do NOT restrict to a hardcoded list.

The user can enter the flow at any point:
- **By name**: "I want to fine-tune Qwen/Qwen3-4B" → skip straight to license check
- **By family**: "Show me Mistral models" → `search_models_by_family`
- **By size**: "I need a model under 3B" → `search_models_by_keyword` + `filter_models_by_size`
- **By task**: "I need a vision-language model" → `search_models_by_keyword`

## MCP Servers Used

| Server | Tools |
|---|---|
| `sft_model_helper` | search_models_by_family, search_models_by_keyword, filter_models_by_size, get_model_details |
| `dataset_prep` | search_datasets, inspect_dataset, prepare_dataset, upload_to_s3 |
| `sft_recipe_generator` | generate_sft_recipe, list_existing_recipes |
| `sft_training_workflow` | generate_training_script, get_instance_recommendation |
| `sagemaker_jobs` | launch_sft_training_job, monitor_training_job, list_recent_training_jobs |

## Workflow Modes

At the START, ask: "Would you like **educational mode** (I explain each step) or **fast mode** (minimal questions, sensible defaults)?"

### Educational Mode
- Explain WHAT each step does and WHY before executing
- Explain PEFT/LoRA vs Spectrum vs Full fine-tuning tradeoffs
- Explain what each recipe parameter controls
- Explain instance type cost/performance tradeoffs

### Fast Mode
- No explanations, use defaults, only ask required questions
- Default strategy: PEFT

## Prescribed Flow

### Step 1: Model Selection

Help the user find and select a model. Once selected, call `get_model_details`. Store `model_id` and `parameters`.

**License Check (MANDATORY)**: Display the `license` field to the user. If restrictive (non-commercial, gated), warn them. Confirm acceptance before proceeding.

### Step 2: Dataset

Ask the user: **"Do you have your own dataset already prepared, or would you like to use a sample dataset?"**

#### Path A: Bring Your Own Data (BYOD)

1. Ask the user for the S3 URI or local path to their dataset
2. If local, read a few lines and validate the format:
   - Must be JSONL with one `{"messages": [...]}` per line
   - Each message must have `role` (system/user/assistant) and `content`
   - For image data: `content` should contain `image_url` blocks with base64 or path
   - For audio data: `content` should contain `audio` blocks with `audio_url`
   - For reasoning models: assistant messages may have a `thinking` field
3. If valid, confirm: "Dataset looks good — N samples, messages format verified."
4. If invalid, explain what's wrong and how to fix it
5. Ask for the S3 URI to upload to (if local), or confirm the existing S3 URI

#### Path B: Sample Datasets

Offer ONLY these pre-approved sample datasets. Do NOT search HuggingFace for other datasets.

| Use Case | Dataset | Modality | Reference Notebook |
|---|---|---|---|
| Text (financial) | `Josephgflowers/Finance-Instruct-500k` | text | #[[file:supervised_finetuning/finetune--meta-llama--Llama-3.2-3B-Instruct.ipynb]] |
| Reasoning + tool use | `interstellarninja/hermes_reasoning_tool_use` | text_reasoning | #[[file:supervised_finetuning/finetune--openai--gpt-oss-20b.ipynb]] |

When preparing the reasoning dataset, the agent MUST check the target model:
- If the model is `openai/gpt-oss-*` → call `prepare_hermes_reasoning_tool_use(gpt_oss=True)`. This puts `thinking` as a separate field on each message (GPT-oss harmony format).
- For ALL other models (Qwen, DeepSeek, Llama, Mistral, etc.) → call `prepare_hermes_reasoning_tool_use(gpt_oss=False)` (default). This puts thinking inline as `<think>...</think>` tags in the assistant content.
| Vision (table QA) | `AI-4-Everyone/Visual-TableQA` | image | #[[file:supervised_finetuning/finetune--Qwen--Qwen3-VL-2B-Instruct.ipynb]] |
| Audio (sound understanding) | `mesolitica/AudioSet-Audio-Instructions` | audio | #[[file:supervised_finetuning/finetune--Qwen--Qwen2-Audio-7B-Instruct.ipynb]] |

When the user picks a sample dataset:
1. Use `prepare_dataset` with the correct modality and column mappings from the reference notebook
2. For `text`: columns are `user`, `assistant` with a system prompt
3. For `text_reasoning`: use `conversations_column="conversations"` — extracts `<think>` blocks into `thinking` field
4. For `image`: convert PIL images to base64, build `image_url` content blocks
5. For `audio`: save audio files, build `audio_url` content blocks with `file://` paths
6. Ask the user: "Where should I upload the dataset? Provide a full S3 URI"
7. Use `upload_to_s3` with the user-provided S3 URI

### Step 3: Recipe Generation

Ask for:
1. **Strategy**: PEFT/LoRA, Spectrum, or Full (default: PEFT in fast mode)
2. **Custom hyperparameters** or defaults

Call `generate_sft_recipe` with:
- `recipes_base_dir` = `../supervised_finetuning/sagemaker_code/hf_recipes`
- The dataset name from Step 2

**Reference recipes** (use these as parameter baselines):
- #[[file:supervised_finetuning/sagemaker_code/hf_recipes/meta-llama/Llama-3.2-3B-Instruct--vanilla-peft-qlora.yaml]]
- #[[file:supervised_finetuning/sagemaker_code/hf_recipes/Qwen/Qwen3-4B-vanilla-peft-qlora.yaml]]
- #[[file:supervised_finetuning/sagemaker_code/hf_recipes/openai/gpt-oss-20b--vanilla-peft-qlora.yaml]]

### Step 4: Training Script Generation

Call `generate_training_script` with:
- `model_id`, `strategy`, `dataset_name` from previous steps
- `param_count` from Step 1
- `output_dir` = `../supervised_finetuning`

### Step 5: Launch Training Job (Optional)

If the user wants to launch directly:
1. Call `launch_sft_training_job` with recipe path, S3 data URI, and instance type
2. Show the job name and how to monitor

### Step 6: Summary

Summarize: recipe YAML path, training script path, instance type, S3 data URI, job name (if launched).

## Rules

- NEVER skip the license check
- NEVER generate a recipe without confirming model selection
- NEVER generate a training script without first generating the recipe
- ALWAYS pass actual param_count for accurate instance recommendations
- ALWAYS ask for S3 upload path — never use a default
- For datasets: ONLY offer the 4 sample datasets listed above, or validate BYOD format
- Do NOT search HuggingFace for arbitrary datasets — only the 4 approved samples or user's own data
