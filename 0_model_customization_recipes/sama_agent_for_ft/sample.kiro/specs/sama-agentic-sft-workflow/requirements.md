# Requirements Document

## Introduction

This feature provides an agentic fine-tuning workflow for the SAMA (Science Agents) project, covering both Supervised Fine-Tuning (SFT) and GRPO RLVR (Preference Optimization) workloads. The system is composed of six MCP servers that orchestrate the full pipeline: model discovery on HuggingFace Hub, dataset preparation across modalities, recipe generation, training script creation, SageMaker job launch, and job monitoring. Two skills (`#supervised-finetuning` and `#grpo-rlvr`) prescribe the end-to-end flows with educational and fast modes.

## Glossary

- **Model_Helper_Server** (`sft_model_helper`): Queries HuggingFace Hub, filters models by family/size, returns metadata including license.
- **Dataset_Prep_Server** (`dataset_prep`): Searches HF Hub for datasets, formats them into messages-style JSONL for four modalities (text, text_reasoning, image, audio), and uploads to S3.
- **SFT_Recipe_Server** (`sft_recipe_generator`): Generates SFT training recipe YAMLs for PEFT/LoRA, Spectrum, or Full strategies.
- **GRPO_Recipe_Server** (`grpo_recipe_generator`): Generates GRPO RLVR training recipe YAMLs with generation-specific parameters.
- **Training_Workflow_Server** (`sft_training_workflow`): Generates SageMaker training scripts from templates with instance recommendations.
- **SageMaker_Job_Server** (`sagemaker_jobs`): Launches SFT and GRPO training jobs on SageMaker and monitors their status.
- **Approved_Model_Families**: The 18 HuggingFace organizations whose models are supported (meta-llama, Qwen, openai, google, microsoft, deepseek-ai, mistralai, CohereForAI, nvidia, apple, ibm-granite, Salesforce, moonshotai, MiniMaxAI, THUDM, HuggingFaceTB, tiiuae, NousResearch).

## Requirements

### Requirement 1: Model Discovery by Family

**User Story:** As a data scientist, I want to search for models on HuggingFace Hub by model family, so that I can explore available models within a specific organization.

#### Acceptance Criteria

1. WHEN a user provides a model family name, THE Model_Helper_Server SHALL query HuggingFace Hub and return models belonging to that family with model identifier, parameter count, downloads, and likes.
2. WHEN a model family does not exist, THE Model_Helper_Server SHALL return an empty result set with a descriptive message.
3. THE Model_Helper_Server SHALL only return models from the Approved_Model_Families list.

### Requirement 2: Model Filtering by Size Range

**User Story:** As a data scientist, I want to filter models by parameter size range to fit my compute budget.

#### Acceptance Criteria

1. WHEN a user specifies min/max parameter counts, THE Model_Helper_Server SHALL return only models within that range (inclusive).
2. IF max < min, THE Model_Helper_Server SHALL return an error indicating the range is invalid.

### Requirement 3: Model Selection with License Check

**User Story:** As a data scientist, I want to select a model and verify its license before proceeding.

#### Acceptance Criteria

1. WHEN a user selects a model, THE Model_Helper_Server SHALL return full metadata including license information.
2. THE workflow SHALL display the license to the user and require confirmation before proceeding to recipe generation.

### Requirement 4: Dataset Search and Discovery

**User Story:** As a data scientist, I want to search HuggingFace Hub for datasets by keyword.

#### Acceptance Criteria

1. WHEN a user provides a keyword, THE Dataset_Prep_Server SHALL return matching datasets with id, downloads, and tags.
2. THE Dataset_Prep_Server SHALL allow inspecting a dataset's schema, columns, and sample rows before formatting.

### Requirement 5: Modality-Aware Dataset Preparation

**User Story:** As a data scientist, I want to format datasets into messages-style JSONL appropriate for my model's modality.

#### Acceptance Criteria

1. THE Dataset_Prep_Server SHALL support four modality formats: text, text_reasoning, image, and audio.
2. FOR text_reasoning modality, THE Dataset_Prep_Server SHALL extract `<think>` blocks into a `thinking` field on assistant messages.
3. FOR image modality, THE Dataset_Prep_Server SHALL produce messages with `image_url` content blocks.
4. FOR audio modality, THE Dataset_Prep_Server SHALL produce messages with audio content blocks.
5. THE Dataset_Prep_Server SHALL write output as JSONL with one `{"messages": [...]}` object per line.

### Requirement 6: S3 Upload

**User Story:** As a data scientist, I want to upload prepared datasets to S3 for SageMaker training.

#### Acceptance Criteria

1. THE Dataset_Prep_Server SHALL upload a local JSONL file to S3 and return the S3 URI.
2. IF no bucket is specified, THE Dataset_Prep_Server SHALL use the SageMaker default bucket.

### Requirement 7: SFT Recipe Generation

**User Story:** As a data scientist, I want to generate SFT recipe YAMLs for PEFT, Spectrum, or Full strategies.

#### Acceptance Criteria

1. THE SFT_Recipe_Server SHALL generate valid YAML containing all fields required by `sft.py`: model_name_or_path, dataset_id_or_path, num_train_epochs, per_device_train_batch_size, output_dir.
2. THE SFT_Recipe_Server SHALL write recipes to `supervised_finetuning/sagemaker_code/hf_recipes/<org>/<model>--<flavor>-<strategy>.yaml`.
3. THE SFT_Recipe_Server SHALL accept custom hyperparameter overrides while preserving defaults for unspecified parameters.

### Requirement 8: GRPO Recipe Generation

**User Story:** As a data scientist, I want to generate GRPO RLVR recipe YAMLs with generation-specific parameters.

#### Acceptance Criteria

1. THE GRPO_Recipe_Server SHALL generate YAML with GRPO-specific fields: num_generations, max_grpo_completion_length, mask_truncated_completions, constant LR scheduler, step-based checkpointing.
2. THE GRPO_Recipe_Server SHALL write recipes to `preference_optimization/grpo_rlvr/sagemaker_code/hf_recipes/<org>/<model>--grpo.yaml`.
3. THE GRPO_Recipe_Server SHALL NOT include PEFT/LoRA/Spectrum parameters in GRPO recipes.

### Requirement 9: SFT Training Script Generation

**User Story:** As a data scientist, I want to generate a SageMaker training script with strategy selection blocks.

#### Acceptance Criteria

1. THE Training_Workflow_Server SHALL generate a Python script with active code for the selected strategy and commented-out blocks for the other two.
2. THE Training_Workflow_Server SHALL recommend instance types based on model parameter count and strategy.

### Requirement 10: SageMaker Job Launch

**User Story:** As a data scientist, I want to launch training jobs directly from the agent.

#### Acceptance Criteria

1. THE SageMaker_Job_Server SHALL launch SFT jobs using `sm_accelerate_train.sh` with the recipe path.
2. THE SageMaker_Job_Server SHALL launch GRPO jobs using `sm_accelerate_grpo_train.sh` with recipe path, tools_script, reward_fn, and zero_stage arguments.
3. THE SageMaker_Job_Server SHALL return the job name for monitoring.

### Requirement 11: Job Monitoring

**User Story:** As a data scientist, I want to check the status of my training jobs.

#### Acceptance Criteria

1. THE SageMaker_Job_Server SHALL return job status, timing info, failure reason (if failed), and model artifacts location (if completed).
2. THE SageMaker_Job_Server SHALL list recent training jobs with optional name filtering.

### Requirement 12: Workflow Mode Support

**User Story:** As a data scientist, I want educational or fast mode to control the verbosity of the workflow.

#### Acceptance Criteria

1. THE workflow SHALL ask the user to choose educational or fast mode at the start.
2. IN educational mode, THE workflow SHALL explain each step, concept, and tradeoff before executing.
3. IN fast mode, THE workflow SHALL use sensible defaults and only ask required questions.

### Requirement 13: MCP Server Structure Compliance

**User Story:** As a developer, I want all MCP servers to follow consistent conventions.

#### Acceptance Criteria

1. ALL six MCP servers SHALL be implemented as FastMCP servers with pyproject.toml, Python package with `__init__.py` and `server.py`, and README.md.
2. ALL tools SHALL return dictionaries with a `status` field indicating success or error.
3. ALL tools SHALL include a `next_step` field guiding the agent to the next action in the workflow.

### Requirement 14: Error Handling

**User Story:** As a data scientist, I want clear error messages when something goes wrong.

#### Acceptance Criteria

1. IF the HuggingFace Hub API is unreachable, THE Model_Helper_Server and Dataset_Prep_Server SHALL return descriptive connectivity errors.
2. IF recipe or script generation fails to write to disk, THE server SHALL return the file path and failure reason.
3. IF a SageMaker job launch fails, THE SageMaker_Job_Server SHALL return the error from the SageMaker API.
