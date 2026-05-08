---
inclusion: fileMatch
fileMatchPattern: "preference_optimization/**"
---

# GRPO RLVR Context

When working with files under `preference_optimization/`, you are in the GRPO RLVR (Preference Optimization) workload.

## Key Paths
- Recipes: `preference_optimization/grpo_rlvr/sagemaker_code/hf_recipes/<org>/<model>--grpo.yaml`
- Training scripts: `preference_optimization/grpo_rlvr/finetune-tool-call--<org>--<model>-<domain>.py`
- GRPO trainer: `preference_optimization/grpo_rlvr/sagemaker_code/grpo_trainer_v2.py`
- Shell launcher: `preference_optimization/grpo_rlvr/sagemaker_code/sm_accelerate_grpo_train.sh`
- Tools functions: `preference_optimization/grpo_rlvr/sagemaker_code/tools_funcs/`
- Reward functions: `preference_optimization/grpo_rlvr/sagemaker_code/rewards/`

## GRPO Recipe Parameters (differ from SFT)
- `num_generations`: Completions per prompt (typically 8)
- `max_grpo_completion_length`: Max tokens per generation (typically 1024)
- `mask_truncated_completions`: true
- `learning_rate`: 5.0e-6 (lower than SFT)
- `lr_scheduler_type`: "constant" (not cosine)
- `num_train_epochs`: 15-25 (higher than SFT)
- `save_strategy`: "steps" with `save_steps: 25`

## Recipe Naming Convention
`<model-name>--grpo.yaml`

## MCP Servers Available
- `sft_model_helper`: Model discovery on HuggingFace Hub (shared)
- `dataset_prep`: Dataset search, format, S3 upload (shared)
- `grpo_recipe_generator`: Generate GRPO RLVR recipe YAMLs
- `sagemaker_jobs`: Launch GRPO jobs, monitor status, list recent jobs

For the full guided workflow, use `#grpo-rlvr` skill in chat.
