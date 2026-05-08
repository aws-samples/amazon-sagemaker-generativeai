---
inclusion: fileMatch
fileMatchPattern: "supervised_finetuning/**"
---

# SFT Context

When working with files under `supervised_finetuning/`, you are in the Supervised Fine-Tuning workload.

## Key Paths
- Recipes: `supervised_finetuning/sagemaker_code/hf_recipes/<org>/<model>--<flavor>-<strategy>.yaml`
- Training scripts: `supervised_finetuning/finetune--<org>--<model>.py`
- SFT trainer: `supervised_finetuning/sagemaker_code/sft.py`
- Shell launcher: `supervised_finetuning/sagemaker_code/sm_accelerate_train.sh`

## SFT Strategies
- PEFT/LoRA: `use_peft: true`, `load_in_4bit: true`, `lora_target_modules`, `lora_r`, `lora_alpha`
- Spectrum: `spectrum_config_path`, `max_steps`
- Full: No adapter config, trains all parameters

## Dataset Modalities
- `text` — standard messages (most models)
- `text_reasoning` — extracts `<think>` blocks (GPT-oss, DeepSeek-R1)
- `image` — multimodal with image_url (Llama-Vision, Qwen-VL)
- `audio` — audio content blocks (Qwen2-Audio)

## Recipe Naming Convention
`<model-name>--<vanilla|liger>-<peft-qlora|spectrum|full>.yaml`

## MCP Servers Available
- `sft_model_helper`: Model discovery on HuggingFace Hub
- `dataset_prep`: Dataset search, format (4 modalities), S3 upload
- `sft_recipe_generator`: Generate SFT recipe YAMLs
- `sft_training_workflow`: Generate training scripts + instance recommendations
- `sagemaker_jobs`: Launch SFT jobs, monitor status, list recent jobs

For the full guided workflow, use `#supervised-finetuning` skill in chat.
