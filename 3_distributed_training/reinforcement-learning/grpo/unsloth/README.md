# GRPOTrainer Fine-Tuning with Unsloth on SageMaker

This project provides a SageMaker-compatible setup for running **GRPO-based reinforcement learning fine-tuning** on an LLM using the [Unsloth](https://github.com/unslothai/unsloth) library.

> ✅ This version supports **single-GPU training only** and does **not use distributed data parallelism**.

---

## 📦 Contents

- `sagemaker_grpo_training_wb_tracing.py`: Main training script using `GRPOTrainer` from `trl` and `FastLanguageModel` from Unsloth, with Weights & Biases tracing.
- `entrypoint.sh`: Entrypoint for SageMaker container execution.
- `launch_unsloth_GRPO_sagemaker_training.py`: Script to upload code to S3 and launch a SageMaker training job.
- `Dockerfile`: Docker environment with `unsloth`, `transformers`, and `rouge`.
- `requirements.txt`: Python dependencies for the training script.
- `estimator_requirements.txt`: SageMaker estimator dependencies.

---

## 🚀 How to Launch (Single GPU)

1. **Build & Push Docker Image**
```bash
docker build -t unsloth-train-grpo .
aws ecr create-repository --repository-name unsloth-train-grpo
docker tag unsloth-train-grpo:latest <your-ecr-uri>/unsloth-train-grpo:latest
docker push <your-ecr-uri>/unsloth-train-grpo:latest
```

2. **Launch Job from Python**
```bash
python launch_unsloth_sagemaker_training.py
```

---

## 📘 Key Features

- **Model**: `meta-llama/Llama-3.2-1B-Instruct` with 4-bit quantization and LoRA patching
- **Dataset**: Automatically loads from HuggingFace: `w601sxs/processed_simpleCoT_b1ade`
- **Reward Functions**:
  - Answer similarity (ROUGE-L)
  - Length-based reward
  - Format compliance

---

## 🛠️ Configurable Parameters

You can control these via CLI or environment variables:

- `--model_output_dir`
- `--per_device_train_batch_size`
- `--gradient_accumulation_steps`
- `--num_train_epochs`

---

## 🧱 Requirements

Your container should include:

```
unsloth
trl
rouge
datasets
peft
transformers
torch >= 2.0
```

---

## 📎 Notes

- `vLLM` is **disabled** due to MKL threading issues inside SageMaker.
- No multi-GPU logic (DDP or SMP) is active in this setup.
- The next steps here is to enable DDP or SMP for this setup
