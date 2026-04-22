# OpenVLA LoRA Fine-Tuning on SageMaker

Fine-tune [OpenVLA-7B](https://huggingface.co/openvla/openvla-7b) with LoRA on BridgeData V2 using Amazon SageMaker.

## Model Overview

OpenVLA is a 7B-parameter Vision-Language-Action model built on a Prismatic VLM backbone, pre-trained on the Open X-Embodiment dataset (970k robot demonstrations). It predicts 7-DoF continuous robot actions (x, y, z, roll, pitch, yaw, gripper) by discretizing each dimension into 256 bins mapped to the last 256 tokens in the vocabulary.

This pipeline fine-tunes only LoRA adapter weights (rank 16) on top of the frozen base model, targeting the attention projection layers (`q_proj`, `v_proj`, `k_proj`, `out_proj`).

## Folder Structure

```
OpenVLA_SMTJ/
├── 01_data_preparation.ipynb          # Step 1: Data download, generation, conversion, upload
├── 02_training_job.ipynb              # Step 2: Launch SageMaker training job
├── 03_evaluation.ipynb                # Step 3: Evaluate fine-tuned vs base model
├── README.md
└── scripts/                           # Uploaded to SageMaker training container
    ├── run_finetuning.sh              # Container entry point (multi-node aware)
    ├── train_openvla_lora.py          # Main LoRA training script
    ├── validate_openvla.py            # Standalone validation script
    ├── requirements.txt               # Python dependencies for the container
    ├── recipes/
    │   └── openvla_config.yaml        # Training hyperparameters
    ├── accelerate_configs/
    │   ├── ddp.yaml                   # Multi-GPU DDP config
    │   ├── deepspeed_zero2.yaml       # DeepSpeed ZeRO Stage 2
    │   └── deepspeed_zero3.yaml       # DeepSpeed ZeRO Stage 3
    └── utils/
        ├── __init__.py
        ├── openvla_utils.py           # ActionTokenizer, prompt builders
        └── data_utils.py              # Optical flow, dataset validation, HF conversion
```

## Pipeline Steps

### Step 1: Data Preparation (`01_data_preparation.ipynb`)

**What it does:**

1. **Downloads BridgeData V2** from HuggingFace (`VyoJ/BridgeData-V2-Scripted-Images`) — 8,802 episodes, each with 3 frames (first, intermediate, frame_43).

2. **Generates 7-DoF actions via optical flow** — For each pair of consecutive frames, computes dense optical flow (Farneback method) and maps mean pixel displacement to action dimensions. Only x/y dimensions get non-zero values; z, rotation, and gripper remain zero (can't be inferred from 2D images).

3. **Saves episode folders** — Each episode becomes a directory with:
   - `images/` — 3 JPEG frames
   - `actions.npy` — shape `(3, 7)` float32 array
   - `language.txt` — instruction string ("move object to target")
   - `metadata.json` — episode metadata

4. **Validates the intermediate dataset** — Checks every episode for missing files, NaN/Inf actions, and correct shape.

5. **Converts to HuggingFace DatasetDict** — Loads all valid episodes into a `DatasetDict` with `train` (90%) and `validation` (10%) splits. Saved to disk in Arrow format.

6. **Uploads to S3** — Uploads the HuggingFace dataset to the SageMaker default bucket for the training job.

**Runtime:** ~15-20 minutes for 600 episodes. The process is fast because it works directly with the 3 original frames — no interpolation, no video encoding.

**Output:** HuggingFace DatasetDict at `s3://<bucket>/openvla-finetuning/datasets/bridge_hf_synthetic/`

---

### Step 2: SageMaker Training Job (`02_training_job.ipynb`)

**What it does:**

1. **Configures the SageMaker ModelTrainer** with:
   - Instance: `ml.g6e.48xlarge` (8× L40S 48GB) or `ml.p4d.24xlarge` (8× A100 40GB)
   - Docker image: PyTorch 2.7 GPU training container
   - Source code: the `scripts/` folder (uploaded automatically)
   - Environment variables: HF token, accelerate config path, training recipe path

2. **Sets up input data channel** — Points the `train` channel to the S3 dataset URI. SageMaker downloads it to `/opt/ml/input/data/train/` inside the container.

3. **Launches the training job** — Submits asynchronously (non-blocking).

**What happens inside the container (`run_finetuning.sh`):**

1. Logs into HuggingFace (to download `openvla/openvla-7b`)
2. Installs flash-attention
3. Detects multi-node topology from SageMaker environment variables (`SM_HOSTS`, `SM_CURRENT_HOST`, `SM_NUM_GPUS`)
4. Launches `accelerate launch` with the selected config (DDP or DeepSpeed) pointing to `train_openvla_lora.py`

**What happens in `train_openvla_lora.py`:**

1. Loads the OpenVLA-7B model and processor from HuggingFace
2. Freezes all base model weights
3. Applies LoRA adapters (rank 16, alpha 16) to `q_proj`, `v_proj`, `k_proj`, `out_proj`
4. Loads the HuggingFace dataset from `/opt/ml/input/data/train/`
5. Builds training prompts: `"In: What action should the robot take to {instruction}?\nOut:"` followed by discretized action tokens
6. Trains with cross-entropy loss on action tokens only (instruction tokens masked with -100)
7. Saves checkpoints every 500 steps, final model + LoRA weights to `/opt/ml/model/`

**Training config (from `recipes/openvla_config.yaml`):**

| Parameter | Value |
|---|---|
| LoRA rank | 16 |
| Learning rate | 1e-4 |
| LR scheduler | Cosine |
| Warmup steps | 500 |
| Batch size | 4 per device |
| Epochs | 10 |
| Mixed precision | bf16 |
| Gradient checkpointing | Enabled |
| Optimizer | AdamW |

4. **Downloads model artifacts** — After the job completes, downloads and extracts `model.tar.gz` from S3.

**Output:** Fine-tuned LoRA weights + processor at `./model_artifacts/<job_name>/extracted/`

---

### Step 3: Evaluation (`03_evaluation.ipynb`)

**What it does:**

1. **Loads the fine-tuned model** and the base model sequentially (to fit in GPU memory).

2. **Runs inference on validation samples** — For each sample:
   - Builds the prompt from the language instruction
   - Runs a forward pass through the model
   - Extracts the last 7 logits (action dimensions), takes argmax to get token IDs
   - Decodes token IDs back to continuous actions via `ActionTokenizer`

3. **Computes error metrics:**
   - **L1 Error (Mean Absolute Error)** — Average absolute difference per action dimension
   - **L2 Error (RMSE)** — Root mean squared error across dimensions

4. **Compares base vs fine-tuned** — Prints a side-by-side table with improvement percentages.

**Expected results (from prior runs):**

| Metric | Base Model | Fine-Tuned | Improvement |
|---|---|---|---|
| L1 Mean | ~0.31 | ~0.17 | ~44% |
| L2 Mean | ~0.41 | ~0.33 | ~21% |

---

## Infrastructure Requirements

| Stage | Instance | GPU Memory | Notes |
|---|---|---|---|
| Data Preparation | Any (CPU ok) | None | Runs in notebook instance |
| Training | ml.g6e.48xlarge or ml.p4d.24xlarge | 8× 48GB+ | 7B model + LoRA + gradients |
| Evaluation | g6e.xlarge+ | 1× 48GB+ | Single GPU inference |

