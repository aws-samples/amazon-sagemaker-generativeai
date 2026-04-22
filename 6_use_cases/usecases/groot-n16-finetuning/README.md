# GR00T N1.6 Fine-Tuning on SageMaker

Fine-tune [NVIDIA GR00T N1.6-3B](https://huggingface.co/nvidia/GR00T-N1.6-3B) on BridgeData V2 using Amazon SageMaker.

## Model Overview

GR00T N1.6 is NVIDIA's 3B-parameter foundation model for humanoid robot control. It uses a diffusion-based action head with a vision-language backbone, designed for multi-modal robot learning. The model expects data in LeRobot V2 format (parquet tables + MP4 videos + JSON metadata) and uses NVIDIA's Isaac-GR00T framework for training and evaluation.

This pipeline fine-tunes the visual encoder, projector, and diffusion model while keeping the LLM backbone frozen (`--no-tune-llm`).

## Folder Structure

```
GR00T_SMTJ/
├── 01_data_preparation.ipynb          # Step 1: Data download, conversion to LeRobot V2, upload
├── 02_training_job.ipynb              # Step 2: Launch SageMaker training job
├── 03_evaluation.ipynb                # Step 3: Open-loop evaluation (fine-tuned vs base)
├── README.md
└── scripts/                           # Uploaded to SageMaker training container
    ├── run_finetuning.sh              # Container entry point (clones Isaac-GR00T, installs, trains)
    ├── bridge_modality_config.py      # Maps data columns to model inputs (camera, state, action, language)
    ├── create_groot_test_dataset.py   # Creates held-out test set (episodes 600+)
    └── requirements.txt               # Python dependencies
```

## Pipeline Steps

### Step 1: Data Preparation (`01_data_preparation.ipynb`)

**What it does:**

1. **Downloads BridgeData V2** from HuggingFace (`VyoJ/BridgeData-V2-Scripted-Images`) — same source as OpenVLA, 8,802 episodes with 3 frames each.

2. **Generates episode folders** — Same intermediate format as OpenVLA: `images/`, `actions.npy`, `language.txt`, `metadata.json` per episode. Actions generated via optical flow on the 3 original frames.

3. **Validates the intermediate dataset** — Checks for NaN, Inf, missing files, correct action shape.

4. **Interpolates frames from 3 → 30 per episode** — GR00T requires longer sequences (action horizon = 16 steps). Uses linear blending between consecutive frames to create smooth 30-frame episodes. This is the most compute-intensive data prep step.

5. **Re-computes optical flow on 30 frames** — After interpolation, runs Farneback optical flow on all 29 consecutive frame pairs (vs only 2 pairs for the original 3 frames). This gives denser, smoother action trajectories.

6. **Encodes MP4 videos via ffmpeg** — Each 30-frame episode is written as temporary PNGs, then encoded to H.264 MP4 using `ffmpeg`. This subprocess spawns once per episode (600 times total) and is the main bottleneck.

7. **Writes LeRobot V2 format:**
   - `data/chunk-000/episode_XXXXXX.parquet` — Per-episode parquet with columns: `observation.state`, `action`, `episode_index`, `frame_index`, `index`, `task_index`, `timestamp`
   - `videos/chunk-000/observation.images.front/episode_XXXXXX.mp4` — Encoded video per episode
   - `meta/info.json` — Dataset schema, feature definitions, splits, paths
   - `meta/episodes.jsonl` — Episode metadata (index, tasks, length)
   - `meta/tasks.jsonl` — Task descriptions with indices
   - `meta/modality.json` — Maps video/state/action/language to data keys
   - `meta/stats.json` — Dataset-wide statistics (mean, std, min, max, q01, q99)

8. **Uploads to S3** — Uploads the entire LeRobot V2 dataset to the SageMaker default bucket.

**Runtime:** ~30-45 minutes for 600 episodes. The extra time vs OpenVLA comes from frame interpolation (image processing on every episode), 15× more optical flow computation, and ffmpeg video encoding (600 subprocess calls).

**Output:** LeRobot V2 dataset at `s3://<bucket>/groot-finetuning/datasets/bridge_lerobot/`

---

### Step 2: SageMaker Training Job (`02_training_job.ipynb`)

**What it does:**

1. **Configures the SageMaker ModelTrainer** with:
   - Instance: `ml.g6e.48xlarge` (8× L40S 48GB) or `ml.p4d.24xlarge` (8× A100 40GB)
   - Docker image: PyTorch 2.7 GPU training container
   - Source code: the `scripts/` folder
   - Environment variables: HF token, `MAX_STEPS`, `BATCH_SIZE`, `SAVE_STEPS`, NCCL settings

2. **Sets up input data channel** — Points `train` to the S3 LeRobot dataset. SageMaker downloads it to `/opt/ml/input/data/train/`.

3. **Launches the training job** asynchronously.

**What happens inside the container (`run_finetuning.sh`):**

1. Logs into HuggingFace
2. Installs system dependencies (`libgl1-mesa-glx`, `libglib2.0-0`, `git-lfs`, `ffmpeg`)
3. **Clones Isaac-GR00T** from GitHub (including submodules)
4. **Installs flash-attention** — Tries a prebuilt wheel from HuggingFace first, falls back to building from source
5. **Patches Isaac-GR00T's `pyproject.toml`** — Removes `tensorrt` and `onnx` dependencies (not needed for training, cause install failures)
6. **Installs Isaac-GR00T** as an editable package (`pip install -e .`)
7. **Downloads GR00T N1.6-3B** base model from HuggingFace
8. **Copies `bridge_modality_config.py`** into the Isaac-GR00T directory
9. **Launches distributed training** via `torch.distributed.run` with `gr00t/experiment/launch_finetune.py`

**Training configuration (CLI flags):**

| Parameter | Value |
|---|---|
| Base model | `nvidia/GR00T-N1.6-3B` |
| Embodiment tag | `NEW_EMBODIMENT` |
| Max steps | 2,000 |
| Global batch size | 32 |
| Save steps | 2,000 |
| Tune LLM | No (`--no-tune-llm`) |
| Tune visual encoder | Yes (`--tune-visual`) |
| Tune projector | Yes (`--tune-projector`) |
| Tune diffusion model | Yes (`--tune-diffusion-model`) |
| Color jitter | brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08 |

**Key difference from OpenVLA:** There are no Accelerate configs or YAML recipes. Isaac-GR00T manages distributed training internally via `torch.distributed.run` and takes all hyperparameters as CLI arguments.

4. **Downloads model artifacts** — After completion, downloads and extracts `model.tar.gz`.

**Output:** Fine-tuned checkpoint at `./model_artifacts/<job_name>/extracted/bridge_finetune/checkpoint-2000`

---

### Step 3: Evaluation (`03_evaluation.ipynb`)

**What it does:**

1. **Sets up Isaac-GR00T locally** — Clones the repo, installs flash-attn, patches and installs GR00T (same process as the training container).

2. **Downloads the base model** from HuggingFace (`nvidia/GR00T-N1.6-3B`).

3. **Evaluates the fine-tuned model on training data** — Runs `gr00t/eval/open_loop_eval.py` with the fine-tuned checkpoint on 5 trajectories from the training set.

4. **Evaluates the base model for comparison** — Since the base model doesn't know about the custom `NEW_EMBODIMENT` tag, the notebook monkey-patches `Gr00tPolicy.__init__` to inject the bridge modality config and normalization statistics at runtime.

5. **Creates a held-out test dataset** — Runs `create_groot_test_dataset.py` to generate 100 test episodes from BridgeData indices 600-699 (never seen during training). Uses the same interpolation + optical flow + LeRobot V2 conversion pipeline.

6. **Evaluates both models on the test set** — Runs open-loop eval on the held-out data to measure generalization.

**Evaluation metrics:**
- **MSE (Mean Squared Error)** — Squared difference between predicted and ground-truth actions
- **MAE (Mean Absolute Error)** — Absolute difference per action dimension

**Expected results (from prior runs on 600 episodes, 2,000 steps):**

| Metric | Base Model | Fine-Tuned | Improvement |
|---|---|---|---|
| MSE (train) | 1.031e-06 | 5.983e-09 | 172× smaller (99.4% ↓) |
| MAE (train) | 3.782e-04 | 3.103e-05 | 12.2× smaller (91.8% ↓) |
| MSE (test) | 1.066e-06 | 5.318e-09 | 200× smaller (99.5% ↓) |
| MAE (test) | 3.766e-04 | 3.008e-05 | 12.5× smaller (92.0% ↓) |

Test performance matches training performance → the model generalizes, doesn't just memorize.

---

## The `bridge_modality_config.py` File

This is the closest thing GR00T has to a "recipe." It tells the model how to interpret the dataset:

- **video** → `front` camera (single view, delta index 0)
- **state** → `arm` (7-DoF, indices 0-6)
- **action** → `arm` (7-DoF, action horizon of 16 steps, absolute representation, non-EEF)
- **language** → `annotation.human.task_description` (maps to `task_index` in the dataset)

The modality keys here must match the keys in `meta/modality.json` of the LeRobot V2 dataset.

---

## Infrastructure Requirements

| Stage | Instance | GPU Memory | Notes |
|---|---|---|---|
| Data Preparation | Any (CPU ok) | None | Needs ffmpeg installed |
| Training | ml.g6e.48xlarge or ml.p4d.24xlarge | 8× 48GB+ | 3B model + visual/diffusion tuning |
| Evaluation | g6e.48xlarge or p4d.24xlarge | 8× 48GB+ | Full model inference |

