#!/bin/bash
# Cosmos Transfer 2.5 — Training Entrypoint (v2 — clean ubuntu base)
#
# Works with Dockerfile.v2 (ubuntu:22.04 + uv sync).
# The venv is at /workspace/cosmos-transfer2.5/.venv
#
# Environment variables (set via PyTorchJob manifest):
#   DATASET_PATH       — Path to prepared dataset (FSx or local)
#   CHECKPOINT_PATH    — Path to Cosmos-Transfer2.5-2B checkpoints
#   OUTPUT_PATH        — Path for training output
#   NUM_GPUS_PER_NODE  — GPUs per node (default: 8)
#   HF_TOKEN           — HuggingFace token (for checkpoint download)
#   WANDB_API_KEY      — Weights & Biases API key (optional)
#   COSMOS_CP_SIZE     — Context parallelism size (default: 8)
#   MAX_ITER           — Override max iterations (optional)

set -e

echo "============================================"
echo " Cosmos Transfer 2.5 — Drive-Dreams Training"
echo " Node: $(hostname)"
echo " Rank: ${RANK:-0} / World Size: ${WORLD_SIZE:-1}"
echo " GPUs per node: ${NUM_GPUS_PER_NODE:-8}"
echo " Context parallelism: ${COSMOS_CP_SIZE:-8}"
echo "============================================"

cd /workspace/cosmos-transfer2.5

# ── Activate venv ──
source .venv/bin/activate

# ── Set LD_LIBRARY_PATH for transformer_engine / NVRTC ──
NVIDIA_LIBS=$(find .venv -path "*/nvidia/*/lib" -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NVIDIA_LIBS}/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# ── HuggingFace login ──
if [ -n "${HF_TOKEN}" ]; then
    python3 -c "from huggingface_hub import login; login(token='${HF_TOKEN}')" 2>/dev/null || true
fi

# ── Download model checkpoint if needed ──
CHECKPOINT_DIR="${CHECKPOINT_PATH:-./checkpoints/Cosmos-Transfer2.5-2B}"
if [ ! -d "$CHECKPOINT_DIR" ] || [ -z "$(ls -A $CHECKPOINT_DIR 2>/dev/null)" ]; then
    echo "[INFO] Downloading Cosmos-Transfer2.5-2B checkpoint..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('nvidia/Cosmos-Transfer2.5-2B', local_dir='$CHECKPOINT_DIR')
"
fi

# ── Link dataset and checkpoints ──
DATASET_DIR="${DATASET_PATH:-/data/drive_dreams_cosmos_dataset}"
mkdir -p assets checkpoints
ln -sf "$DATASET_DIR" assets/drive_dreams_cosmos_dataset 2>/dev/null || true
ln -snf "$CHECKPOINT_DIR" checkpoints/Cosmos-Transfer2.5-2B 2>/dev/null || true

# ── Set output directory ──
export IMAGINAIRE_OUTPUT_ROOT="${OUTPUT_PATH:-/workspace/output}"
mkdir -p "$IMAGINAIRE_OUTPUT_ROOT"

# ── WandB ──
if [ -n "${WANDB_API_KEY}" ]; then
    export WANDB_MODE=${WANDB_MODE:-online}
else
    export WANDB_MODE=offline
fi
touch /root/.netrc 2>/dev/null || true

# ── Verify environment ──
echo ""
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')
import transformer_engine as te; print(f'transformer_engine: {te.__version__}')
import flash_attn; print(f'flash_attn: OK')
import cosmos_transfer2; print(f'cosmos_transfer2: OK')
"

# ── Launch training ──
NUM_GPUS=${NUM_GPUS_PER_NODE:-8}
echo ""
echo "[INFO] Starting training..."
echo "[INFO] Dataset: $DATASET_DIR"
echo "[INFO] Checkpoints: $CHECKPOINT_DIR"
echo "[INFO] Output: $IMAGINAIRE_OUTPUT_ROOT"
echo ""

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=${WORLD_SIZE:-1} \
    --node_rank=${RANK:-0} \
    --master_addr=${MASTER_ADDR:-localhost} \
    --master_port=${MASTER_PORT:-29500} \
    -m scripts.train \
    --config=cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/config.py \
    -- experiment=transfer2_drive_dreams_hdmap_posttrain

echo ""
echo "=== Training complete on $(hostname) ==="
