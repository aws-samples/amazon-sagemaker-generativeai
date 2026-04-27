#!/bin/bash
# Cosmos Transfer 2.5 — Training Entrypoint for HyperPod EKS
#
# This script runs inside the training container on each pod.
# The PyTorchJob operator sets MASTER_ADDR, MASTER_PORT, WORLD_SIZE,
# RANK, and LOCAL_RANK automatically.
#
# Environment variables (set via PyTorchJob manifest):
#   DATASET_PATH       — FSx mount path to prepared dataset
#   CHECKPOINT_PATH    — FSx mount path to model checkpoints
#   OUTPUT_PATH        — FSx mount path for training output
#   EXPERIMENT         — Experiment name (default: multiview)
#   NUM_GPUS_PER_NODE  — GPUs per node (default: 8)
#   HF_TOKEN           — HuggingFace token (for checkpoint download)
#   ENABLE_WANDB       — Set to "true" to enable W&B logging

set -e

echo "============================================"
echo " Cosmos Transfer 2.5 — HyperPod EKS Training"
echo " Node: $(hostname)"
echo " Rank: ${RANK:-0} / World Size: ${WORLD_SIZE:-1}"
echo " GPUs per node: ${NUM_GPUS_PER_NODE:-8}"
echo "============================================"

cd /workspace/cosmos-transfer2.5

# ── Activate venv or use system Python ──
# If SKIP_VENV is set, use the NGC system Python (PyTorch 2.9)
if [ "${SKIP_VENV}" = "true" ]; then
    echo "[INFO] Using NGC system Python (skipping venv)"
else
    export PATH="$(pwd)/.venv/bin:$PATH"
    source .venv/bin/activate
fi

# ── Set LD_LIBRARY_PATH for transformer_engine ──
NVIDIA_CUBLAS_LIB=$(find .venv -path "*/nvidia/cublas/lib" -type d 2>/dev/null | head -1)
NVIDIA_CUDNN_LIB=$(find .venv -path "*/nvidia/cudnn/lib" -type d 2>/dev/null | head -1)
export LD_LIBRARY_PATH="${NVIDIA_CUBLAS_LIB:+$NVIDIA_CUBLAS_LIB:}${NVIDIA_CUDNN_LIB:+$NVIDIA_CUDNN_LIB:}/usr/local/cuda/lib64"

# ── HuggingFace login (for checkpoint download) ──
if [ -n "${HF_TOKEN}" ]; then
    echo "[INFO] Logging in to HuggingFace..."
    huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential
fi

# ── Download model checkpoint if not on FSx ──
CHECKPOINT_DIR="${CHECKPOINT_PATH:-./checkpoints/Cosmos-Transfer2.5-2B}"
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "[INFO] Downloading Cosmos-Transfer2.5-2B checkpoint..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('nvidia/Cosmos-Transfer2.5-2B', local_dir='$CHECKPOINT_DIR')
print('[INFO] Checkpoint download complete.')
"
fi

# ── Link dataset from FSx mount ──
DATASET_DIR="${DATASET_PATH:-/fsx/datasets/drive_dreams_cosmos_dataset}"
if [ -d "$DATASET_DIR" ] && [ ! -e "assets/drive_dreams_cosmos_dataset" ]; then
    ln -sf "$DATASET_DIR" assets/drive_dreams_cosmos_dataset
    echo "[INFO] Linked dataset: $DATASET_DIR -> assets/drive_dreams_cosmos_dataset"
fi

# ── Set output directory ──
export IMAGINAIRE_OUTPUT_ROOT="${OUTPUT_PATH:-/fsx/output/cosmos_transfer}"
mkdir -p "$IMAGINAIRE_OUTPUT_ROOT"

# ── WandB ──
if [ "${ENABLE_WANDB}" != "true" ]; then
    export WANDB_MODE=disabled
fi

# ── Resolve experiment ──
EXPERIMENT="${EXPERIMENT:-multiview}"
case $EXPERIMENT in
    multiview)
        CONFIG="cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py"
        EXP_NAME="transfer2_auto_multiview_post_train_example"
        ;;
    drive_dreams)
        CONFIG="cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/config.py"
        EXP_NAME="transfer2_drive_dreams_hdmap_posttrain"
        ;;
    *)
        CONFIG="cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/config.py"
        EXP_NAME="$EXPERIMENT"
        ;;
esac

NUM_GPUS=${NUM_GPUS_PER_NODE:-8}

echo ""
echo "[INFO] Experiment:  $EXP_NAME"
echo "[INFO] Config:      $CONFIG"
echo "[INFO] Dataset:     $DATASET_DIR"
echo "[INFO] Output:      $IMAGINAIRE_OUTPUT_ROOT"
echo "[INFO] GPUs/node:   $NUM_GPUS"
echo "[INFO] LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo ""

# ── Verify environment ──
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')
import transformer_engine as te
print(f'transformer_engine: {te.__version__}')
"

# ── Launch training ──
# torchrun is configured by the PyTorchJob operator via env vars:
#   MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK, LOCAL_RANK
# We only set --nproc_per_node; the operator handles the rest.
echo "[INFO] Starting training..."
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=${WORLD_SIZE:-1} \
    --node_rank=${RANK:-0} \
    --master_addr=${MASTER_ADDR:-localhost} \
    --master_port=${MASTER_PORT:-29500} \
    -m scripts.train \
    --config=$CONFIG \
    -- experiment=$EXP_NAME

echo ""
echo "=== Training complete on $(hostname) ==="
echo "Checkpoints: $IMAGINAIRE_OUTPUT_ROOT"
