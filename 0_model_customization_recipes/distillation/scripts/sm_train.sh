#!/bin/bash
#
# SageMaker SeqKD (sequence-level knowledge distillation) launcher.
# Installs deps, resolves the per-node GPU count, then launches train_seqkd.py under torchrun.
#
# Usage (passed through by ModelTrainer's SourceCode command):
#   ./sm_train.sh --max-length 4096 --lora-r 64 --lr 2e-5 --epochs 2 \
#                 --batch-size 1 --grad-accum 2 --fft false [--max-steps N ...]
#
# The train channel mounts at /opt/ml/input/data/train (val at /opt/ml/input/data/val); the model
# is written to /opt/ml/model. train_seqkd.py reads those defaults, so only hyperparameters are passed.
#
# NOTE (Qwen3.5-4B is a linear-attn/conv1d HYBRID): train_seqkd.py's --install-kernels (default on)
# force-installs triton 3.8 + flash-linear-attention 0.5.2 with --no-deps at start (the fla/triton pair
# collides with the base image's torch pin under a normal resolver). Keep --attn sdpa on SageMaker's
# managed driver (flash_attention_2 CUDA-IMAs there). Both are train_seqkd.py defaults.

set -euo pipefail
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. Deps (the verl SMTJ image already ships torch/transformers/flash-attn; this adds trl/peft/datasets).
python3 -m pip install --upgrade uv >/dev/null 2>&1 || true
if command -v uv &>/dev/null; then
    uv pip install --system -r "${SCRIPT_DIR}/requirements_verl.txt"
else
    python3 -m pip install -r "${SCRIPT_DIR}/requirements_verl.txt"
fi

# 2. Per-node GPU count.
NUM_GPUS="${SM_NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d '[:space:]')}"
[[ "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] could not resolve GPU count"; exit 1; }
echo "[INFO] launching train_seqkd.py on ${NUM_GPUS} GPU(s) with args: $*"

# 3. Launch. train_seqkd.py handles the --no-deps kernel bootstrap and rank-0 causal-conv1d itself.
torchrun --nnodes=1 --nproc_per_node="${NUM_GPUS}" "${SCRIPT_DIR}/train_seqkd.py" "$@"
