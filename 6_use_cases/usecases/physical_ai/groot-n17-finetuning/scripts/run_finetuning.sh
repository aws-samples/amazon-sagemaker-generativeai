#!/bin/bash
# GR00T N1.7 — SageMaker Training Job Entry Point
# This script runs inside the SageMaker training container.
#
# SageMaker provides:
#   /opt/ml/input/data/train/  — training dataset (downloaded from S3)
#   /opt/ml/model/             — where to write final model artifacts
#   /opt/ml/checkpoints/       — checkpoint storage (synced to S3)
#   SM_NUM_GPUS                — number of GPUs on the instance
#   SM_HOSTS / SM_CURRENT_HOST — multi-node info

set -e

echo "============================================"
echo " GR00T N1.7 — SageMaker Training Job"
echo "============================================"

# Debug: show what SageMaker uploaded to /opt/ml/code
echo "[DEBUG] Contents of /opt/ml/code/:"
find /opt/ml/code -maxdepth 3 -type f 2>/dev/null | head -30 || true
echo "---"

# --- HuggingFace login ---
if [ -n "${HF_TOKEN}" ]; then
    echo "Logging in to HuggingFace..."
    huggingface-cli login --token ${HF_TOKEN}
elif [ -n "${HF_token}" ]; then
    echo "Logging in to HuggingFace..."
    huggingface-cli login --token ${HF_token}
fi

# --- Install system deps ---
echo "=== Installing system dependencies ==="
apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 git-lfs ffmpeg 2>/dev/null || true

# Verify ffmpeg is available (required for video loading)
if ! command -v ffmpeg &>/dev/null; then
    echo "[WARN] ffmpeg not found via apt. Trying conda/pip fallback..."
    pip install --no-cache-dir imageio-ffmpeg 2>/dev/null || true
    # Try static ffmpeg binary as last resort
    if ! command -v ffmpeg &>/dev/null; then
        echo "[INFO] Downloading static ffmpeg binary..."
        curl -sL https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz | tar xJ -C /tmp/
        cp /tmp/ffmpeg-*-amd64-static/ffmpeg /usr/local/bin/
        cp /tmp/ffmpeg-*-amd64-static/ffprobe /usr/local/bin/
        chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe
    fi
fi
echo "[INFO] ffmpeg: $(which ffmpeg 2>/dev/null || echo 'NOT FOUND')"

# --- Clone and install Isaac-GR00T ---
echo "=== Cloning Isaac-GR00T (n1.7-release) ==="
GROOT_DIR="/opt/ml/code/Isaac-GR00T"
if [ ! -d "$GROOT_DIR" ]; then
    git clone --recurse-submodules --branch n1.7-release https://github.com/NVIDIA/Isaac-GR00T.git "$GROOT_DIR"
fi
cd "$GROOT_DIR"
git submodule update --init --recursive

# --- Install flash-attn (prebuilt wheel) ---
echo "=== Installing flash-attn ==="
PYTHON_VER=$(python3 -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
CXX11_ABI=$(python3 -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")
FLASH_WHEEL_URL="https://huggingface.co/strangertoolshf/flash_attention_2_wheelhouse/resolve/main/flash_attn-2.8.3/torch2.7/cu12/abi${CXX11_ABI}/${PYTHON_VER}/flash_attn-2.8.3+cu12torch2.7cxx11abi${CXX11_ABI}-${PYTHON_VER}-${PYTHON_VER}-linux_x86_64.whl"
pip install --no-cache-dir "$FLASH_WHEEL_URL" || pip install --no-cache-dir --no-build-isolation "flash-attn==2.7.4.post1" || echo "WARNING: flash-attn install failed"

# --- Patch and install GR00T ---
echo "=== Installing GR00T ==="
PYPROJECT="pyproject.toml"
if grep -q "tensorrt" "$PYPROJECT"; then
    cp "$PYPROJECT" "${PYPROJECT}.bak"
    sed -i '/tensorrt/d' "$PYPROJECT"
    sed -i '/onnx/d' "$PYPROJECT"
fi
# Relax Python version pin (GR00T requires ==3.10.* but SageMaker container has 3.12)
sed -i 's/requires-python\s*=\s*"==3\.10\.\*"/requires-python = ">=3.10"/' "$PYPROJECT"
sed -i 's/python_requires\s*=\s*"==3\.10\.\*"/python_requires = ">=3.10"/' "$PYPROJECT"
pip install -e . --no-build-isolation || { [ -f "${PYPROJECT}.bak" ] && mv "${PYPROJECT}.bak" "$PYPROJECT"; exit 1; }
[ -f "${PYPROJECT}.bak" ] && mv "${PYPROJECT}.bak" "$PYPROJECT"

pip install "huggingface-hub>=0.30.0,<1.0"

# --- Install torchcodec (required for video loading in LeRobot datasets) ---
echo "=== Installing torchcodec ==="
pip install --no-cache-dir torchcodec || echo "WARNING: torchcodec install failed"

# Check if torchcodec is importable; if not, patch GR00T to use ffmpeg backend
TORCHCODEC_OK=0
python3 -c "import torchcodec; print('torchcodec OK')" 2>/dev/null && TORCHCODEC_OK=1 || true

if [ "$TORCHCODEC_OK" -eq 0 ]; then
    echo "[WARN] torchcodec not importable. Patching GR00T to use ffmpeg video backend..."
    # Replace all default video_backend="torchcodec" with "ffmpeg" across the codebase
    find "${GROOT_DIR}" -name "*.py" -type f -exec grep -l 'torchcodec' {} \; 2>/dev/null | while read f; do
        sed -i 's/video_backend\s*=\s*"torchcodec"/video_backend = "ffmpeg"/g' "$f"
        sed -i 's/video_backend: str = "torchcodec"/video_backend: str = "ffmpeg"/g' "$f"
    done
    echo "[INFO] Patched all torchcodec defaults to ffmpeg"
fi

# --- Download base model ---
echo "=== Downloading GR00T N1.7-3B ==="
python3 -c "
from huggingface_hub import snapshot_download
path = snapshot_download('nvidia/GR00T-N1.7-3B')
print(f'Model downloaded to: {path}')
"

# --- Resolve paths ---
DATASET_PATH="/opt/ml/input/data/train"
OUTPUT_DIR="/opt/ml/model/bridge_finetune"
CHECKPOINT_DIR="/opt/ml/checkpoints"

# Find modality config — check common SageMaker source locations
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODALITY_CONFIG=""
for candidate in \
    "$SCRIPT_DIR/utils/bridge_modality_config.py" \
    "$SCRIPT_DIR/bridge_modality_config.py" \
    "/opt/ml/code/utils/bridge_modality_config.py" \
    "/opt/ml/code/bridge_modality_config.py" \
    "/opt/ml/code/scripts/utils/bridge_modality_config.py" \
    "/opt/ml/input/data/code/utils/bridge_modality_config.py"; do
    if [ -f "$candidate" ]; then
        MODALITY_CONFIG="$candidate"
        echo "[INFO] Found modality config at: $candidate"
        break
    fi
done

if [ -z "$MODALITY_CONFIG" ]; then
    FOUND=$(find /opt/ml -name "bridge_modality_config.py" -not -path "*/Isaac-GR00T/*" -type f 2>/dev/null | head -1)
    if [ -n "$FOUND" ]; then
        MODALITY_CONFIG="$FOUND"
        echo "[INFO] Found modality config via search: $FOUND"
    fi
fi

if [ -z "$MODALITY_CONFIG" ]; then
    echo "ERROR: bridge_modality_config.py not found"
    echo "Source files under /opt/ml/code (excluding Isaac-GR00T):"
    find /opt/ml/code -not -path "*/Isaac-GR00T/*" -type f 2>/dev/null
    exit 1
fi

# Copy into Isaac-GR00T directory where launch_finetune.py runs
cp "$MODALITY_CONFIG" "$GROOT_DIR/bridge_modality_config.py"
MODALITY_CONFIG="$GROOT_DIR/bridge_modality_config.py"
echo "[INFO] Modality config copied to: $MODALITY_CONFIG"

mkdir -p "$OUTPUT_DIR" "$CHECKPOINT_DIR"

# --- Training parameters (from env or defaults) ---
NUM_GPUS=${SM_NUM_GPUS:-8}
MAX_STEPS=${MAX_STEPS:-2000}
BATCH_SIZE=${BATCH_SIZE:-32}
SAVE_STEPS=${SAVE_STEPS:-2000}
MODEL_PATH=${MODEL_PATH:-"nvidia/GR00T-N1.7-3B"}

echo ""
echo "Training config:"
echo "  GPUs: $NUM_GPUS"
echo "  Max steps: $MAX_STEPS"
echo "  Batch size: $BATCH_SIZE"
echo "  Dataset: $DATASET_PATH"
echo "  Output: $OUTPUT_DIR"
echo ""

# --- Fix: Move HF dataset cache to shared memory to avoid NCCL timeout at step 250 ---
# Without this, cache writes to disk cause I/O contention between GPU ranks
export HF_DATASETS_CACHE="/dev/shm/hf_cache"
mkdir -p /dev/shm/hf_cache
echo "[INFO] Dataset cache location: $HF_DATASETS_CACHE"

# --- Launch training ---
if [ "$NUM_GPUS" -gt 1 ]; then
    TRAIN_CMD="CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1))) python -m torch.distributed.run \
        --nproc_per_node=$NUM_GPUS --standalone \
        gr00t/experiment/launch_finetune.py"
else
    TRAIN_CMD="CUDA_VISIBLE_DEVICES=0 python gr00t/experiment/launch_finetune.py"
fi

eval $TRAIN_CMD \
    --base-model-path $MODEL_PATH \
    --dataset-path $DATASET_PATH \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path $MODALITY_CONFIG \
    --num-gpus $NUM_GPUS \
    --output-dir $OUTPUT_DIR \
    --save-total-limit 5 \
    --save-steps $SAVE_STEPS \
    --max-steps $MAX_STEPS \
    --global-batch-size $BATCH_SIZE \
    --no-tune-llm \
    --tune-visual \
    --tune-projector \
    --tune-diffusion-model \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader-num-workers 4

echo ""
echo "=== Training complete! ==="
echo "Model artifacts saved to: $OUTPUT_DIR"
echo "Checkpoints saved to: $CHECKPOINT_DIR"
