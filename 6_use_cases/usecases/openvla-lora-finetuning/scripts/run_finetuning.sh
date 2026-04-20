#!/bin/bash
# OpenVLA — SageMaker Training Job Entry Point
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
echo " OpenVLA — SageMaker Training Job"
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
apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 ffmpeg 2>/dev/null || true

# --- Install additional Python deps ---
echo "=== Installing Python dependencies ==="
pip install --no-cache-dir peft>=0.14.0 bitsandbytes accelerate>=1.0.0 2>/dev/null || true

# --- Resolve paths ---
DATASET_PATH="/opt/ml/input/data/train"
OUTPUT_DIR="/opt/ml/model/openvla_lora_finetune"
CHECKPOINT_DIR="/opt/ml/checkpoints"

# Find openvla_utils.py — SageMaker may place source files in different locations
UTILS_FILE=""
for candidate in \
    "/opt/ml/code/utils/openvla_utils.py" \
    "/opt/ml/code/openvla_utils.py" \
    "/opt/ml/code/scripts/utils/openvla_utils.py" \
    "/opt/ml/code/scripts/openvla_utils.py" \
    "$(dirname "$0")/utils/openvla_utils.py" \
    "$(dirname "$0")/openvla_utils.py"; do
    if [ -f "$candidate" ]; then
        UTILS_FILE="$candidate"
        echo "[INFO] Found openvla_utils.py at: $candidate"
        break
    fi
done

if [ -z "$UTILS_FILE" ]; then
    FOUND=$(find /opt/ml/code -name "openvla_utils.py" -type f 2>/dev/null | head -1)
    if [ -n "$FOUND" ]; then
        UTILS_FILE="$FOUND"
        echo "[INFO] Found openvla_utils.py via search: $FOUND"
    fi
fi

if [ -z "$UTILS_FILE" ]; then
    echo "ERROR: openvla_utils.py not found anywhere under /opt/ml/code/"
    find /opt/ml/code -type f -name "*.py" 2>/dev/null
    exit 1
fi

# Find train_openvla_lora.py
TRAIN_SCRIPT=""
for candidate in \
    "/opt/ml/code/train_openvla_lora.py" \
    "/opt/ml/code/scripts/train_openvla_lora.py" \
    "$(dirname "$0")/train_openvla_lora.py"; do
    if [ -f "$candidate" ]; then
        TRAIN_SCRIPT="$candidate"
        echo "[INFO] Found train_openvla_lora.py at: $candidate"
        break
    fi
done

if [ -z "$TRAIN_SCRIPT" ]; then
    FOUND=$(find /opt/ml/code -name "train_openvla_lora.py" -type f 2>/dev/null | head -1)
    if [ -n "$FOUND" ]; then
        TRAIN_SCRIPT="$FOUND"
        echo "[INFO] Found train_openvla_lora.py via search: $FOUND"
    fi
fi

if [ -z "$TRAIN_SCRIPT" ]; then
    echo "ERROR: train_openvla_lora.py not found anywhere under /opt/ml/code/"
    find /opt/ml/code -type f -name "*.py" 2>/dev/null
    exit 1
fi

# Ensure openvla_utils.py is in a utils/ subfolder next to the training script
TRAIN_DIR=$(dirname "$TRAIN_SCRIPT")
mkdir -p "$TRAIN_DIR/utils"
if [ "$UTILS_FILE" != "$TRAIN_DIR/utils/openvla_utils.py" ]; then
    cp "$UTILS_FILE" "$TRAIN_DIR/utils/openvla_utils.py"
    echo "[INFO] Copied openvla_utils.py to $TRAIN_DIR/utils/"
fi
# Ensure utils/__init__.py exists
if [ ! -f "$TRAIN_DIR/utils/__init__.py" ]; then
    echo "from .openvla_utils import ActionTokenizer, PurePromptBuilder, VicunaV15ChatPromptBuilder" > "$TRAIN_DIR/utils/__init__.py"
    echo "[INFO] Created $TRAIN_DIR/utils/__init__.py"
fi

mkdir -p "$OUTPUT_DIR" "$CHECKPOINT_DIR"

# --- Disable wandb (avoid login errors in SageMaker containers) ---
export WANDB_MODE=disabled
export WANDB_DISABLED=true

# --- Training parameters (from env or defaults) ---
NUM_GPUS=${SM_NUM_GPUS:-8}
MAX_STEPS=${MAX_STEPS:-5000}
BATCH_SIZE=${BATCH_SIZE:-4}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
LORA_RANK=${LORA_RANK:-16}
SAVE_STEPS=${SAVE_STEPS:-500}
MODEL_PATH=${MODEL_PATH:-"openvla/openvla-7b"}

echo ""
echo "Training config:"
echo "  GPUs: $NUM_GPUS"
echo "  Max steps: $MAX_STEPS"
echo "  Batch size: $BATCH_SIZE (per device)"
echo "  Learning rate: $LEARNING_RATE"
echo "  LoRA rank: $LORA_RANK"
echo "  Model: $MODEL_PATH"
echo "  Dataset: $DATASET_PATH"
echo "  Output: $OUTPUT_DIR"
echo ""

# --- Launch training with Accelerate ---

# Generate accelerate config for multi-GPU
ACCELERATE_CONFIG="/tmp/accelerate_config.yaml"
cat > "$ACCELERATE_CONFIG" << EOF
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_machines: 1
num_processes: $NUM_GPUS
mixed_precision: bf16
use_cpu: false
EOF

echo "[INFO] Accelerate config:"
cat "$ACCELERATE_CONFIG"
echo ""

TRAIN_CMD="accelerate launch --config_file $ACCELERATE_CONFIG $TRAIN_SCRIPT \
    --pretrained_model_name_or_path $MODEL_PATH \
    --train_data_dir $DATASET_PATH \
    --output_dir $OUTPUT_DIR \
    --max_train_steps $MAX_STEPS \
    --train_batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --rank $LORA_RANK \
    --lora_target_modules q_proj,v_proj,k_proj,out_proj \
    --lr_scheduler cosine \
    --lr_warmup_steps 500 \
    --gradient_checkpointing \
    --mixed_precision bf16 \
    --allow_tf32 \
    --checkpointing_steps $SAVE_STEPS \
    --checkpoints_total_limit 3 \
    --dataloader_num_workers 4 \
    --report_to tensorboard \
    --num_validation_samples 0 \
    --resolution 224 \
    --random_flip \
    --action_dim 7"

echo "=== Launching training ==="
echo "Command: $TRAIN_CMD"
echo ""

eval $TRAIN_CMD

echo ""
echo "=== Training complete! ==="
echo "Model artifacts saved to: $OUTPUT_DIR"

# Copy final model to /opt/ml/model for SageMaker to pick up
if [ -d "$OUTPUT_DIR" ] && [ "$OUTPUT_DIR" != "/opt/ml/model" ]; then
    echo "Copying model artifacts to /opt/ml/model/..."
    cp -r "$OUTPUT_DIR"/* /opt/ml/model/ 2>/dev/null || true
fi

echo "Done."
