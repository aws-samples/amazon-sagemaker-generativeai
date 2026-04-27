#!/bin/bash
# Cosmos Transfer 2.5 — Async Evaluation Entrypoint for HyperPod EKS
#
# Runs on a separate eval node (e.g. g6e.48xlarge). Watches for new
# checkpoints, converts DCP → .pt, and runs inference + evaluation
# without blocking the training job.
#
# Environment variables:
#   OUTPUT_PATH        — FSx path where training writes checkpoints
#   DATASET_PATH       — FSx path to test dataset
#   EVAL_INTERVAL      — Seconds between checkpoint polls (default: 300)
#   NUM_GPUS           — GPUs on eval node (default: 8)

set -e

cd /workspace/cosmos-transfer2.5
export PATH="$(pwd)/.venv/bin:$PATH"
source .venv/bin/activate

NVIDIA_CUBLAS_LIB=$(find .venv -path "*/nvidia/cublas/lib" -type d 2>/dev/null | head -1)
export LD_LIBRARY_PATH="${NVIDIA_CUBLAS_LIB:+$NVIDIA_CUBLAS_LIB:}/usr/local/cuda/lib64"

OUTPUT_DIR="${OUTPUT_PATH:-/fsx/output/cosmos_transfer}"
TEST_DATASET="${DATASET_PATH:-/fsx/datasets/drive_dreams_multiview_test}"
EVAL_INTERVAL="${EVAL_INTERVAL:-300}"
NUM_GPUS="${NUM_GPUS:-8}"
EVAL_RESULTS_DIR="${OUTPUT_DIR}/eval_results"
CONVERTED_MARKER_DIR="${OUTPUT_DIR}/.eval_converted"

mkdir -p "$EVAL_RESULTS_DIR" "$CONVERTED_MARKER_DIR"

echo "============================================"
echo " Cosmos Transfer 2.5 — Async Evaluation"
echo " Watching: $OUTPUT_DIR"
echo " Test data: $TEST_DATASET"
echo " Poll interval: ${EVAL_INTERVAL}s"
echo "============================================"

# Find the checkpoints directory (training creates nested paths)
find_checkpoints_dir() {
    find "$OUTPUT_DIR" -type d -name "checkpoints" 2>/dev/null | head -1
}

# Convert a DCP checkpoint to .pt
convert_checkpoint() {
    local ckpt_dir="$1"
    echo "[EVAL] Converting $ckpt_dir ..."
    python scripts/convert_distcp_to_pt.py "$ckpt_dir/model" "$ckpt_dir"
    echo "[EVAL] Converted: $ckpt_dir/model_ema_bf16.pt"
}

# Run inference with a converted checkpoint
run_inference() {
    local ckpt_pt="$1"
    local iter_name="$2"
    local output_dir="$EVAL_RESULTS_DIR/inference_$iter_name"

    echo "[EVAL] Running inference for $iter_name ..."
    torchrun --nproc_per_node=$NUM_GPUS --master_port=12342 \
        -m examples.inference \
        -i "$TEST_DATASET/eval_specs.json" \
        -o "$output_dir" \
        --checkpoint-path "$ckpt_pt" \
        --disable-guardrails \
        2>&1 | tee "$EVAL_RESULTS_DIR/inference_${iter_name}.log"

    echo "[EVAL] Inference complete: $output_dir"
}

# Main polling loop
echo "[EVAL] Starting checkpoint watch loop..."
while true; do
    CKPT_BASE=$(find_checkpoints_dir)

    if [ -n "$CKPT_BASE" ] && [ -f "$CKPT_BASE/latest_checkpoint.txt" ]; then
        LATEST_ITER=$(cat "$CKPT_BASE/latest_checkpoint.txt")
        CKPT_DIR="$CKPT_BASE/$LATEST_ITER"
        MARKER="$CONVERTED_MARKER_DIR/$LATEST_ITER.done"

        if [ ! -f "$MARKER" ] && [ -d "$CKPT_DIR/model" ]; then
            echo ""
            echo "[EVAL] New checkpoint detected: $LATEST_ITER"

            # Convert
            convert_checkpoint "$CKPT_DIR"

            # Run inference if test dataset exists
            CKPT_PT="$CKPT_DIR/model_ema_bf16.pt"
            if [ -f "$CKPT_PT" ] && [ -d "$TEST_DATASET" ]; then
                run_inference "$CKPT_PT" "$LATEST_ITER"
            fi

            # Mark as processed
            touch "$MARKER"
            echo "[EVAL] Finished processing $LATEST_ITER"
        fi
    fi

    echo "[EVAL] Sleeping ${EVAL_INTERVAL}s ($(date '+%H:%M:%S'))..."
    sleep "$EVAL_INTERVAL"
done
