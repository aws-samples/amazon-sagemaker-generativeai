#!/usr/bin/env bash
# measure_all.sh
# Iterates over quantized model directories and runs GPU utilization measurement script

set -euo pipefail

# Usage
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <BASE_MODEL_DIR> <OUTPUT_PATH> <DEVICE_MAP>"
  echo "Example: $0 /path/to/models /path/to/output auto"
  exit 1
fi

# Input arguments
BASE_MODEL_DIR="$1"
OUTPUT_PATH="$2"
DEVICE_MAP="$3"
SCRIPT="measure_gpu_utilization.py"

# Ensure output directory exists
mkdir -p "$OUTPUT_PATH"

# Iterate through each subdirectory in BASE_MODEL_DIR
for SUBDIR in "$BASE_MODEL_DIR"/*; do
  if [ -d "$SUBDIR" ]; then
    MODEL_NAME=$(basename "$SUBDIR")
    FULL_MODEL_PATH="$BASE_MODEL_DIR/$MODEL_NAME"
    
    echo "============================================================"
    echo "Measuring GPU utilization for model: $MODEL_NAME"
    echo "------------------------------------------------------------"

    python3 "$SCRIPT" \
      --quantized-model-path "$FULL_MODEL_PATH" \
      --output-path "$OUTPUT_PATH" \
      --device-map "$DEVICE_MAP"

    echo
  fi
done
