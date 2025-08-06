#!/bin/bash
set -e

# Default values
NUM_GPUS=""
CONFIG_PATH=""

# Parse named arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --num_process)
      NUM_GPUS="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: ./accelerate_sagemaker_train.sh --num_process <NUM_GPUS> --config <CONFIG_YAML>"
      exit 1
      ;;
  esac
done

# Validate inputs
if [[ -z "$NUM_GPUS" || -z "$CONFIG_PATH" ]]; then
  echo "Error: --num_process and --config are required."
  echo "Usage: ./accelerate_sagemaker_train.sh --num_process <NUM_GPUS> --config <CONFIG_YAML>"
  exit 1
fi

# Install Python dependencies
echo "Installing Python packages from requirements.txt..."
python3 -m pip install -r ./requirements.txt

# Launch fine-tuning with Accelerate + DeepSpeed (Zero3)
accelerate launch \
  --config_file accelerate/zero3.yaml \
  --num_processes "$NUM_GPUS" \
  gpt_oss_sft.py \
  --config "$CONFIG_PATH"
