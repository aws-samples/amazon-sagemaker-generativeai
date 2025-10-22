#!/bin/bash

huggingface-cli login --token $HF_token

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected ${NUM_GPUS} GPUs on the machine"



# Launch fine-tuning with Accelerate + DeepSpeed (Zero3)
accelerate launch \
  --config_file accelerate_configs/deepspeed_zero3.yaml \
  --num_processes ${NUM_GPUS} \
  run_grpo.py \
  --config $CONFIG_PATH