#!/bin/bash

huggingface-cli login --token $HF_token
aws s3 cp $data_location /opt/ml/input/data/dataset.hf --recursive

ls /opt/ml/input/data/

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected ${NUM_GPUS} GPUs on the machine"

accelerate launch --num_processes ${NUM_GPUS} --config_file accelerate_configs/deepspeed_zero3.yaml vlm_sft.py --config receipes/sft-vlm.yaml