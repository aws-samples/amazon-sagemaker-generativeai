#!/bin/bash

huggingface-cli login --token $HF_token

aws s3 cp $train_data_location /opt/ml/input/data/dataset/

aws s3 cp $validation_data_location /opt/ml/input/data/dataset/

#aws s3 cp $test_data_location /opt/ml/input/data/dataset/

accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml --num_processes 8 run_training.py --config receipes/google-vit-base-patch16-224.yaml
# accelerate launch --config_file accelerate_configs/deepspeed_zero1.yaml --num_processes 8 scripts/run_training.py --config receipes/google-vit-base-patch16-224.yaml
