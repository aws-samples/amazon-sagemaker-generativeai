#!/bin/bash

git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .

# pip install git+https://github.com/huggingface/diffusers.git@v0.34.0

cd examples/dreambooth
pip install -r requirements_flux.txt
pip install deepspeed wandb prodigyopt


# Single GPU training
python3 /opt/ml/input/data/code/train_dreambooth_lora_flux.py \
            --pretrained_model_name_or_path black-forest-labs/FLUX.1-dev \
            --instance_data_dir /opt/ml/input/data/train \
            --output_dir /opt/ml/checkpoints \
            --mixed_precision bf16 \
            --instance_prompt 'a photo of sks dog' \
            --resolution 512 \
            --train_batch_size 1 \
            --guidance_scale 1 \
            --gradient_accumulation_steps 4 \
            --optimizer prodigy \
            --learning_rate 1. \
            --report_to wandb \
            --lr_scheduler constant \
            --lr_warmup_steps 0 \
            --max_train_steps 500 \
            --validation_prompt 'A photo of sks dog in a bucket' \
            --validation_epochs 25 \
            --checkpointing_steps 100 \
            --seed 0

# Multi GPU training
# accelerate launch --config_file /opt/ml/input/data/code/default_config.yaml \
#             /opt/ml/input/data/code/train_dreambooth_lora_flux.py \
#             --pretrained_model_name_or_path black-forest-labs/FLUX.1-dev \
#             --instance_data_dir /opt/ml/input/data/train \
#             --output_dir /opt/ml/checkpoints \
#             --mixed_precision bf16 \
#             --instance_prompt 'a photo of sks dog' \
#             --resolution 512 \
#             --train_batch_size 1 \
#             --guidance_scale 1 \
#             --gradient_accumulation_steps 4 \
#             --optimizer prodigy \
#             --learning_rate 1. \
#             --report_to wandb \
#             --lr_scheduler constant \
#             --lr_warmup_steps 0 \
#             --max_train_steps 500 \
#             --validation_prompt 'A photo of sks dog in a bucket' \
#             --validation_epochs 25 \
#             --checkpointing_steps 100 \
#             --seed 0