#!/bin/bash

export ACCELERATE_CONFIG="./accelerate_configs/deepspeed_zero3.yaml"

# export training_recipe="./recipes/sft-spectrum-Qwen3-1.7B.yaml"
# export data_location="s3://sagemaker-us-east-1-340043819279/datasets/nvidia_function_calling/train/dataset.json"

echo "using ACCELERATE_CONFIG: $ACCELERATE_CONFIG"
echo "using Training Recipe: $training_recipe"

pip install yq

export MODEL_ID_OR_LOCATION=$(yq -r ".model_name_or_path" $training_recipe)
export MODEL_DOWNLOAD_LOCATION=$(yq -r ".model_download_location" $training_recipe)
export DATASET_LOCAL_LOCATION=$(yq -r ".dataset_local_location" $training_recipe)
export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true

if [ -n "${HF_token}" ]; then
    huggingface-cli login --token ${HF_token}
fi

#model_location env variable overrides MODEL_ID_OR_LOCATION
tmp_model_to_use=""

if [ -n "$model_location" ]; then
    tmp_model_to_use=$model_location
else
    tmp_model_to_use=$MODEL_ID_OR_LOCATION
fi

tmp_model_location="/opt/ml/tmp"

mkdir -p $MODEL_DOWNLOAD_LOCATION

# Check if the string ends with the suffix
if [[ "$tmp_model_to_use" == "s3:"* ]]; then
    echo "The model is an S3 location, downloading from '$tmp_model_to_use'"

    # Check if the string ends with the suffix
    if [[ "$tmp_model_to_use" == *".tar.gz" ]]; then
        echo "The model location '$tmp_model_to_use' ends with '.tar.gz'. Need to unpack."
        mkdir -p $tmp_model_location
        aws s3 cp $tmp_model_to_use $tmp_model_location
        tar -xzvf "$tmp_model_location/model.tar.gz" -C $MODEL_DOWNLOAD_LOCATION
    else
      echo "The model location '$tmp_model_to_use' looks to be unpacked, copying directly."
      aws s3 cp $tmp_model_to_use $tmp_model_location --recursive
    fi
else
  echo "The model does not look to be an an S3 location, downloading '$tmp_model_to_use' from HuggingFace"
  huggingface-cli download $tmp_model_to_use --local-dir $MODEL_DOWNLOAD_LOCATION
fi

aws s3 cp $data_location $DATASET_LOCAL_LOCATION

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected ${NUM_GPUS} GPUs on the machine"

accelerate launch --config_file $ACCELERATE_CONFIG --num_processes ${NUM_GPUS} run_sft.py --config $training_recipe