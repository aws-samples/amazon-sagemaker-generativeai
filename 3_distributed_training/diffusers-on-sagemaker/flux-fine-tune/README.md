# Fine-tune Flux with DreamBooth LoRA Hugging Face Diffusers

Prerequisites:
- AWS account with SageMaker access
     - Make sure you have the service quota for a p4de or p5 instance for SageMaker Training Job
- Hugging Face account with API token
- Weights & Biases account with API key

1. git clone this repository
``` bash
git clone https://github.com/black-forest-labs/flux-fine-tune.git
cd flux-fine-tune
```
2. Install uv and restart your shell
``` bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
3. Create a virtual environment
``` bash
uv venv --prompt flux --python 3.12
source .venv/bin/activate
```
4. Install dependencies
``` bash
uv pip install -r requirements.txt
```
5. Add your hugging face and wandb API keys to the env-example file and rename it to `.env`

5. Run the flux-fine-tune-sagemaker.ipynb notebook in Jupyter Lab or Jupyter Notebook.



``` bash
export MODEL_NAME="black-forest-labs/FLUX.1-dev"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="trained-flux-lora-081025-0735"

accelerate launch --config_file /home/ubuntu/flux-fine-tune/default_config.yaml train_dreambooth_lora_flux.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --guidance_scale=1 \
  --gradient_accumulation_steps=4 \
  --optimizer="prodigy" \
  --learning_rate=1. \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=25 \
  --seed="0" > train.log 2>&1 &
  ```