# Fine-Tuning Qwen2.5 7B Instruct with Unsloth on SageMaker

This repository contains everything needed to fine-tune the `unsloth/Qwen2.5-7B-Instruct` model using the Unsloth library and Amazon SageMaker with a custom Docker container.

## Project Structure

```
.
├── Dockerfile
├── entrypoint.sh
├── requirements.txt
├── sagemaker_unsloth_qwen2_5_train.py
├── launch_unsloth_sagemaker_training.py
└── data/
    └── alpaca_autotune_datagen_with_cot_4000.json
```

## Setup Instructions

### 1. Build and Push Docker Image to ECR

```bash
docker build -t unsloth-train .
aws ecr get-login-password --region <your-region> | docker login --username AWS --password-stdin <your-ecr-repo>
docker tag unsloth-train:latest <your-ecr-repo>:latest
docker push <your-ecr-repo>:latest
```

## Fine-Tuning Script

The training script `sagemaker_unsloth_qwen2_5_train.py`:

- Loads Alpaca-style data from a JSON file
- Applies formatting using the Alpaca prompt template
- Uses `SFTTrainer` from the `trl` library for fine-tuning
- Saves the model to `/opt/ml/model`

## Launch the Training Job

```bash
python launch_unsloth_sagemaker_training.py
```

This script:

- Uploads the training script and dataset to S3
- Configures a SageMaker training job with your custom Docker image
- Specifies hyperparameters and triggers model training

## IAM Role Requirements

Ensure your SageMaker execution role includes permissions for:

- Amazon S3 (read/write)
- Amazon SageMaker
- Amazon ECR

Example policy:

```json
{
  "Effect": "Allow",
  "Action": [
    "s3:*",
    "sagemaker:*",
    "ecr:*"
  ],
  "Resource": "*"
}
```

## Output

Model artifacts will be saved to `/opt/ml/model` and uploaded to your S3 bucket by SageMaker.

## Cleanup

To avoid charges, remember to:

- Stop SageMaker jobs when finished
- Delete unused Docker images and models
- Clean up S3 objects you no longer need

## References

- https://github.com/unslothai/unsloth
- https://huggingface.co/unsloth
- https://docs.aws.amazon.com/sagemaker/

