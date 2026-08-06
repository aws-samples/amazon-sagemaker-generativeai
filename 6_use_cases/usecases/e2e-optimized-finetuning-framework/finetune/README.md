# AWS SageMaker LLM Fine-Tuning Framework 🤖

A comprehensive framework for fine-tuning Large Language Models (LLMs) using Parameter-Efficient Fine-Tuning (PEFT) methods on AWS SageMaker, with optimized deployment using FasterTransformer.

## Table of Contents

- [1.0 Key Features and Services](#10-key-features-and-services)
- [2.0 Architecture](#20-architecture)
  - [2.1 Overall Workflow](#21-overall-workflow)
- [3.0 Getting Started](#30-getting-started)
  - [3.1 Prerequisites](#31-prerequisites)
  - [3.2 Environment Setup](#32-environment-setup)
  - [3.3 Package Installation](#33-package-installation)
- [4.0 Configuration](#40-configuration)
  - [4.1 AWS Credentials](#41-aws-credentials)
  - [4.2 Training Configuration](#42-training-configuration)
  - [4.3 Data Configuration](#43-data-configuration)
  - [4.4 Deployment Configuration](#44-deployment-configuration)
- [5.0 Running the Application](#50-running-the-application)
  - [5.1 Training](#51-training)
  - [5.2 Deployment](#52-deployment)
  - [5.3 Training and Deployment](#53-training-and-deployment)
- [6.0 Troubleshooting](#60-troubleshooting)
- [7.0 Cleanup](#70-cleanup)

## 1.0 Key Features and Services

### AWS Services
- **Amazon SageMaker** - Managed training and deployment infrastructure
- **Amazon S3** - Data storage and model artifacts
- **Multi-region support** - Deploy across AWS regions

### Fine-Tuning Methods
- **IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations)** - Efficient parameter scaling
- **LoRA (Low-Rank Adaptation)** - Low-rank matrix decomposition
- **P-Tuning** - Prompt-based tuning with learnable embeddings
- **Prompt Tuning** - Soft prompt optimization

### Evaluation Metrics
- **ROUGE** - Text summarization quality
- **BLEU** - Translation and generation quality
- **BERTScore** - Semantic similarity evaluation

### Features
- Configuration-driven approach with YAML files
- Automatic data quality checking
- Model training with 8-bit quantization support
- Optimized inference
- Multi-region deployment capabilities

## 2.0 Architecture

### 2.1 Overall Workflow

#### Phase 1: Configuration & Setup
```
User updates YAML configs → System validates configs → AWS credentials loaded → 
Environment variables set → AWS services connection established (SageMaker, S3)
```

**Key Components:**
- Configuration validation via `utils/config_util.py`
- AWS service initialization via `services/awsmanager_service.py`
- Environment setup with tokenizer parallelism settings

#### Phase 2: Data Preparation
```
CSV data loaded from S3 → Data quality check (optional) → 
Train/test split (85/15) → Data preprocessing & tokenization → 
Upload processed data to S3
```

**Key Components:**
- Data loading: `utils/data_loading_util.py`
- Quality checks: `utils/training_data_quality_check.py`
- Data preprocessing: `utils/training_utils/data_prep.py`
- Configurable train/test split ratio

#### Phase 3: Model Training
```
HuggingFace base model (FLAN-T5) loaded → 
PEFT configuration applied (IA3/LoRA/P-Tuning/Prompt Tuning) → 
SageMaker training job launched on GPU instance → 
Model trained with specified hyperparameters → 
Training artifacts saved to S3 → 
Evaluation metrics computed (ROUGE, BLEU, BERTScore)
```

**Key Components:**
- Model preparation: `utils/training_utils/model_prep.py`
- PEFT configuration: Defined in `configs/training.yaml`
- Training orchestration: `training_thread.py`
- Optimizer setup: `utils/training_utils/optimizer_prep.py`
- Training execution: `utils/training_utils/training.py`
- Evaluation: `utils/training_utils/evaluation.py`

#### Phase 4: Model Deployment
```
Trained model retrieved from S3 → 
FasterTransformer optimization applied → 
Model configuration prepared → 
SageMaker endpoint created → 
Model loaded on inference instance → 
Endpoint ready for predictions
```

**Key Components:**
- Deployment orchestration: `launch_deployment.py`
- FasterTransformer integration: `fastertransformer-t5/model.py`
- Configuration preparation: `utils/deployment_utils/config_prep.py`
- Deployment utilities: `utils/deployment_utils/deploy_fastertransformer.py`

#### Phase 5: Inference
```
User sends text input → Endpoint processes request → 
Model generates output → Response returned to user
```

**Key Components:**
- Inference script: `inference.py`
- Interactive notebook: `inference.ipynb`
- Generation parameters configured in `configs/generation.yaml`

> **Note:** You can add an architecture diagram here to visualize the workflow.

## 3.0 Getting Started

### 3.1 Prerequisites

Before using this framework, ensure you have:

1. **AWS Account** with appropriate permissions:
   - `sagemaker:*` - Full SageMaker access
   - `s3:*` - Full S3 access
   - `iam:PassRole` - Role passing for SageMaker

2. **AWS CLI** installed and configured:
   - Follow the [AWS CLI Getting Started Guide](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html)
   - Configure your AWS credentials using `aws configure`

3. **Python Environment**:
   - Python 3.10 or higher
   - CUDA-compatible GPU (for local testing, optional)

4. **SageMaker Studio or Notebook Instance** (Recommended):
   - Attach `AmazonSageMakerFullAccess` policy
   - Attach `AmazonS3FullAccess` policy

### 3.2 Environment Setup

> **Python 3.10+ is required.** A `venv` inherits the Python version of the interpreter that creates it and cannot upgrade it, so `python -m venv` must be run with a 3.10+ interpreter (e.g. via `pyenv` or the python.org installer). macOS system Python is often 3.9 — check with `python --version` first.

#### Option 1: Virtual Environment (venv)

```bash
# Create virtual environment (use a Python 3.10+ interpreter)
python -m venv finetune-env

# Activate environment
# On macOS/Linux:
source finetune-env/bin/activate
# On Windows:
finetune-env\Scripts\activate
```

#### Option 2: Conda Environment (Recommended)

```bash
# Create conda environment
conda create --name finetune-env python=3.10 -y

# Activate environment
conda activate finetune-env
```

### 3.3 Package Installation

#### Install Required Packages

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Install specific CUDA version for bitsandbytes
export BNB_CUDA_VERSION=121
pip install triton==2.0.0
```

> **macOS / non-GPU note:** `bitsandbytes` ships no macOS (arm64) wheel — it is CUDA-only and used for 8-bit quantization on the remote SageMaker GPU instance, not locally. If `pip install -r requirements.txt` fails on `bitsandbytes`, install the other packages and skip it; local orchestration (launching training/deployment) does not need it.

## 4.0 Configuration

The framework uses YAML configuration files located in the `configs/` directory. All configurations must be properly set before running training or deployment.

### 4.1 AWS Credentials

**File:** `configs/aws_setup.yaml`

```yaml
region_name: us-east-1       # NOTE: the key is region_name, not region
default_bucket: finetune-blog
# profile_name: finetune     # Optional: AWS CLI profile; omit to use the default profile
# role: arn:aws:iam::ACCOUNT_ID:role/ROLE_NAME  # See role note below
```

**Parameters:**
- `region_name`: AWS region for SageMaker training and deployment.
- `default_bucket`: S3 bucket for storing data and model artifacts.
- `profile_name`: AWS CLI profile. If omitted, the **default** profile is used.
- `role`: SageMaker execution role (trust policy must allow `sagemaker.amazonaws.com`). Set via `--role`, this key, or `SAGEMAKER_ROLE`; auto-detected inside SageMaker.

> **Important:** Ensure your S3 bucket name is globally unique and the bucket exists (or can be created) in the specified region.

### 4.2 Training Configuration

**File:** `configs/training.yaml`

```yaml
---
train_job_name: ~  # Auto-generated if not specified
training_instance: 'ml.g5.12xlarge'
train_script: training_thread.py
    
model_id: 'google/flan-t5-large'        
training_type: ia3  # Options: ia3, lora, p_tune, prompt_tuning

label_pad_token_id: -100

data_params:
    max_ip_len: 600
    max_op_len: 400
    train_data_frac: 1
    tr_batch_size: 2
    eval_batch_size: 2

task_prefix: | 
             Generate the text:
```

**Key Parameters:**

| Parameter | Description | Options/Examples |
|-----------|-------------|------------------|
| `training_instance` | SageMaker instance type | `ml.g5.12xlarge`, `ml.p3.8xlarge` |
| `model_id` | Base model from HuggingFace | `google/flan-t5-large`, `google/flan-t5-xl` |
| `training_type` | PEFT method | `ia3`, `lora`, `p_tune`, `prompt_tuning` |
| `max_ip_len` | Maximum input sequence length | 600 tokens |
| `max_op_len` | Maximum output sequence length | 400 tokens |
| `tr_batch_size` | Training batch size | 2 (adjust based on GPU memory) |
| `eval_batch_size` | Evaluation batch size | 2 |

**PEFT Method Configurations:**

**IA3 (Default):**
```yaml
ia3:
    target_modules: ["q", "v", "k", "o", "wi_1", "wi_0", "wo"]
```
q -> 
v -> 

**LoRA:**
```yaml
lora:
    r: 4
    lora_alpha: 32
    lora_dropout: 0.1
    target_modules: ["q"]
    bias: none
```

**P-Tuning:**
```yaml
p_tune:
    num_virtual_tokens: 20
    encoder_hidden_size: 512
```

**Prompt Tuning:**
```yaml
prompt_tuning:
    num_virtual_tokens: 10
    prompt_tuning_init: TEXT
    prompt_tuning_init_text: | 
                             Summarize this info:
```

**Training Parameters:**
```yaml
training_params:
    num_epochs: 1
    lr: 1.0e-2
    wt_decay: 1.0e-8
    
metric_type: 'lexical'  # Options: lexical, semantic
```

### 4.3 Data Configuration

**File:** `configs/data.yaml`

```yaml
update_input_data: True
s3_root_folder: 'finetune-blog'
s3_input_data_folder: 'data'
s3_input_data_file: 'synthetic_training_set.csv'
s3_out_train_artifact_path: 's3://your-bucket/finetune-blog/results/'
test_size: 0.15
```

**Parameters:**
- `update_input_data`: Set to `True` to regenerate and upload training data
- `s3_root_folder`: Root folder in S3 bucket for project files
- `s3_input_data_file`: Name of CSV file containing training data
- `test_size`: Fraction of data for validation (0.15 = 15%)

**Data Format:**
Your CSV file should contain at least two columns:
- Input text column (source)
- Output text column (target)

### 4.4 Deployment Configuration

**File:** `configs/faster_transformer.yaml`

```yaml
# FasterTransformer configuration for optimized inference
# Add your deployment-specific parameters here
```

**File:** `configs/generation.yaml`

```yaml
# Generation parameters for inference
# Configure parameters like max_length, temperature, etc.
```

## 5.0 Running the Application

The framework supports three main operations through the `main.py` script.

> **Run from the framework root, as a module.** The code uses absolute `finetune.*` imports, so run it from the directory that *contains* the `finetune/` package (i.e. `e2e-optimized-finetuning-framework/`), not from inside `finetune/`. Use `python -m finetune.main ...`. Running `python main.py` from inside `finetune/` fails with `ModuleNotFoundError: No module named 'finetune'`.

> **Credentials & role flags.** When running outside SageMaker, pass your authenticated profile and a SageMaker execution role, e.g. `--profile finetune --role arn:aws:iam::<ACCOUNT_ID>:role/<sagemaker-execution-role>`.

### 5.1 Training

Train a model using Parameter-Efficient Fine-Tuning:

```bash
python -m finetune.main --train \
    --profile <your-profile> \
    --role arn:aws:iam::<ACCOUNT_ID>:role/<sagemaker-execution-role>
```

**Optional: Data Quality Check**
```bash
python -m finetune.main --train --train_dataset_quality_check
```

This will:
1. Load and validate configurations
2. Connect to AWS services (SageMaker, S3)
3. Prepare and upload training data to S3
4. Launch SageMaker training job
5. Save trained model artifacts to S3
6. Compute evaluation metrics

**Training Output:**
- Model artifacts saved to S3 path specified in `configs/data.yaml`
- Training job name (needed for deployment): `model-name-fine-tune-method-YYYY-MM-DD-HH-MM-SS-SSS`
- Logs available in SageMaker console

### 5.2 Deployment

Deploy a pre-trained model to a SageMaker endpoint:

```bash
python -m finetune.main --deploy \
    --model_name model-name-fine-tune-method-YYYY-MM-DD-HH-MM-SS-SSS \
    --endpoint_name model-name-fine-tune-method-YYYY-MM-DD-HH-MM-SS-SSS-endpoint \
    --profile <your-profile> \
    --role arn:aws:iam::<ACCOUNT_ID>:role/<sagemaker-execution-role>
```

**Parameters:**
- `--model_name`: Name of the trained model (training job name)
- `--endpoint_name`: Name for the SageMaker endpoint (optional, auto-generated if not provided)
- `--s3_folder_existance`: Add this flag if model artifacts already exist in S3
- `--profile` / `--role`: Same as training — pass your authenticated profile and a SageMaker execution role when running outside SageMaker. Omitting `--profile` falls back to the (possibly expired) default profile and surfaces as `HeadBucket ... 400 Bad Request` (see Issue 2b).

**Deployment from Existing S3 Model:**
```bash
python -m finetune.main --deploy \
    --model_name model-name-fine-tune-method-YYYY-MM-DD-HH-MM-SS-SSS \
    --endpoint_name my-inference-endpoint \
    --s3_folder_existance \
    --profile <your-profile> \
    --role arn:aws:iam::<ACCOUNT_ID>:role/<sagemaker-execution-role>
```

**Testing the Endpoint:**

Invoke the deployed endpoint directly with the AWS CLI (replace `finetune` with your profile name). The endpoint is in `us-west-2`, `text` is a **list**, and `parameters` uses `temperature` / `max_seq_len`:

```bash
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name <your-endpoint-name> \
  --region us-west-2 \
  --profile <your-profile> \
  --cli-binary-format raw-in-base64-out \
  --content-type application/json \
  --body '{"text": ["<your-input-text>"], "parameters": {"temperature": 0.1, "max_seq_len": 128}}' \
  /dev/stdout
```

> - `--cli-binary-format raw-in-base64-out` is **required** on AWS CLI v2. Without it, `--body` is base64-decoded before sending and the container fails with `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xb5 ... (424 prediction failure)`.
> - Wrap the whole `--body` value in single quotes so the shell treats `&&` inside the payload literally instead of as command separators.

### 5.3 Training and Deployment

Train and immediately deploy the model:

```bash
python -m finetune.main --train --deploy
```

This will:
1. Complete the training process
2. Automatically deploy the trained model to a SageMaker endpoint
3. Endpoint name will be: `{training_job_name}-endpoint`

## 6.0 Troubleshooting

### Common Issues and Solutions

#### Issue 1: Out of Memory During Training

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce batch size in `configs/training.yaml`:
```yaml
data_params:
    tr_batch_size: 1
    eval_batch_size: 1
```

2. Enable 8-bit quantization:
```yaml
load_in_8bit:
    True: True
```

3. Use a larger instance type:
```yaml
training_instance: 'ml.g5.24xlarge'
```

#### Issue 2: AWS Credentials Not Found

**Symptoms:**
```
botocore.exceptions.NoCredentialsError: Unable to locate credentials
```

**Solutions:**
1. Configure AWS CLI:
```bash
aws configure
```

2. Set environment variables:
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

3. Set `profile_name` in `configs/aws_setup.yaml` (or pass `--profile`)

#### Issue 2b: Expired Token / Wrong Profile

**Symptoms:**
```
An error occurred (ExpiredToken) ... The security token included in the request is expired
An error occurred (400) when calling the HeadBucket operation: Bad Request
```

**Solutions:**
1. Refresh the credentials for the profile you intend to use, then verify **that same profile**:
```bash
aws sts get-caller-identity --profile <your-profile>
```
2. A bare `aws sts get-caller-identity` (no `--profile`) checks the **default** profile, which may be a different/expired set of credentials than the one you refreshed. Make sure the app targets the refreshed profile via `--profile` or `profile_name` in `aws_setup.yaml`.

#### Issue 3: SageMaker Training Job Fails

**Symptoms:**
- Training job shows "Failed" status in SageMaker console

**Solutions:**
1. Check CloudWatch logs for detailed error messages
2. Verify IAM role has necessary permissions:
   - `AmazonSageMakerFullAccess`
   - `AmazonS3FullAccess`
3. Ensure instance type is available in your region
4. Check S3 bucket permissions

#### Issue 4: Model Not Found in S3

**Symptoms:**
```
ClientError: The specified key does not exist
```

**Solutions:**
1. Verify training job completed successfully
2. Check S3 path in `configs/data.yaml`
3. Wait for model artifacts to be uploaded (may take a few minutes)
4. Use `--s3_folder_existance` flag only if artifacts exist

## 7.0 Cleanup

To avoid ongoing AWS charges, delete resources after use:

### Delete SageMaker Endpoint

```python
import boto3

sagemaker_client = boto3.client('sagemaker', region_name='us-east-1')

# Delete endpoint
sagemaker_client.delete_endpoint(
    EndpointName='your-endpoint-name'
)

# Delete endpoint configuration
sagemaker_client.delete_endpoint_config(
    EndpointConfigName='your-endpoint-config-name'
)

# Delete model
sagemaker_client.delete_model(
    ModelName='your-model-name'
)
```

### Delete S3 Objects

```bash
# Delete specific folder
aws s3 rm s3://your-bucket/finetune-blog/ --recursive

# Or delete entire bucket
aws s3 rb s3://your-bucket --force
```

### Stop SageMaker Notebook Instance

If using SageMaker Notebook:
```bash
aws sagemaker stop-notebook-instance --notebook-instance-name your-instance-name
```

---

## Additional Resources

- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [PEFT Library Documentation](https://huggingface.co/docs/peft/)
- [Parameter-Efficient Fine-Tuning Methods](https://huggingface.co/blog/peft)
