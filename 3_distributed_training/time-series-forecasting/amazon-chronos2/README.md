# Deploy and Fine-tune Amazon Chronos-2 on SageMaker

This repository provides end-to-end examples for deploying and fine-tuning [amazon/chronos-2](https://huggingface.co/amazon/chronos-2), a foundation model for time series forecasting.

## Overview

Chronos-2 is a foundation model for time series forecasting that builds on [Chronos](https://arxiv.org/abs/2403.07815) and Chronos-Bolt. It offers significant improvements in capabilities and can handle diverse forecasting scenarios not supported by earlier models.

**Technical Report:** [Chronos-2: From Univariate to Universal Forecasting](https://arxiv.org/abs/2510.15821v1)

### Capabilities Comparison

| Capability | Chronos | Chronos-Bolt | Chronos-2 |
|------------|---------|--------------|-----------|
| Univariate Forecasting | Yes | Yes | Yes |
| Cross-learning across items | No | No | Yes |
| Multivariate Forecasting | No | No | Yes |
| Past-only (real/categorical) covariates | No | No | Yes |
| Known future (real/categorical) covariates | Partial | Partial | Yes |
| Fine-tuning support | Yes | Yes | Yes |
| Max Context Length | 512 | 2048 | 8192 |

## Repository Structure

```
amazon-chronos2/
├── deploy-and-fine-tune-amazon-chronos2.ipynb  # Main notebook
└── sagemaker_code/
    ├── chronos2_finetune.py                    # Fine-tuning script
    ├── train.sh                                # Training entrypoint
    ├── requirements.txt                        # Training dependencies
    ├── recipes/
    │   └── amazon/
    │       ├── chronos-2--full.yaml            # Full fine-tuning config
    │       └── chronos-2--lora.yaml            # LoRA fine-tuning config
    └── djl_inference/
        ├── model.py                            # DJL inference handler
        ├── serving.properties                  # DJL serving config
        └── requirements.txt                    # Inference dependencies
```

## Prerequisites

- AWS Account with SageMaker access
- IAM role with permissions for SageMaker, S3, and (optionally) MLflow
- Python 3.10+
- GPU instance for training (recommended: `ml.g5.xlarge` or `ml.g6.2xlarge`)

## Quick Start

### 1. Deploy from SageMaker JumpStart

```python
from sagemaker.jumpstart.model import JumpStartModel

model = JumpStartModel(
    model_id="huggingface-forecasting-chronos2",
    role=role,
)

predictor = model.deploy(
    instance_type="ml.g5.xlarge",
    endpoint_name="chronos2-endpoint"
)
```

### 2. Run Inference

```python
sample_input = {
    "inputs": [{
        "target": [0.0, 4.0, 5.0, 1.5, -3.0, -5.0, -3.0, 1.5, 5.0, 4.0],
        "past_covariates": {
            "feat_1": [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 2.0, 0.0, 3.0, 1.0],
            "feat_2": ["A", "A", "A", "B", "B", "A", "B", "C", "A", "A"]
        },
        "future_covariates": {
            "feat_1": [0.5, 1.0, 1.0],
            "feat_2": ["C", "B", "C"]
        }
    }],
    "parameters": {
        "prediction_length": 3,
        "quantile_levels": [0.1, 0.5, 0.9]
    }
}

response = predictor.predict(sample_input)
```

## Fine-tuning

### Training Data Format

Prepare your data as JSONL with the following structure:

```json
{
    "target": [1.0, 2.0, 3.0, ...],
    "past_covariates": {
        "covariate_1": [0.1, 0.2, 0.3, ...],
        "covariate_2": [1, 0, 1, ...]
    },
    "future_covariates": {
        "known_covariate_1": null,
        "known_covariate_2": null
    }
}
```

Key points:
- `target`: Required. Array of target values to forecast.
- `past_covariates`: Optional. Dictionary of covariate arrays aligned with target. Include both past-only and known future covariates here with their historical values.
- `future_covariates`: Optional. Dictionary with keys only (values set to `null`). Indicates which covariates will be available at inference time.

### Training Configuration

Two fine-tuning modes are supported:

**Full Fine-tuning** (`chronos-2--full.yaml`):
```yaml
finetune_mode: full
learning_rate: 1.0e-5
num_steps: 5000
batch_size: 32
prediction_length: 24
```

**LoRA Fine-tuning** (`chronos-2--lora.yaml`):
```yaml
finetune_mode: lora
learning_rate: 1.0e-4
num_steps: 5000
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
```

### Launch Training Job

```python
from sagemaker.modules.train import ModelTrainer
from sagemaker.modules.configs import InputData, SourceCode, Compute

model_trainer = ModelTrainer(
    training_image=pytorch_image_uri,
    source_code=SourceCode(
        source_dir="./sagemaker_code",
        command="bash train.sh recipes/amazon/chronos-2--full.yaml"
    ),
    compute=Compute(
        instance_type="ml.g6.xlarge",
        instance_count=1
    ),
    role=role
)

model_trainer.train(
    input_data_config=[
        InputData(channel_name="training", data_source=s3_training_uri)
    ]
)
```

## Deploy Fine-tuned Model

The `sagemaker_code/djl_inference/` directory contains a custom DJL handler for deploying fine-tuned models.

### Handler Features

The `model.py` handler supports:
- Loading data from S3 URIs (parquet or CSV)
- Inline data as dictionaries or lists
- Configurable prediction parameters
- Multiple time series in a single request

### Deployment Steps

1. Update `serving.properties` with your model S3 path:
```properties
engine=Python
option.model_id=s3://your-bucket/path/to/finetuned-model/
option.dtype=fp32
option.task=custom
```

2. Package and deploy:
```python
from sagemaker.djl_inference import DJLModel

model = DJLModel(
    model_data=s3_code_uri,
    role=role,
    image_uri=djl_image_uri
)

predictor = model.deploy(
    instance_type="ml.g5.xlarge",
    endpoint_name="chronos2-finetuned-endpoint"
)
```

### Inference with S3 Data

```python
test_input = {
    "context_data": "s3://bucket/path/to/context.parquet",
    "future_data": "s3://bucket/path/to/future.parquet",
    "parameters": {
        "prediction_length": 24,
        "quantile_levels": [0.1, 0.5, 0.9],
        "id_column": "id",
        "timestamp_column": "timestamp",
        "target": "target"
    }
}

response = predictor.predict(test_input)
```

## Local Inference

For local development and testing:

```python
from chronos import BaseChronosPipeline, Chronos2Pipeline

pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cuda"
)

pred_df = pipeline.predict_df(
    context_df,
    future_df=future_df,
    prediction_length=24,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column="id",
    timestamp_column="timestamp",
    target="target"
)
```

## MLflow Integration

Training metrics are logged to MLflow when configured:

```python
training_env = {
    "MLFLOW_EXPERIMENT_NAME": "chronos2-finetuning",
    "MLFLOW_TRACKING_URI": mlflow_tracking_server_arn,
    "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING": "true"
}
```

Logged metrics include:
- Training loss per step
- Validation loss (if validation split > 0)
- Training duration
- Model hyperparameters

## Instance Recommendations

| Task | Recommended Instance |
|------|---------------------|
| Notebook execution | `ml.c5.2xlarge` or `ml.g4dn.xlarge` |
| Fine-tuning | `ml.g5.xlarge` or `ml.g6.xlarge` |
| Inference endpoint | `ml.g5.xlarge` |

## References

- [Chronos-2 on Hugging Face](https://huggingface.co/amazon/chronos-2)
- [Technical Report: Chronos-2](https://arxiv.org/abs/2510.15821v1)
- [Original Chronos Paper](https://arxiv.org/abs/2403.07815)
- [Chronos GitHub Repository](https://github.com/amazon-science/chronos-forecasting)

## License

This project is licensed under the MIT License. See the LICENSE file for details.
