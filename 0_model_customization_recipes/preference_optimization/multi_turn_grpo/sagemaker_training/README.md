# SageMaker Multi-Turn GRPO Training

Train Multi-Turn GRPO models using AWS SageMaker Training Jobs with managed infrastructure.

## 📋 Overview

This directory contains everything needed to launch Multi-Turn GRPO training jobs on AWS SageMaker. SageMaker provides managed infrastructure, automatic scaling, and integrated monitoring.

## 🚀 Quick Start

### Using Jupyter Notebook (Recommended)

```bash
jupyter notebook notebooks/launch_training_job.ipynb
```

The notebook will guide you through:
1. Setting up AWS credentials and roles
2. Configuring training parameters
3. Launching the training job
4. Monitoring progress

### Using Python Script

```python
import sagemaker
from sagemaker.pytorch import PyTorch

# Configure training job
estimator = PyTorch(
    entry_point='train.py',
    source_dir='.',
    role='<your-sagemaker-role>',
    instance_type='ml.p4d.24xlarge',
    instance_count=1,
    framework_version='2.0.0',
    py_version='py310',
    hyperparameters={
        'config_file': 'hf_recipes/Qwen/Qwen3-1.7B--mt-grpo.yaml',
        'epochs': 15,
    }
)

# Launch training
estimator.fit()
```

## 📁 Directory Structure

```
sagemaker_training/
├── notebooks/
│   └── launch_training_job.ipynb  # Job launcher notebook
├── scripts/
│   └── sm_mt_grpo_train.sh        # Training script for SageMaker
├── configs/
│   └── accelerate/                # Accelerate configurations
├── hf_recipes/
│   └── Qwen/                      # Model-specific recipes
├── rewards/
│   └── triviaqa_reward.py         # Reward function
├── tools_funcs/
│   └── wiki_search.py             # Wikipedia search tool
├── mt_grpo_trainer.py             # Core trainer implementation
├── train.py                       # SageMaker entry point
├── requirements.txt               # Python dependencies
├── __init__.py                    # Package initialization
└── README.md                      # This file
```

## 🔧 Prerequisites

### AWS Setup
- AWS Account with SageMaker access
- IAM role with SageMaker permissions:
  - `AmazonSageMakerFullAccess`
  - S3 read/write access
  - ECR access (for custom containers)
- S3 bucket for training artifacts

### Local Setup
- Python 3.10+
- AWS CLI configured
- SageMaker Python SDK: `pip install sagemaker`

## 🎯 Training Configuration

### Instance Types

Recommended instance types for different model sizes:

| Model Size | Instance Type | GPUs | Memory | Cost/Hour* |
|------------|---------------|------|--------|------------|
| 1-3B | ml.g5.2xlarge | 1x A10G | 24GB | ~$1.50 |
| 3-7B | ml.g5.12xlarge | 4x A10G | 96GB | ~$7.00 |
| 7-13B | ml.p4d.24xlarge | 8x A100 | 320GB | ~$32.00 |
| 13B+ | ml.p5.48xlarge | 8x H100 | 640GB | ~$98.00 |

*Approximate on-demand pricing, varies by region

### Configuration Files

Edit `hf_recipes/Qwen/Qwen3-1.7B--mt-grpo.yaml`:

```yaml
model_name_or_path: Qwen/Qwen2.5-1.5B-Instruct
num_epochs: 15
learning_rate: 5e-7
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
```

## 📊 Monitoring

### SageMaker Console
- Navigate to SageMaker → Training Jobs
- View real-time metrics and logs
- Monitor resource utilization

### CloudWatch Logs
- Automatic logging to CloudWatch
- Search and filter training logs
- Set up alarms for failures

### Notebook Monitoring
```python
# In your notebook
estimator.logs()  # Stream logs in real-time
```

## 💰 Cost Optimization

### Tips to Reduce Costs

1. **Use Spot Instances**
   ```python
   estimator = PyTorch(
       ...,
       use_spot_instances=True,
       max_wait=7200,  # 2 hours
       max_run=3600,   # 1 hour
   )
   ```
   Save up to 70% on training costs.

2. **Right-Size Instances**
   - Start with smaller instances for testing
   - Scale up only when needed
   - Use multi-instance training for large models

3. **Enable Checkpointing**
   - Save checkpoints to S3 regularly
   - Resume from checkpoints if interrupted
   - Useful with spot instances

4. **Use Managed Spot Training**
   - Automatic checkpoint/resume
   - Handles spot interruptions
   - Significant cost savings

## 🔍 Key Files

### `train.py`
SageMaker entry point that:
- Parses hyperparameters from SageMaker
- Sets up distributed training
- Launches the training script
- Handles checkpointing and logging

### `mt_grpo_trainer.py`
Core trainer implementation:
- Multi-turn GRPO algorithm
- Reward calculation
- Policy updates
- Shared with local training

### `sm_mt_grpo_train.sh`
Training script that:
- Configures environment for SageMaker
- Starts vLLM server
- Launches distributed training
- Handles cleanup

### `notebooks/launch_training_job.ipynb`
Interactive notebook for:
- Configuring training parameters
- Launching jobs
- Monitoring progress
- Retrieving results

## 🚦 Training Workflow

### 1. Prepare Configuration

```bash
# Edit training config
vim hf_recipes/Qwen/Qwen3-1.7B--mt-grpo.yaml
```

### 2. Launch Training Job

Open `notebooks/launch_training_job.ipynb` and run cells to:
- Set up AWS credentials
- Configure training parameters
- Launch the job

### 3. Monitor Progress

```python
# In notebook
estimator.logs()  # Stream logs

# Or check SageMaker console
# https://console.aws.amazon.com/sagemaker/home#/jobs
```

### 4. Retrieve Results

```python
# Get model artifacts
model_data = estimator.model_data
print(f"Model saved to: {model_data}")

# Download to local
!aws s3 cp {model_data} ./model.tar.gz
!tar -xzf model.tar.gz
```

## 🐛 Troubleshooting

### Common Issues

**Job Fails to Start**
- Check IAM role permissions
- Verify S3 bucket access
- Ensure instance type is available in region

**Out of Memory**
- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps`
- Use larger instance type

**Slow Training**
- Enable distributed training
- Use multiple instances
- Optimize data loading

**High Costs**
- Use spot instances
- Right-size instance types
- Enable early stopping

### Debugging

```python
# Enable debug mode
estimator = PyTorch(
    ...,
    debugger_hook_config=False,  # Disable debugger to reduce overhead
    environment={
        'LOGLEVEL': 'DEBUG',
    }
)
```

## 📈 Performance Optimization

### Multi-Instance Training

For large models, use multiple instances:

```python
estimator = PyTorch(
    ...,
    instance_count=2,  # Use 2 instances
    distribution={
        'torch_distributed': {
            'enabled': True
        }
    }
)
```

### Custom Docker Container

For advanced use cases, use a custom container:

```python
estimator = PyTorch(
    ...,
    image_uri='<your-ecr-image>',
)
```

### Hyperparameter Tuning

Use SageMaker Hyperparameter Tuning:

```python
from sagemaker.tuner import HyperparameterTuner

tuner = HyperparameterTuner(
    estimator,
    objective_metric_name='eval_reward',
    hyperparameter_ranges={
        'learning_rate': ContinuousParameter(1e-7, 1e-5),
        'per_device_train_batch_size': IntegerParameter(1, 4),
    },
    max_jobs=10,
    max_parallel_jobs=2,
)

tuner.fit()
```

## 🔐 Security Best Practices

1. **Use IAM Roles**: Never hardcode credentials
2. **Encrypt Data**: Enable S3 encryption
3. **VPC Configuration**: Run in private VPC
4. **Network Isolation**: Enable network isolation
5. **Audit Logs**: Enable CloudTrail logging

## 📚 Additional Resources

- [SageMaker Training Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/train-model.html)
- [SageMaker Python SDK](https://sagemaker.readthedocs.io/)
- [SageMaker Examples](https://github.com/aws/amazon-sagemaker-examples)
- [Cost Optimization Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/inference-cost-optimization.html)

## 🤝 Support

For issues or questions:
1. Check SageMaker console for job status
2. Review CloudWatch logs
3. Verify IAM permissions
4. Test with smaller instance first

## 📄 License

See repository root for license information.
