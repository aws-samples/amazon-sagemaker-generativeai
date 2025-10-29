# SAMA RL - Simple API for Reinforcement Learning with Language Models

SAMA RL provides a clean, intuitive API for training language models with reinforcement learning algorithms like GRPO (Group Relative Policy Optimization). It includes deployment and inference capabilities for trained models.

## Key Features

- YAML Configuration - Clean, readable config files for all settings
- SageMaker Integration - Automatic training job deployment and scaling
- Model Deployment - Deploy trained models to SageMaker endpoints
- Inference API - Simple interface for model inference
- Cost Optimization - Intelligent instance selection based on model size

## Quick Start

### 1. Installation

```bash
pip install -e .
```

### 2. Complete Example

```python
from sama_rl import GRPO, create_inference_model

# 1. Create reward function
def length_reward(completions, **kwargs):
    target_length = 400
    tokenizer = kwargs.get('tokenizer')
    rewards = []
    for completion in completions:
        if tokenizer:
            num_tokens = len(tokenizer.encode(completion, add_special_tokens=False))
        else:
            num_tokens = len(completion.split())
        reward = -(abs(num_tokens - target_length) ** 2) / 1000
        rewards.append(reward)
    return rewards

# 2. Train model
trainer = GRPO(
    yaml_file="sama_rl/qwen2-0.5b-grpo-config.yaml",
    reward_functions=[length_reward],
    max_steps=100,
    instance_type="ml.g4dn.2xlarge"
)
trainer.train()

# 3. Deploy model
endpoint_name = trainer.deploy()

# 4. Run inference
model = create_inference_model(endpoint_name)
completion = model.generate("What is machine learning?", max_new_tokens=200)
```

## Project Structure

```
sama_rl/
├── __init__.py                           # Core imports
├── grpo.py                              # GRPO trainer with SageMaker integration
├── deployment.py                        # Model deployment utilities
├── inference.py                         # Inference utilities
├── config_loader.py                     # YAML configuration loader
├── sagemaker_train.py                   # SageMaker training script
├── requirements.txt                     # Dependencies
├── qwen2-0.5b-grpo-config.yaml        # Example config for Qwen2-0.5B
└── qwen2-1.5b-helpfulness-config.yaml # Example config for Qwen2-1.5B

notebooks/
└── SAMA_GRPO.ipynb                     # Complete examples and usage guide
```

## Configuration

SAMA RL uses YAML files for configuration:

```yaml
model:
  name: "Qwen/Qwen2-0.5B-Instruct"
  trust_remote_code: true

data:
  dataset_name: "trl-lib/tldr"
  train_split: "train[:8000]"
  test_split: "test[:200]"

training:
  max_steps: 800
  learning_rate: 5e-5
  per_device_train_batch_size: 4
  fp16: true
  gradient_checkpointing: true

grpo:
  num_generations: 2
  max_completion_length: 768
  temperature: 0.7

sagemaker:
  instance_type: "ml.g4dn.2xlarge"
  max_run: 3600
```

## Training

### New Training Job

```python
trainer = GRPO(
    yaml_file="config.yaml",
    reward_functions=[reward_function],
    max_steps=100
)
trainer.train()
```

### Load Existing Training Job

```python
trainer = GRPO(training_job_name="sama-grpo-qwen205binstruct-1234567890")
endpoint_name = trainer.deploy()
```

## Deployment

### Automatic Instance Selection

```python
# Auto-selects appropriate GPU instance based on model size
endpoint_name = trainer.deploy()
```

### Manual Instance Selection

```python
# Specify instance type
endpoint_name = trainer.deploy(instance_type="ml.g5.2xlarge")
```

### Supported Instance Types

- ml.g5.xlarge - Small models (0.5B-1B parameters)
- ml.g5.2xlarge - Medium models (1B-3B parameters)  
- ml.g5.4xlarge - Large models (7B+ parameters)
- ml.g5.12xlarge - Very large models (13B+ parameters)

## Inference

### Basic Usage

```python
from sama_rl import create_inference_model

model = create_inference_model(endpoint_name)
completion = model.generate(
    prompt="Explain reinforcement learning",
    max_new_tokens=200,
    temperature=0.7
)
```

### Batch Inference

```python
prompts = ["Question 1", "Question 2", "Question 3"]
completions = model.batch_inference(prompts, max_new_tokens=100)
```

### Repetition Control

```python
completion = model.generate(
    prompt="What is the capital of France?",
    max_new_tokens=100,
    temperature=0.7,
    stop_on_repetition=True  # Prevents repetitive output
)
```

## Training Job Management

### Job Naming Convention

Training jobs are automatically named: `sama-grpo-{modelname}-{timestamp}`

### Monitoring

```python
# Check status
print(trainer.get_training_job_status())

# Get model artifacts
print(trainer.get_model_artifacts())

# View logs
trainer.get_logs()
```

## Configuration Overrides

All parameters can be overridden at runtime:

```python
trainer = GRPO(
    yaml_file="config.yaml",
    reward_functions=[reward_func],
    # Training parameters
    max_steps=500,
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    # GRPO parameters
    num_generations=1,
    temperature=0.6,
    # Data parameters
    dataset_name="custom/dataset",
    # Model parameters
    model_name="microsoft/DialoGPT-medium"
)
```

## Best Practices

1. **Test with small max_steps first** - Use max_steps=10-50 for initial testing
2. **Use appropriate instance types** - Let auto-selection choose optimal instances
3. **Monitor costs** - Set max_run times to prevent runaway costs
4. **Control repetition** - Use stop_on_repetition=True for cleaner outputs
5. **Start small, scale up** - Begin with smaller models and datasets

## API Reference

### GRPO Class

```python
GRPO(
    yaml_file: str = None,
    reward_functions: List[Callable] = None,
    training_job_name: str = None,  # For loading existing jobs
    instance_type: str = None,
    wandb_api_key: str = None,
    **overrides  # Any config parameter
)
```

### Inference Class

```python
create_inference_model(
    endpoint_name: str,
    base_model_name: str = "Qwen/Qwen2-0.5B-Instruct"
)
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License.

---

SAMA RL: Simple, reliable reinforcement learning for language models.
