# Multi-Turn GRPO (MTGRPO)

Multi-Turn Group Relative Policy Optimization for training language models with multi-turn conversational reinforcement learning.

## 📁 Project Structure

```
MTGRPO/
├── local_training/          # Train on EC2/local instances (P5, P4d, etc.)
│   ├── notebooks/           # Jupyter notebooks for local training
│   ├── scripts/             # Training scripts and utilities
│   ├── configs/             # Accelerate and training configs
│   ├── hf_recipes/          # HuggingFace model recipes
│   ├── rewards/             # Reward function implementations
│   ├── tools_funcs/         # Tool functions (e.g., wiki search)
│   └── *.md                 # Documentation files
│
└── sagemaker_training/      # Train on SageMaker Training Jobs
    ├── notebooks/           # Jupyter notebooks for launching jobs
    ├── scripts/             # Training entry point scripts
    ├── configs/             # Accelerate and training configs
    ├── hf_recipes/          # HuggingFace model recipes
    ├── rewards/             # Reward function implementations
    ├── tools_funcs/         # Tool functions (e.g., wiki search)
    └── README.md            # SageMaker-specific documentation
```

## 🚀 Quick Start

### Local Training (EC2/P5 Instances)

For training on your own EC2 instances or local machines with GPUs:

1. Navigate to `local_training/`
2. Review the documentation:
   - `README.md` - Complete setup and usage guide
   - `QUICKSTART.md` - Fast setup instructions
   - `GETTING_STARTED.md` - Detailed walkthrough
3. Use the notebook: `notebooks/local_training.ipynb`
4. Or run scripts directly from `scripts/`

**Key Features:**
- Multi-GPU training with DeepSpeed ZeRO-3
- vLLM server for fast inference during rollouts
- Full control over training environment
- Cost-effective for extended training runs

### SageMaker Training Jobs

For managed training on AWS SageMaker:

1. Navigate to `sagemaker_training/`
2. Review `README.md` for setup instructions
3. Use the notebook: `notebooks/launch_training_job.ipynb`
4. Launch training jobs with automatic scaling and monitoring

**Key Features:**
- Managed infrastructure and scaling
- Built-in monitoring and logging
- Easy experiment tracking
- Pay-per-use pricing model

## 🎯 Use Cases

**Choose Local Training when:**
- You have dedicated GPU instances (EC2 P5, P4d, etc.)
- You need full control over the training environment
- Running long training jobs where reserved instances are cost-effective
- Debugging and iterating on training code

**Choose SageMaker Training when:**
- You want managed infrastructure
- Need automatic scaling and monitoring
- Running one-off experiments
- Want integrated experiment tracking with SageMaker

## 📊 Architecture

Both training modes use the same core components:

- **Multi-Turn GRPO Trainer**: Custom trainer for multi-turn RL
- **Reward Functions**: TriviaQA and custom reward implementations
- **Tool Functions**: Wikipedia search and other tools for agent interactions
- **HuggingFace Recipes**: Pre-configured training recipes for popular models

### Training Flow

1. **Rollout Phase**: Generate multi-turn conversations using vLLM
2. **Reward Calculation**: Evaluate responses using reward functions
3. **Policy Update**: Update model using GRPO algorithm
4. **Repeat**: Iterate for specified number of epochs

## 🔧 Configuration

Training configurations are stored in `configs/` and `hf_recipes/`:

- **Accelerate configs**: Multi-GPU training setup (DeepSpeed, FSDP)
- **Model recipes**: Hyperparameters for specific models (Qwen, Llama, etc.)
- **Reward configs**: Reward function parameters

## 📝 Key Files

### Local Training
- `mt_grpo_trainer.py` - Core trainer implementation
- `requirements_local.txt` - Python dependencies
- `scripts/local_mt_grpo_train.sh` - Main training script
- `scripts/launch_local_training_job.sh` - Job launcher
- `scripts/run_training_auto.py` - Automated training runner
- `scripts/test_setup.py` - Environment verification

### SageMaker Training
- `mt_grpo_trainer.py` - Core trainer implementation
- `requirements.txt` - Python dependencies
- `train.py` - SageMaker entry point
- `scripts/sm_mt_grpo_train.sh` - Training script for SageMaker
- `notebooks/launch_training_job.ipynb` - Job launcher notebook

## 🛠️ Requirements

### Local Training
- Python 3.10+
- PyTorch 2.8.0+
- 8x GPUs (A100 or H100 recommended)
- Java 21 (for Pyserini/Wikipedia search)
- CUDA 12.1+

### SageMaker Training
- AWS Account with SageMaker access
- IAM role with SageMaker permissions
- S3 bucket for artifacts
- Python 3.10+ (for launching jobs)

## 📚 Documentation

Each training mode has its own detailed documentation:

- **Local Training**: See `local_training/README.md`
- **SageMaker Training**: See `sagemaker_training/README.md`

## 🤝 Contributing

When adding new features:
1. Update both local and SageMaker implementations if applicable
2. Add tests in the appropriate `scripts/` directory
3. Update relevant documentation
4. Test on both training modes before committing

## 📄 License

See repository root for license information.
