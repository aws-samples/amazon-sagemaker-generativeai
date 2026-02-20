# Local Multi-Turn GRPO Training

Train Multi-Turn GRPO models on your own EC2 instances or local machines with GPUs.

## 📋 Overview

This directory contains everything needed to run Multi-Turn GRPO training on local or EC2 GPU instances (P5, P4d, etc.). The training uses a distributed setup with vLLM for fast inference during rollouts.

## 🏗️ Architecture

- **GPU 7**: Dedicated vLLM server for fast inference during rollouts
- **GPUs 0-6**: Distributed training with DeepSpeed ZeRO-3 or FSDP
- **Multi-Turn RL**: Conversational agent training with tool use

## 🚀 Quick Start

### Option 1: Using Jupyter Notebook

```bash
jupyter notebook notebooks/local_training.ipynb
```

Follow the step-by-step instructions in the notebook.

### Option 2: Using Scripts

```bash
# 1. Install dependencies
pip install -r requirements_local.txt

# 2. Verify setup
python scripts/test_setup.py

# 3. Launch training (automated)
python scripts/run_training_auto.py

# Or launch training (step-by-step)
python scripts/run_training_step_by_step.py

# Or use the shell script directly
bash scripts/local_mt_grpo_train.sh
```

## 📁 Directory Structure

```
local_training/
├── notebooks/
│   └── local_training.ipynb       # Interactive training notebook
├── scripts/
│   ├── local_mt_grpo_train.sh     # Main training script
│   ├── launch_local_training_job.sh  # Job launcher
│   ├── run_training_auto.py       # Automated training runner
│   ├── run_training_step_by_step.py  # Step-by-step runner
│   └── test_setup.py              # Environment verification
├── configs/
│   └── accelerate/                # Accelerate configurations
├── hf_recipes/
│   └── Qwen/                      # Model-specific recipes
├── rewards/
│   └── triviaqa_reward.py         # Reward function
├── tools_funcs/
│   └── wiki_search.py             # Wikipedia search tool
├── mt_grpo_trainer.py             # Core trainer implementation
├── requirements_local.txt         # Python dependencies
└── *.md                           # Documentation files
```

## 🔧 Prerequisites

### Hardware
- 8x GPUs (A100 80GB or H100 recommended)
- 500GB+ disk space
- 256GB+ RAM

### Software
- Python 3.10+
- PyTorch 2.8.0+
- CUDA 12.1+
- Java 21 (for Pyserini/Wikipedia search)

### Installation

```bash
# Install Python dependencies
pip install -r requirements_local.txt

# Verify Java installation
java --version  # Should show Java 21

# Test setup
python scripts/test_setup.py
```

## 🎯 Training Configuration

Edit the configuration files in `hf_recipes/` to customize training:

```yaml
# Example: hf_recipes/Qwen/Qwen3-1.7B--mt-grpo.yaml
model_name_or_path: Qwen/Qwen2.5-1.5B-Instruct
num_epochs: 15
learning_rate: 5e-7
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
```

## 📊 Monitoring

Training metrics are logged to:
- **MLflow**: Local MLflow server (check `mlflow.db`)
- **Weights & Biases**: If configured (check `wandb/`)
- **Console**: Real-time training logs

## 🐛 Troubleshooting

### Common Issues

**GPU Out of Memory**
- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps`
- Enable gradient checkpointing

**vLLM Server Fails**
- Check GPU 7 is available: `nvidia-smi`
- Verify model fits in GPU memory
- Check vLLM logs: `vllm_server.log`

**Java Not Found**
- Install Java 21: `sudo apt install openjdk-21-jdk`
- Set JAVA_HOME: `export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64`

**Slow Training**
- Verify all GPUs are being used: `nvidia-smi`
- Check network bandwidth between GPUs
- Enable flash attention if supported

## 📚 Documentation

- `README.md` (this file) - Complete setup and usage
- `QUICKSTART.md` - Fast setup for experienced users
- `GETTING_STARTED.md` - Detailed walkthrough for beginners
- `SUMMARY.md` - Technical summary
- `SAGEMAKER_STYLE_USAGE.md` - SageMaker-style API usage

## 🔍 Key Scripts

### `local_mt_grpo_train.sh`
Main training script that:
1. Starts vLLM server on GPU 7
2. Launches distributed training on GPUs 0-6
3. Handles cleanup on exit

### `run_training_auto.py`
Automated training runner that:
- Validates environment
- Starts training automatically
- Monitors progress

### `run_training_step_by_step.py`
Interactive training runner that:
- Guides through each step
- Allows manual intervention
- Useful for debugging

### `test_setup.py`
Environment verification that checks:
- GPU availability
- Python packages
- Java installation
- Disk space

## 💡 Tips

1. **Start Small**: Test with a small model (1.5B) before scaling up
2. **Monitor GPUs**: Use `watch -n 1 nvidia-smi` to monitor GPU usage
3. **Save Checkpoints**: Training saves checkpoints every N steps
4. **Use tmux/screen**: Run training in a persistent session
5. **Check Logs**: Monitor `vllm_server.log` for inference issues

## 🚦 Training Workflow

1. **Prepare Environment**
   ```bash
   pip install -r requirements_local.txt
   python scripts/test_setup.py
   ```

2. **Configure Training**
   - Edit `hf_recipes/Qwen/Qwen3-1.7B--mt-grpo.yaml`
   - Adjust hyperparameters as needed

3. **Launch Training**
   ```bash
   python scripts/run_training_auto.py
   # or
   bash scripts/local_mt_grpo_train.sh
   ```

4. **Monitor Progress**
   - Watch console output
   - Check MLflow UI: `mlflow ui`
   - Monitor GPU usage: `nvidia-smi`

5. **Evaluate Results**
   - Check saved checkpoints
   - Review training metrics
   - Test model inference

## 📈 Performance Optimization

- **DeepSpeed ZeRO-3**: Efficient memory usage across GPUs
- **Flash Attention**: 2-4x faster attention computation
- **vLLM**: Fast inference during rollouts (10-20x speedup)
- **Gradient Checkpointing**: Trade compute for memory

## 🤝 Support

For issues or questions:
1. Check documentation in this directory
2. Review error logs and troubleshooting section
3. Test with `scripts/test_setup.py`
4. Verify configuration files

## 📄 License

See repository root for license information.
