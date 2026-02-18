# Getting Started with Local P5 MT-GRPO Training

## Quick Start (5 Minutes)

### 1. Set Wandb Credentials (Optional)

```bash
export WANDB_API_KEY="your-key"
export WANDB_ENTITY="suryachaitanya-aws-org"
export WANDB_PROJECT="mt-grpo-training"
```

Skip this if you don't want wandb tracking - training works fine without it!

### 2. Launch Training

```bash
cd /home/ubuntu/mt-grpo/local_training
./launch_local_training_job.sh --config hf_recipes/Qwen/Qwen3-0.6B--mt-grpo.yaml
```

That's it! The script will:
- Launch a fresh Docker container
- Install Java 21 and Python dependencies
- Start vLLM server on GPU 7
- Start training on GPUs 0-6
- Show you the logs in real-time

### 3. Monitor Training

Press **Ctrl+C** to stop watching logs (training continues in background)

```bash
# Reconnect to logs
docker logs -f mt-grpo-training-<timestamp>

# Check GPU usage
nvidia-smi

# Check outputs
ls -lh ~/mt-grpo-outputs/
```

### 4. Stop Training

```bash
docker stop mt-grpo-training-<timestamp>
```

## What This Does

This setup mimics SageMaker training jobs but runs locally on your P5 instance:

- **Fresh container per job** - Clean environment every time
- **Automatic dependency installation** - Java, Python packages, etc.
- **GPU allocation** - GPU 7 for vLLM, GPUs 0-6 for training
- **Background execution** - Training continues even if you disconnect
- **Output persistence** - Saves to `~/mt-grpo-outputs/`

## Architecture

```
┌─────────────────────────────────────────┐
│         P5 Instance (8 GPUs)            │
├─────────────────────────────────────────┤
│                                         │
│  GPU 7: vLLM Server                     │
│  └─ Fast inference (port 8000)         │
│                                         │
│  GPUs 0-6: Training                     │
│  └─ DeepSpeed ZeRO-3 + Accelerate      │
│                                         │
└─────────────────────────────────────────┘
```

## Files Included

- **launch_local_training_job.sh** - Main script (SageMaker-style)
- **requirements_local.txt** - Working dependency versions
- **hf_recipes/Qwen/** - Model configurations
- **mt_grpo_trainer.py** - Training implementation
- **configs/accelerate/** - DeepSpeed configuration
- **rewards/** - Reward functions
- **tools_funcs/** - Tool implementations

## Configuration

Edit YAML files in `hf_recipes/Qwen/` to customize:

```yaml
model_name_or_path: "Qwen/Qwen3-0.6B"
max_steps: 200
learning_rate: 1.0e-6
per_device_train_batch_size: 2
num_generations: 7
max_env_steps: 5
turn_advantage_coef: 1.0
```

## Outputs

Training saves to `~/mt-grpo-outputs/`:

```
mt-grpo-outputs/
├── model/          # Final trained model
├── checkpoints/    # Training checkpoints
└── output/         # Logs and metrics
```

## Wandb Integration (Optional)

Set these before training to enable wandb tracking:

```bash
export WANDB_API_KEY="your-key-from-wandb.ai"
export WANDB_ENTITY="suryachaitanya-aws-org"
export WANDB_PROJECT="mt-grpo-training"
```

If not set, training still works - logs are just saved locally.

## Common Commands

```bash
# Launch training
./launch_local_training_job.sh --config hf_recipes/Qwen/Qwen3-0.6B--mt-grpo.yaml

# Stop watching logs (training continues)
Ctrl+C

# Reconnect to logs
docker logs -f <container-name>

# Stop training
docker stop <container-name>

# List running containers
docker ps | grep mt-grpo

# Check GPU usage
nvidia-smi

# View outputs
ls -lh ~/mt-grpo-outputs/
```

## Troubleshooting

### Container exits immediately
```bash
docker logs <container-name>
```

### Out of memory
Reduce in config YAML:
- `per_device_train_batch_size: 1`
- `num_generations: 5`

### Port already in use
```bash
pkill -f "vllm.entrypoints.openai.api_server"
```

## Advanced Options

```bash
# Run in foreground (Ctrl+C stops training)
./launch_local_training_job.sh --config <yaml> --foreground

# Don't follow logs automatically
./launch_local_training_job.sh --config <yaml> --no-follow

# Specify GPU count
./launch_local_training_job.sh --config <yaml> --num_gpus 8

# Custom vLLM port
./launch_local_training_job.sh --config <yaml> --vllm_port 8001
```

## What Gets Trained

- **Task**: TriviaQA question answering with Wikipedia search
- **Method**: Multi-Turn GRPO (Group Relative Policy Optimization)
- **Tools**: Wikipedia search via Pyserini
- **Rewards**: 
  - Turn-level: Tool execution success, answer in results
  - Outcome-level: Exact match, format compliance

## Next Steps

1. **Run your first training** with the quick start above
2. **Monitor progress** with wandb or local logs
3. **Customize configs** in `hf_recipes/Qwen/`
4. **Try different models** (Qwen3-0.6B, Qwen3-1.7B, etc.)

## Support

- Full documentation: `README.md`
- SageMaker comparison: `SAGEMAKER_STYLE_USAGE.md`
- Quick reference: `QUICKSTART.md`

---

**Ready to start?**

```bash
./launch_local_training_job.sh --config hf_recipes/Qwen/Qwen3-0.6B--mt-grpo.yaml
```
