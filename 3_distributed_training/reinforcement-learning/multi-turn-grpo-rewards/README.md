# SageMaker MT-GRPO Training

Multi-Turn Group Relative Policy Optimization (MT-GRPO) training on AWS SageMaker with distributed training and co-located vLLM inference.

## Structure

```
sagemaker-mt-grpo/
├── notebooks/
│   ├── complete_training_guide.ipynb  # Main training notebook
│   └── train_mt_grpo.ipynb           # Legacy notebook
├── scripts/
│   ├── train.py                      # SageMaker entry point
│   ├── requirements.txt              # Pinned dependencies
│   └── verifiers/                    # Training code
│       ├── examples/
│       │   └── triviaqa_search.py   # TriviaQA training script
│       └── trainers/
│           └── mt_grpo_env_trainer.py  # Co-located vLLM trainer
└── configs/
    └── zero3.yaml                    # DeepSpeed ZeRO-3 config
```

## Quick Start

1. Open `notebooks/complete_training_guide.ipynb`
2. Configure for **ml.p4d.24xlarge** (8x A100 GPUs)
3. Run all cells

Training will start with:
- **7 GPUs** for distributed training (DeepSpeed ZeRO-3)
- **1 GPU** dedicated to vLLM inference
- **Qwen2.5-3B** model (fits comfortably in memory)

## Architecture

### Co-Located vLLM (Current Setup)

```
┌─────────────────────────────────────────┐
│  ml.p4d.24xlarge (8x A100 40GB)        │
├─────────────────────────────────────────┤
│  Training: GPUs 0-6 (DeepSpeed ZeRO-3) │
│  vLLM:     GPU 7 (tp=1, dedicated)     │
└─────────────────────────────────────────┘
```

**Advantages:**
- ✅ Simple architecture (in-process)
- ✅ Low latency (no network overhead)
- ✅ Easy to debug and monitor
- ✅ Works great for models ≤7B

**Limitations:**
- ❌ vLLM limited to `tensor_parallel_size=1`
- ❌ Can't use models that don't fit on 1 GPU

### GPU Allocation

| Component | GPUs | Purpose |
|-----------|------|---------|
| Training | 0-6 (7 GPUs) | Model training with DeepSpeed |
| vLLM | 7 (1 GPU) | Generation for GRPO rollouts |

**Critical:** `num_generations` must divide evenly into `(num_gpus - 1) × batch_size`
- Example: 7 training GPUs × 2 batch size = 14 → use `num_generations=14` or `num_generations=7`

## Configuration

### Recommended Settings (8 GPU Setup)

```python
hyperparameters = {
    'model_name': 'Qwen/Qwen2.5-3B',      # 3B fits well, 7B needs careful tuning
    'num_gpus': 8,                         # Total GPUs (7 train + 1 vLLM)
    'learning_rate': 1e-6,
    'num_generations': 14,                 # Must divide into 7×2=14
    'per_device_train_batch_size': 2,
    'grad_accum_steps': 4,
    'num_iterations': 2,
    'max_steps': 300,
    'beta': 0,
    'trainer': 'mt_grpo',
    'turn_advantage_coef': 1,
}
```

### Environment Variables

```python
environment = {
    'VLLM_WORKER_MULTIPROC_METHOD': 'spawn',  # Prevents NCCL timeout
    'WANDB_API_KEY': 'your_key_here',         # Optional: W&B logging
}
```

## Model Selection Guide

| Model | Memory | Recommended Setup | Notes |
|-------|--------|-------------------|-------|
| **Qwen2.5-3B** | ~6GB | ✅ 1 GPU (tp=1) | **Recommended** - fits easily |
| **Qwen2.5-7B** | ~14GB | ⚠️ 1 GPU (tp=1) | Tight fit, may need `max_model_len` tuning |
| **Qwen2.5-13B+** | >26GB | ❌ Needs tp>1 | Requires external vLLM server |

### Memory Breakdown (7B Model on A100 40GB)

```
Total GPU Memory:        40.00 GB
GPU Memory Used (85%):   33.57 GB
├─ Model Weights:        14.27 GB
├─ Activation Peak:      17.39 GB
├─ Non-Torch Memory:      0.09 GB
└─ KV Cache:              1.82 GB  ← Often insufficient for 131K context
```

**Solution for 7B:** Set `max_model_len=8192` or use 3B model

## Key Fixes Applied

### 1. NCCL Timeout Prevention
```python
environment = {'VLLM_WORKER_MULTIPROC_METHOD': 'spawn'}
```
Prevents vLLM from creating conflicting NCCL process groups.

### 2. vLLM Memory Configuration
```python
training_args.vllm_gpu_memory_utilization = 0.85  # Use 85% of dedicated GPU
training_args.vllm_device = f"cuda:{args.num_gpus-1}"  # Explicit GPU assignment
```

### 3. Pinned Dependencies
`requirements.txt` uses exact versions to avoid 30+ minute dependency resolution:
- `torch==2.5.1`
- `vllm==0.7.3`
- `transformers==4.49.0`
- TRL from specific commit

### 4. Batch Size Math
```python
# Global batch size = (num_gpus - 1) × per_device_batch_size
# num_generations must divide evenly into global batch size
num_generations = 14  # For 7 training GPUs × 2 batch size
```

## Instance Recommendations

### Development/Testing
- **ml.g5.24xlarge** (4x A10G, 24GB each) - ~$10/hour
  - Use Qwen2.5-3B
  - 3 training GPUs + 1 vLLM GPU
  - `num_generations=6` (3×2=6)

### Production (Recommended)
- **ml.p4d.24xlarge** (8x A100, 40GB each) - ~$32/hour
  - Use Qwen2.5-3B or 7B
  - 7 training GPUs + 1 vLLM GPU
  - `num_generations=14` (7×2=14)

### Large Models
- **ml.p4de.24xlarge** (8x A100, 80GB each) - ~$40/hour
  - Can handle 13B+ models
  - More KV cache space for long contexts

## Monitoring

### CloudWatch Logs
Automatic logging shows:
- Training progress and loss
- vLLM initialization and memory usage
- Generation throughput

### Weights & Biases
Add API key to environment variables for:
- Real-time metrics
- Loss curves
- Generation samples

### Key Metrics to Watch
```
INFO vLLM instance can use: 33.57GiB
INFO model weights take: 14.27GiB
INFO KV Cache: 1.82GiB
INFO # cuda blocks: 2131
```

## Troubleshooting

### NCCL Timeout During vLLM Init
**Symptom:** Hangs at "Initializing distributed environment"
**Fix:** Ensure `VLLM_WORKER_MULTIPROC_METHOD=spawn` is set

### ValueError: Batch Size Not Divisible
**Symptom:** `The global train batch size (7 x 2) must be evenly divisible by the number of generations per prompt (16)`
**Fix:** Set `num_generations` to valid value (e.g., 14, 7, or 2 for 7 training GPUs)

### ValueError: Max Seq Len Larger Than KV Cache
**Symptom:** `The model's max seq len (131072) is larger than the maximum number of tokens that can be stored in KV cache`
**Fix:** 
- Use smaller model (Qwen2.5-3B instead of 7B)
- Or set `max_model_len=8192` in vLLM config

### OOM Errors
**Fix:**
- Reduce `per_device_train_batch_size` (2 → 1)
- Reduce `num_generations` (14 → 7)
- Use Qwen2.5-3B instead of 7B
- Increase `vllm_gpu_memory_utilization` (0.85 → 0.90)

### Slow Dependency Installation (30+ minutes)
**Symptom:** pip backtracking for extended periods
**Fix:** Already fixed - `requirements.txt` has pinned versions

## Advanced: External vLLM Server

For models requiring `tensor_parallel_size > 1`:

1. Switch to `mt_grpo_trainer.py` (not `mt_grpo_env_trainer.py`)
2. Run vLLM as separate server process
3. Configure training to use HTTP API

**Trade-offs:**
- ✅ Supports tensor parallelism (tp>1)
- ✅ Can handle 13B+ models
- ❌ Network overhead
- ❌ More complex setup

## Outputs

Training artifacts saved to S3:
- **Model checkpoints**: `s3://{bucket}/mt-grpo-training/checkpoints/`
- **Final model**: `s3://{bucket}/mt-grpo-training/output/`
- **Logs**: CloudWatch Logs (auto-configured)

## Cost Optimization

1. **Spot Instances**: 70% savings (add `use_spot_instances=True`)
2. **Right-size instance**: Start with g5.24xlarge for 3B models
3. **Checkpoint frequency**: Balance between safety and S3 costs
4. **Early stopping**: Monitor validation metrics

## Next Steps

After successful training:
1. Download model from S3
2. Evaluate on TriviaQA test set
3. Deploy with SageMaker Endpoints or vLLM server
4. Fine-tune hyperparameters based on results

## References

- [TRL Documentation](https://huggingface.co/docs/trl)
- [vLLM Documentation](https://docs.vllm.ai)
- [DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/)
- [SageMaker Training](https://docs.aws.amazon.com/sagemaker/latest/dg/train-model.html)
