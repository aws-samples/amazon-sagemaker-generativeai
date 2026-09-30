# Sequence-level Knowledge Distillation (SeqKD) for a Tool-Use Agent

Distill a capable teacher's tool-use behavior into a smaller student — **Qwen/Qwen3.5-4B** — with
**Sequence-level Knowledge Distillation (SeqKD)**: supervised fine-tuning on the teacher's own
generated trajectories, evaluated on an agentic **shopping** benchmark.

## Overview

SeqKD is behavior cloning: instead of matching the teacher's logits, the student is fine-tuned on the
teacher's *decoded* trajectories (prompt → completion, loss on the completion only). For a tool-use
agent this transfers the ReAct discipline — search the catalog, inspect candidates, verify constraints,
emit a well-formed tool call, then recommend.

The recipe:
1. **Generate / collect** teacher trajectories on the task (ShoppingBench).
2. **Encode** each assistant step into a `{prompt, completion}` pair using the student's exact
   non-thinking chat template (so training text is byte-identical to inference).
3. **Fine-tune** the student (LoRA or full) on those pairs with a SageMaker `ModelTrainer` training job.
4. **Merge** the adapter and **evaluate** — task success rate (ASR) plus MMLU + IFEval for regression.

Reference results (Qwen3.5-4B student, thinking-off): **ASR 43.2 (base) → 62.4 (LoRA r64)**;
MMLU ~0.697 and IFEval ~82.6 held (negligible forgetting). A mid-size teacher transfers as well as a
much larger one — the ReAct-completion discipline is what carries over.

## Quick Start

```bash
# In distill-tool-call--Qwen--Qwen3.5-4B-shopping.ipynb: encode trajectories, upload to S3, then
cd scripts
bash sm_train.sh --max-length 4096 --lora-r 64 --lr 2e-5 --epochs 2 --batch-size 1 --grad-accum 2 --fft false
```

The notebook wraps this in a SageMaker `ModelTrainer` job (8-GPU `ml.p5.48xlarge`).

## Files

| File | Purpose |
|---|---|
| `distill-tool-call--Qwen--Qwen3.5-4B-shopping.ipynb` | End-to-end: data prep → SeqKD training → merge → eval. |
| `evaluate-tool-call-accuracy.ipynb` | Base-vs-student tool-use ASR + MMLU/IFEval portfolio, with charts. |
| `scripts/train_seqkd.py` | TRL SFT entry point (LoRA/FFT, Qwen3.5 VLM class, kernel bootstrap). |
| `scripts/sm_train.sh` | Launcher: installs deps, resolves GPUs, runs `torchrun train_seqkd.py`. |
| `scripts/requirements_verl.txt` | trl/peft/datasets/einops (the kernel pair installs `--no-deps` at runtime). |
| `scripts/eval_lmeval.sh` | Offline lm-eval portfolio: MMLU + non-thinking IFEval. |
| `scripts/shopping_asr_eval.sh` | Tool-use ASR wiring: vLLM serve → ShoppingBench ReAct rollout → ORM scoring (needs the ShoppingBench repo + search server). |

## Notes on the model

Qwen3.5-4B is a **linear-attention / conv1d hybrid**, so it needs `flash-linear-attention` for the
linear-attn path. That package requires `triton>=3.7.1`, which collides with the base image's
`torch`-pinned `triton` under pip's resolver — so `train_seqkd.py` force-installs `triton==3.8.0` +
`flash-linear-attention==0.5.2` with `--no-deps` at job start (`--install-kernels`, default on), and
uses `sdpa` attention (`flash_attention_2` CUDA-IMAs on SageMaker's managed driver). This is why the
notebook uses the HyperPod verl training image rather than a stock PyTorch DLC.

## License

See the repository root `LICENSE`.
