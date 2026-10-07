# Sequence-level Knowledge Distillation (SeqKD) for a Tool-Use Agent

Distill a teacher's tool-use behavior into **Qwen/Qwen3.5-4B** with **sequence-level knowledge distillation**:
supervised fine-tuning on the teacher's trajectories, trained with the serverless `SFTTrainer` (SageMaker Python SDK
v3) and evaluated as an agent on [ShoppingBench](https://github.com/yjwjy/ShoppingBench).

## Overview

1. **Data** (`distill-tool-call--Qwen--Qwen3.5-4B-shopping.ipynb`, step 01): teacher trajectories from
   `oro-ai/sn15-shoppingbench-sft-15k`, one training example per tool-calling step, built with ShoppingBench's own
   prompt functions so training text matches what the benchmark sends at evaluation time. The train/validation split
   is by query.
2. **Training** (step 02): serverless `SFTTrainer`, LoRA; the job registers a ready-to-deploy Model Package.
3. **Evaluation** (`evaluate-tool-call-accuracy.ipynb`): base and student on SageMaker real-time endpoints, run
   through ShoppingBench's agent loop on **held-out** test problems (queries absent from the training corpus), scored
   with the benchmark's rules (ASR), plus optional IFEval.

Reference results (LoRA r64, held-out problems, thinking on), base → student ASR:

| product (103) | shop (105) | voucher (116) | mean |
|---|---|---|---|
| 0.379 → 0.466 | 0.114 → 0.343 | 0.095 → 0.129 | 0.196 → 0.313 |

> **Why held-out problems.** The teacher corpus is built on ShoppingBench test queries (147 of the 250 product test
> queries appear in it verbatim), so a score on the full test set partly measures recall. The evaluation notebook
> scores only test problems whose query never appears in the corpus.

## Files

| File | Purpose |
|---|---|
| `distill-tool-call--Qwen--Qwen3.5-4B-shopping.ipynb` | Data prep, serverless SFT, Model Package |
| `evaluate-tool-call-accuracy.ipynb` | ShoppingBench ASR (and IFEval) for base, student and an optional Bedrock reference model |
| `scripts/setup_shopbench.sh` | Install the ShoppingBench harness, catalog and search index (`--code-only` for data prep) |
| `scripts/make_clean_split.py` | Write the held-out test problems |
| `scripts/score_rollouts.py` | Score trajectories with ShoppingBench's rules |
| `scripts/shopbench_harness.py` | Search server, agent rollouts, scoring and health checks |
| `scripts/sm_endpoints.py` | Deploy / delete the SageMaker endpoints (incl. a ModelBuilder variant) |
| `scripts/sm_openai_relay.py` | Local OpenAI-compatible relay: SageMaker bearer tokens per request, or Bedrock |
| `scripts/run_ifeval.py` | IFEval with lm-evaluation-harness, with a reasoning-leak check |
| `scripts/inspect_teacher_dataset.py` | Inspect the teacher dataset |

## Notes on the model

Qwen3.5-4B is a **linear-attention / conv1d hybrid**, and that is the main reason to train it on the
managed serverless recipe rather than a training script of your own. The linear-attn path needs
`flash-linear-attention`, which requires `triton>=3.7.1` — and that collides with the `triton` pin a
stock PyTorch image carries, so a plain `pip install` either dead-ends on `ResolutionImpossible` or
bumps `torch` to a version that breaks `transformers`. The hybrid also wants `sdpa` attention, because
`flash_attention_2` hits CUDA illegal-memory-access on SageMaker's managed driver. The serverless recipe
handles both; you never see them.

Inference is simpler: the LMI image `djl-inference:0.36.0-lmi25.0.0-cu130` bundles **vLLM 0.20.1**,
which has native kernels for this hybrid, so **no custom inference image is needed**. Note that
`ModelServer.VLLM` (HF vLLM 0.29.0) is *not* a working alternative for this architecture — it
crash-loops on `Failed to promote local KV cache specs`.

## License

See the repository root `LICENSE`.
