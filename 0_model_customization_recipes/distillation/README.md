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
3. **Fine-tune** the student (LoRA or full) on those pairs with the serverless `SFTTrainer` from the
   SageMaker Python SDK v3.
4. **Evaluate** the registered Model Package the job emits. Three axes, on the compute each needs:
   the headline **ASR** plus non-thinking **IFEval**, which need a GPU host because ASR drives a live
   ReAct loop against the benchmark's Lucene search server; managed serverless **MMLU**; and a
   **per-trajectory reasoning judge** that re-reads the saved rollouts and needs no GPU at all.

Reference results (Qwen3.5-4B student, thinking-off, LoRA r16 / alpha 128), measured on the
**decontaminated 103-case** subset of ShoppingBench's product test split: **ASR 0.379 (base) → 0.495
(student)**, with IFEval 0.826 → 0.782 and MMLU 0.697 → 0.700 — agent skill rises substantially while
general instruction-following and knowledge hold. A mid-size teacher transfers as well as a much larger
one; the ReAct-completion discipline is what carries over.

A per-trajectory LLM reasoning judge (pairwise, blind, order-randomized over the same 103 cases) shows
*which* part of the process improved, and it is narrower than the ASR gain suggests: `efficiency` +1.18
(p<0.001) and `search_strategy` +0.37 (p=0.037) move, while `verification` +0.10 and
`constraint_tracking` +0.09 do not. The base model never issues `recommend_product` at all on 36 of 103
cases, and on exactly those `efficiency` goes 1.89 → 3.86 — SeqKD taught the agent to **finish the
rollout** (completion rate 0.650 → 0.864) rather than to pick better products. The judge is validated
against ground truth: where the two arms' ASR differs and the judge is not tied, it prefers the arm that
actually succeeded in 19 of 22 cases (86.4%).

> **The eval set is contaminated out of the box — decontaminate before you quote a number.** The public
> teacher set `oro-ai/sn15-shoppingbench-sft-15k` has only 469 unique queries, and **147 of the 250 eval
> queries appear in it verbatim**; for 110 of those, a training trajectory recommends the eval case's
> exact gold `product_id`. Scoring the same rollouts three ways:
>
> | split | base | student | lift |
> |---|---|---|---|
> | all 250 | 0.432 | 0.636 | +20.4 |
> | **clean 103** | **0.379** | **0.495** | **+11.6** |
> | contaminated 147 | 0.469 | 0.735 | +26.6 |
>
> Contamination nearly doubles the apparent lift. Step 01 of the notebook builds the clean split and
> drops the 5,679 contaminated trajectories from training.

## Quick Start

Open `distill-tool-call--Qwen--Qwen3.5-4B-shopping.ipynb` and run it top to bottom. It encodes the
trajectories, registers them as a dataset, and submits the training job:

```python
from sagemaker.train.sft_trainer import SFTTrainer
from sagemaker.train.common import TrainingType

trainer = SFTTrainer(
    model="huggingface-vlm-qwen3-5-4b",
    training_type=TrainingType.LORA,
    model_package_group=model_package_group,
    training_dataset=train_dataset.arn,
    validation_dataset=val_dataset.arn,
    s3_output_path="s3://<your-bucket>/output/",
    accept_eula=True,
)
trainer.hyperparameters.dataset_max_len = 16384
trainer.hyperparameters.lora_rank = 16
trainer.hyperparameters.lora_alpha = 128
trainer.train()
```

You do not pick an instance type, build a training image, or write a training script — SageMaker runs a
managed recipe and emits an already-merged, registered Model Package.

## Files

| File | Purpose |
|---|---|
| `distill-tool-call--Qwen--Qwen3.5-4B-shopping.ipynb` | End-to-end: data prep → SeqKD training → eval. |
| `evaluate-tool-call-accuracy.ipynb` | Base-vs-student tool-use ASR + IFEval portfolio, with charts. |
| `scripts/eval_lmeval.sh` | Non-thinking IFEval via lm-eval on a local vLLM (includes a `<think>` validity gate). |
| `scripts/shopping_asr_eval.sh` | Tool-use ASR wiring: vLLM serve → ShoppingBench ReAct rollout → ORM scoring (needs the ShoppingBench repo + search server). |
| 04c cell (in the notebook) | Per-trajectory LLM reasoning judge over 04b's saved rollouts — pairwise, blind, five process axes. No GPU, no new inference. |

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
