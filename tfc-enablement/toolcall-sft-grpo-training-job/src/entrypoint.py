#!/usr/bin/env python
"""
Combined entrypoint — run the WHOLE pipeline (prepare -> SFT -> GRPO -> evaluate)
inside ONE SageMaker training job on a single multi-GPU instance.

Used by ../launch_job.py. It shells out to the same step wrappers the per-step
notebooks use (prepare.sh / sft.sh / grpo.sh / evaluate.sh), so DDP (torchrun) and
all step logic have a single source of truth. Artifacts move between steps through
S3 under one --output-s3 prefix.

For the per-step pipeline (each phase its own right-sized job), use the notebooks
in ../notebooks/ instead.

Hyperparameters arrive from ModelTrainer as --key value CLI args.
"""
from __future__ import annotations

import argparse
import os

import _shared
import common

STEP_ENV_KEYS = (
    "BASE_MODEL", "XLAM_DATASET", "N_TRAIN", "N_GRPO", "N_EVAL", "SEQ_LEN", "SEED",
    "ENABLE_THINKING", "ATTN_IMPL", "USE_LORA", "LORA_R", "LORA_ALPHA",
    "SFT_EPOCHS", "SFT_LR", "SFT_BS", "SFT_GRAD_ACCUM",
    "GRPO_EPOCHS", "GRPO_LR", "GRPO_BS", "GRPO_GRAD_ACCUM", "NUM_GENERATIONS",
    "GRPO_BETA", "GRPO_TEMPERATURE", "MAX_COMPLETION_LENGTH", "REWARD_WEIGHTS",
    "USE_VLLM", "EVAL_BASE", "EVAL_MAX_NEW_TOKENS", "EVAL_BATCH_SIZE",
)


def _args_to_env(args) -> dict:
    return {
        "BASE_MODEL": args.base_model,
        "XLAM_DATASET": args.xlam_dataset,
        "N_TRAIN": str(args.n_train),
        "N_GRPO": str(args.n_grpo),
        "N_EVAL": str(args.n_eval),
        "SEQ_LEN": str(args.seq_len),
        "SEED": str(args.seed),
        "ENABLE_THINKING": str(args.enable_thinking),
        "ATTN_IMPL": args.attn_impl,
        "USE_LORA": str(args.use_lora),
        "LORA_R": str(args.lora_r),
        "LORA_ALPHA": str(args.lora_alpha),
        "SFT_EPOCHS": str(args.sft_epochs),
        "SFT_LR": str(args.sft_lr),
        "SFT_BS": str(args.sft_bs),
        "SFT_GRAD_ACCUM": str(args.sft_grad_accum),
        "GRPO_EPOCHS": str(args.grpo_epochs),
        "GRPO_LR": str(args.grpo_lr),
        "GRPO_BS": str(args.grpo_bs),
        "GRPO_GRAD_ACCUM": str(args.grpo_grad_accum),
        "NUM_GENERATIONS": str(args.num_generations),
        "GRPO_BETA": str(args.grpo_beta),
        "GRPO_TEMPERATURE": str(args.grpo_temperature),
        "MAX_COMPLETION_LENGTH": str(args.max_completion_length),
        "REWARD_WEIGHTS": args.reward_weights,
        "USE_VLLM": str(args.use_vllm),
        "EVAL_BASE": str(args.eval_base),
        "EVAL_MAX_NEW_TOKENS": str(args.eval_max_new_tokens),
        "EVAL_BATCH_SIZE": str(args.eval_batch_size),
    }


def main():
    log = common.setup_logging()
    p = argparse.ArgumentParser(description=__doc__)
    _shared.add_pipeline_args(p)
    p.add_argument("--output-s3", required=True,
                   help="s3://.../run_xyz base prefix; substeps write data/, sft_model/, "
                        "grpo_model/, results/ underneath it")
    p.add_argument("--skip-eval", type=int, default=0, choices=[0, 1])
    args = p.parse_args()

    if not args.output_s3.startswith("s3://"):
        raise SystemExit(f"--output-s3 must be an s3:// URI, got {args.output_s3!r}")

    base = args.output_s3.rstrip("/")
    data_s3 = f"{base}/data"
    sft_s3 = f"{base}/sft_model"
    grpo_s3 = f"{base}/grpo_model"
    results_s3 = f"{base}/results"

    here = os.path.dirname(os.path.abspath(__file__))
    env = {**os.environ, **_args_to_env(args)}

    def step(script, extra):
        common.run(["bash", os.path.join(here, script)], env={**env, **extra})

    log.info("PIPELINE start -> %s", base)
    step("prepare.sh", {"OUTPUT_DATA_S3": data_s3})
    step("sft.sh", {"DATA_S3": data_s3, "OUTPUT_SFT_S3": sft_s3})
    step("grpo.sh", {"SFT_MODEL_S3": sft_s3, "DATA_S3": data_s3, "OUTPUT_GRPO_S3": grpo_s3})
    if not args.skip_eval:
        step("evaluate.sh", {
            "DATA_S3": data_s3, "SFT_MODEL_S3": sft_s3,
            "GRPO_MODEL_S3": grpo_s3, "OUTPUT_RESULTS_S3": results_s3,
        })
    log.info("PIPELINE done. Artifacts under %s", base)


if __name__ == "__main__":
    main()
