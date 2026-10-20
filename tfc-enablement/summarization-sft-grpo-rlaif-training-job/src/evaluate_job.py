#!/usr/bin/env python
"""
Step 3 job — evaluate base vs SFT vs SFT+GRPO on the held-out split; upload results.

Single-process (one GPU). Launched by src/evaluate.sh. Uses the EVAL judge
(--eval-judge-model), which MUST differ from the training judge.

Inputs:  --data-s3 (eval split) + --sft-model-s3 + --grpo-model-s3 (+ base model HF)
Outputs: --output-results-s3  (metrics.json/csv, comparison.png, samples.json)
"""
from __future__ import annotations

import argparse

import _shared
import common
from phases import evaluate


def main():
    log = common.setup_logging()
    p = argparse.ArgumentParser(description=__doc__)
    _shared.add_pipeline_args(p)
    p.add_argument("--data-s3", required=True, help="s3://.../data (from step 0)")
    p.add_argument("--sft-model-s3", required=True, help="s3://.../sft_model (or HF id)")
    p.add_argument("--grpo-model-s3", required=True, help="s3://.../grpo_model (or HF id)")
    p.add_argument("--output-results-s3", required=True, help="s3://.../results")
    args = p.parse_args()

    if not args.output_results_s3.startswith("s3://"):
        raise SystemExit(f"--output-results-s3 must be an s3:// URI, got {args.output_results_s3!r}")

    common.assert_gpu_ready()
    common.resolve_hf_token()
    cfg = _shared.cfg_from_args(args)
    paths = common.local_paths()

    common.s3_download_prefix(args.data_s3, paths["data"])
    sft_path = common.resolve_model_input(args.sft_model_s3, paths["sft_model"])
    grpo_path = common.resolve_model_input(args.grpo_model_s3, paths["grpo_model"])

    model_specs = []
    if cfg["eval_base"]:
        model_specs.append(("base", cfg["base_model"]))
    model_specs.append(("SFT", sft_path))
    model_specs.append(("SFT+GRPO", grpo_path))

    log.info("STEP 3 (evaluate): %s", [m[0] for m in model_specs])
    evaluate.run(cfg, paths, model_specs)
    common.s3_upload_dir(paths["results"], args.output_results_s3)
    log.info("STEP 3 done. Results at %s", args.output_results_s3)


if __name__ == "__main__":
    main()
