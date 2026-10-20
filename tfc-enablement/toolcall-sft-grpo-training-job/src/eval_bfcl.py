#!/usr/bin/env python
"""
Step 6 — BFCL (Berkeley Function Calling Leaderboard) on base / SFT / SFT+GRPO.

This is the EXTERNAL, standard tool-calling benchmark — the credible complement to our
in-distribution xLAM verifier. We run the single-turn AST categories plus `irrelevance`
(the over-calling / abstention check our xLAM eval lacks).

Launched by src/eval_bfcl.sh. Needs a GPU (BFCL hosts the model via vLLM).

IMPORTANT, PLEASE READ — this is a SCAFFOLD, not turnkey:
  * BFCL (`bfcl-eval`) CLI flags + category names change across releases. The two
    `bfcl ...` commands in run_bfcl() use the common form; confirm against
    `bfcl --help` on your installed version and adjust if needed.
  * FORMAT ALIGNMENT: BFCL formats tool schemas and parses the model's output using a
    per-model *handler*. We pass --model = the BASE Qwen id so BFCL uses the Qwen
    handler/prompt format (which matches Qwen's native <tool_call> format our SFT used),
    while --local-model-path points at the actual checkpoint. Use the SAME handler for
    all three models so the comparison is apples-to-apples.
  * We copy BFCL's raw score files into the results dir and upload them even if the
    accuracy parse below misses a key, so nothing is lost — inspect them if a number
    looks off.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

import common

LOG = None


def run_bfcl(model_name, local_path, categories, gen_dir):
    """Generate + evaluate one model on the given BFCL categories.

    Version-sensitive: adjust flags to your installed `bfcl-eval` if these error.
    """
    # Generate model responses (BFCL hosts `local_path` with vLLM, formats prompts via
    # the handler keyed to `model_name`).
    common.run([
        "bfcl", "generate",
        "--model", model_name,
        "--test-category", categories,
        "--backend", "vllm",
        "--local-model-path", local_path,
        "--num-gpus", "1",
        "--result-dir", gen_dir,
    ])
    # Score them.
    common.run([
        "bfcl", "evaluate",
        "--model", model_name,
        "--test-category", categories,
        "--result-dir", gen_dir,
        "--score-dir", os.path.join(gen_dir, "score"),
    ])


def parse_bfcl(score_dir):
    """Best-effort: pull an overall accuracy from BFCL score JSONs."""
    out = {}
    for f in glob.glob(os.path.join(score_dir, "**", "*.json"), recursive=True):
        try:
            data = json.load(open(f))
        except Exception:  # noqa: BLE001
            continue
        for key in ("accuracy", "overall_accuracy", "acc"):
            if isinstance(data, dict) and isinstance(data.get(key), (int, float)):
                out[os.path.basename(f)] = float(data[key])
    return out or {"note": "no accuracy key auto-parsed — inspect raw score files in S3"}


def main():
    global LOG
    LOG = common.setup_logging()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", required=True, help="also the BFCL handler/model name")
    p.add_argument("--sft-model-s3", required=True)
    p.add_argument("--grpo-model-s3", required=True)
    p.add_argument("--categories", default="simple,multiple,parallel,parallel_multiple,irrelevance")
    p.add_argument("--eval-base", type=int, default=1)
    p.add_argument("--output-results-s3", required=True)
    args = p.parse_args()

    if not args.output_results_s3.startswith("s3://"):
        raise SystemExit(f"--output-results-s3 must be s3://, got {args.output_results_s3!r}")

    common.assert_gpu_ready()
    common.resolve_hf_token()
    paths = common.local_paths()
    bfcl_root = os.path.join(paths["results"], "bfcl")

    specs = []
    if args.eval_base:
        specs.append(("base", args.base_model))
    specs.append(("SFT", common.resolve_model_input(args.sft_model_s3, paths["sft_model"])))
    specs.append(("SFT+GRPO", common.resolve_model_input(args.grpo_model_s3, paths["grpo_model"])))

    results = []
    for label, path in specs:
        gen_dir = os.path.join(bfcl_root, label.replace("+", "_"))
        os.makedirs(gen_dir, exist_ok=True)
        LOG.info("BFCL: %s (%s) categories=%s", label, path, args.categories)
        try:
            run_bfcl(args.base_model, path, args.categories, gen_dir)
            score = parse_bfcl(os.path.join(gen_dir, "score"))
        except Exception as e:  # noqa: BLE001
            LOG.exception("BFCL failed for %s", label)
            score = {"error": str(e)}
        results.append({"model": label, **score})
        LOG.info("BFCL %s = %s", label, score)

    json.dump(results, open(os.path.join(paths["results"], "bfcl_summary.json"), "w"), indent=2)
    common.s3_upload_dir(paths["results"], args.output_results_s3)
    for r in results:
        LOG.info("RESULT %s", r)
    LOG.info("STEP 6 done -> %s", args.output_results_s3)


if __name__ == "__main__":
    main()
