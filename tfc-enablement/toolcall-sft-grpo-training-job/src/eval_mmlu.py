#!/usr/bin/env python
"""
Step 5 — MMLU regression check via LightEval on base / SFT / SFT+GRPO.

This is NOT the tool-calling value metric — MMLU is multiple-choice world knowledge.
Its purpose is a *regression check*: did tool-calling fine-tuning degrade the model's
general knowledge? You want SFT/GRPO to be roughly on par with base here.

Launched by src/eval_mmlu.sh. Single process, one GPU.

NOTE: LightEval's CLI + task-string format vary across versions. The command in
`run_lighteval` uses the widely-used `accelerate` form; if your installed lighteval
rejects a flag or the task string, adjust MMLU_TASKS / the flags there — the rest of
this orchestration (download, loop, parse, upload) is version-independent.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import common

LOG = None


def run_lighteval(model_path, tasks, out_dir, batch_size):
    os.makedirs(out_dir, exist_ok=True)
    # Version-sensitive line — adjust to your installed lighteval if it errors.
    cmd = [
        "lighteval", "accelerate",
        "--model_args", f"pretrained={model_path},dtype=bfloat16,trust_remote_code=True",
        "--tasks", tasks,
        "--output_dir", out_dir,
    ]
    if batch_size:
        cmd += ["--override_batch_size", str(batch_size)]
    common.run(cmd)


def parse_mmlu(out_dir):
    """Find any results*.json under out_dir and average the MMLU accuracy fields."""
    accs = []
    src = None
    for f in glob.glob(os.path.join(out_dir, "**", "results*.json"), recursive=True):
        try:
            data = json.load(open(f))
        except Exception:  # noqa: BLE001
            continue
        for k, v in (data.get("results") or {}).items():
            if "mmlu" in k.lower() and isinstance(v, dict):
                for mk in ("acc", "acc_norm", "accuracy"):
                    if isinstance(v.get(mk), (int, float)):
                        accs.append(float(v[mk]))
                        src = f
                        break
    return ({"mmlu_acc": sum(accs) / len(accs), "n_keys": len(accs), "source": src}
            if accs else {"mmlu_acc": None, "note": "no mmlu key found — inspect raw results"})


def main():
    global LOG
    LOG = common.setup_logging()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", required=True)
    p.add_argument("--sft-model-s3", required=True)
    p.add_argument("--grpo-model-s3", required=True)
    p.add_argument("--tasks", default="leaderboard|mmlu|5|0")
    p.add_argument("--eval-base", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--output-results-s3", required=True)
    args = p.parse_args()

    if not args.output_results_s3.startswith("s3://"):
        raise SystemExit(f"--output-results-s3 must be s3://, got {args.output_results_s3!r}")

    common.assert_gpu_ready()
    common.resolve_hf_token()
    paths = common.local_paths()

    specs = []
    if args.eval_base:
        specs.append(("base", args.base_model))
    specs.append(("SFT", common.resolve_model_input(args.sft_model_s3, paths["sft_model"])))
    specs.append(("SFT+GRPO", common.resolve_model_input(args.grpo_model_s3, paths["grpo_model"])))

    results = []
    for label, path in specs:
        out_dir = os.path.join(paths["results"], "mmlu", label.replace("+", "_"))
        LOG.info("MMLU: %s (%s) -> %s", label, path, args.tasks)
        run_lighteval(path, args.tasks, out_dir, args.batch_size)
        score = parse_mmlu(out_dir)
        results.append({"model": label, **score})
        LOG.info("MMLU %s = %s", label, score)

    json.dump(results, open(os.path.join(paths["results"], "mmlu_summary.json"), "w"), indent=2)
    common.s3_upload_dir(paths["results"], args.output_results_s3)
    for r in results:
        LOG.info("RESULT %s", r)
    LOG.info("STEP 5 done -> %s", args.output_results_s3)


if __name__ == "__main__":
    main()
