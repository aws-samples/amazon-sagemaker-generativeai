#!/usr/bin/env python
"""
Step 0 job — prepare data, upload the JSONL splits to S3.

Inputs:  base model tokenizer + xLAM dataset (pulled from HF at runtime)
Outputs: --output-data-s3   (sft_train/sft_val/grpo_train/eval .jsonl)

CPU-only; runs fine on a small instance or locally.
"""
from __future__ import annotations

import argparse
import os

import _shared
import common
from phases import prepare


def main():
    log = common.setup_logging()
    p = argparse.ArgumentParser(description=__doc__)
    _shared.add_pipeline_args(p)
    p.add_argument("--output-data-s3", required=True, help="s3://.../data")
    args = p.parse_args()

    if not args.output_data_s3.startswith("s3://"):
        raise SystemExit(f"--output-data-s3 must be an s3:// URI, got {args.output_data_s3!r}")

    common.resolve_hf_token()
    cfg = _shared.cfg_from_args(args)
    paths = common.local_paths()

    log.info("STEP 0 (prepare): %s -> %s", cfg["xlam_dataset"], args.output_data_s3)
    prepare.run(cfg, paths)
    common.s3_upload_dir(paths["data"], args.output_data_s3)
    log.info("STEP 0 done. Data at %s", args.output_data_s3)


if __name__ == "__main__":
    main()
