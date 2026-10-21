#!/usr/bin/env python
"""
Step 1 job — SFT the base model on tool-call format; upload the merged model to S3.

Launched by src/sft.sh, which wraps this in `torchrun` for multi-GPU DDP. Every
rank trains; only rank 0 merges + uploads.

Inputs:  --data-s3 (from step 0) + base model (HF)
Outputs: --output-sft-s3  (merged SFT model, raw files)
"""
from __future__ import annotations

import argparse
import os

import _shared
import common
from phases import sft


def main():
    log = common.setup_logging()
    p = argparse.ArgumentParser(description=__doc__)
    _shared.add_pipeline_args(p)
    p.add_argument("--data-s3", required=True, help="s3://.../data (from step 0)")
    p.add_argument("--output-sft-s3", required=True, help="s3://.../sft_model")
    args = p.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if not args.output_sft_s3.startswith("s3://"):
        raise SystemExit(f"--output-sft-s3 must be an s3:// URI, got {args.output_sft_s3!r}")

    common.assert_gpu_ready()
    common.resolve_hf_token()
    cfg = _shared.cfg_from_args(args)
    paths = common.local_paths()

    # Every rank downloads the (small) data splits.
    common.s3_download_prefix(args.data_s3, paths["data"])

    if rank == 0:
        log.info("STEP 1 (SFT): base=%s -> %s", cfg["base_model"], args.output_sft_s3)
    sft.run(cfg, paths)

    if rank == 0:
        common.s3_upload_dir(paths["sft_model"], args.output_sft_s3)
        log.info("STEP 1 done. SFT model at %s", args.output_sft_s3)
    else:
        log.info("rank %d/%d done; rank 0 handles the upload.", rank, world_size)


if __name__ == "__main__":
    main()
