#!/usr/bin/env python
"""
Step 2 job — GRPO (RLVR) on top of the SFT model; upload the merged model to S3.

Launched by src/grpo.sh under `torchrun` (DDP). Every rank trains + generates
rollouts for its shard; only rank 0 merges + uploads.

Inputs:  --sft-model-s3 (from step 1) + --data-s3 (from step 0)
Outputs: --output-grpo-s3  (merged SFT+GRPO model)

--sft-model-s3 may also be a HF model id / local path for smoke tests (no download).
"""
from __future__ import annotations

import argparse
import os

import _shared
import common
from phases import grpo


def main():
    log = common.setup_logging()
    p = argparse.ArgumentParser(description=__doc__)
    _shared.add_pipeline_args(p)
    p.add_argument("--sft-model-s3", required=True, help="s3://.../sft_model (or HF id for smoke)")
    p.add_argument("--data-s3", required=True, help="s3://.../data (from step 0)")
    p.add_argument("--output-grpo-s3", required=True, help="s3://.../grpo_model")
    args = p.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if not args.output_grpo_s3.startswith("s3://"):
        raise SystemExit(f"--output-grpo-s3 must be an s3:// URI, got {args.output_grpo_s3!r}")

    common.assert_gpu_ready()
    common.resolve_hf_token()
    cfg = _shared.cfg_from_args(args)
    paths = common.local_paths()

    common.s3_download_prefix(args.data_s3, paths["data"])
    # Resolve the SFT model input into a local dir (or pass through a HF id).
    sft_model_path = common.resolve_model_input(args.sft_model_s3, paths["sft_model"])

    if rank == 0:
        log.info("STEP 2 (GRPO/RLVR): %s -> %s", sft_model_path, args.output_grpo_s3)
    grpo.run(cfg, paths, sft_model_path)

    if rank == 0:
        common.s3_upload_dir(paths["grpo_model"], args.output_grpo_s3)
        log.info("STEP 2 done. SFT+GRPO model at %s", args.output_grpo_s3)
    else:
        log.info("rank %d/%d done; rank 0 handles the upload.", rank, world_size)


if __name__ == "__main__":
    main()
