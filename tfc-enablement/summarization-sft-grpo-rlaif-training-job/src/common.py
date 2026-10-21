"""Shared helpers for the summarization SFT -> GRPO (RLAIF) jobs.

Logging, GPU/attn checks, HF token resolution, raw S3 sync (no tar), DDP rank
helpers, and a version-tolerant TRL config builder. Generic — nothing summarization-
specific lives here (that is in data_utils.py and judge_reward.py).
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys

LOG = logging.getLogger("summ-job")


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    return LOG


def is_sagemaker() -> bool:
    return "SM_MODEL_DIR" in os.environ or os.path.isdir("/opt/ml/input")


# ---------------------------------------------------------------------------
# Distributed / device
# ---------------------------------------------------------------------------
def dist_info() -> tuple[int, int, int]:
    """(local_rank, rank, world_size). local_rank is -1 when not distributed."""
    return (
        int(os.environ.get("LOCAL_RANK", -1)),
        int(os.environ.get("RANK", 0)),
        int(os.environ.get("WORLD_SIZE", 1)),
    )


def barrier() -> None:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except Exception:  # noqa: BLE001
        pass


def assert_gpu_ready() -> None:
    """Fail fast on a container whose torch can't see the GPUs (before any download)."""
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA not available in this container. Training needs a GPU; check the "
            "instance type and that the image ships a CUDA-enabled torch."
        )
    LOG.info("CUDA ready: %d GPU(s) visible.", torch.cuda.device_count())


def cuda_device_map(local_rank: int | None = None):
    """device_map pinning the model to this rank's GPU (or None on CPU)."""
    import torch

    if not torch.cuda.is_available():
        return None
    if local_rank is None:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank < 0:
        local_rank = 0
    torch.cuda.set_device(local_rank)
    LOG.info("pinning model load to cuda:%d", local_rank)
    return {"": local_rank}


def flash_attn_available() -> bool:
    import torch

    if not torch.cuda.is_available():
        return False
    try:
        import flash_attn  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def resolve_attn_impl(requested: str, on_cpu: bool = False) -> str:
    """Return an attention impl that will actually load on this device."""
    if requested != "flash_attention_2":
        return requested
    if on_cpu:
        LOG.info("CPU-side load: using 'eager' instead of flash_attention_2.")
        return "eager"
    if not flash_attn_available():
        LOG.warning("flash_attention_2 requested but unavailable — falling back to 'sdpa'.")
        return "sdpa"
    return "flash_attention_2"


# ---------------------------------------------------------------------------
# Hugging Face token
# ---------------------------------------------------------------------------
def aws_region() -> str | None:
    for var in ("AWS_REGION", "AWS_DEFAULT_REGION"):
        if os.environ.get(var):
            return os.environ[var]
    try:
        import boto3

        r = boto3.session.Session().region_name
        if r:
            return r
    except Exception:  # noqa: BLE001
        pass
    return None


def _apply_hf_token(token: str) -> str:
    token = token.strip()
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    try:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
        LOG.info("Logged in to the Hugging Face Hub.")
    except Exception as e:  # noqa: BLE001
        LOG.warning("huggingface_hub.login() failed (%s); env vars are set.", e)
    return token


def resolve_hf_token() -> str | None:
    """env HF_TOKEN/HUGGING_FACE_HUB_TOKEN -> Secrets Manager (HF_TOKEN_SECRET_NAME)."""
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(var):
            return _apply_hf_token(os.environ[var])

    secret_name = os.environ.get("HF_TOKEN_SECRET_NAME")
    if not secret_name:
        LOG.info("No HF token configured; assuming public model access.")
        return None
    try:
        import boto3

        client = boto3.client("secretsmanager", region_name=aws_region())
        token = (client.get_secret_value(SecretId=secret_name).get("SecretString") or "").strip()
        if token.startswith("{"):
            import json

            data = json.loads(token)
            token = (data.get("HF_TOKEN") or data.get("token") or "").strip()
        if not token:
            LOG.warning("Secret '%s' had no usable token.", secret_name)
            return None
        LOG.info("Loaded HF token from Secrets Manager secret '%s'.", secret_name)
        return _apply_hf_token(token)
    except Exception as e:  # noqa: BLE001
        LOG.warning("Could not load HF token from Secrets Manager (%s): %s [region=%s].",
                    secret_name, e, aws_region())
        return None


# ---------------------------------------------------------------------------
# Local scratch paths + raw S3 sync
# ---------------------------------------------------------------------------
def local_paths(work_dir: str | None = None) -> dict:
    if work_dir is None:
        work_dir = "/opt/ml/summ_work" if is_sagemaker() else "./job_out"
    paths = {
        "work": work_dir,
        "data": os.path.join(work_dir, "data"),
        "sft_model": os.path.join(work_dir, "sft_model"),
        "grpo_model": os.path.join(work_dir, "grpo_model"),
        "checkpoints": os.path.join(work_dir, "checkpoints"),
        "results": os.path.join(work_dir, "results"),
    }
    for key in paths:
        os.makedirs(paths[key], exist_ok=True)
    return paths


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"not an s3 uri: {uri}")
    bucket, _, key = uri[len("s3://"):].partition("/")
    return bucket, key.strip("/")


def s3_upload_dir(local_dir: str, s3_uri: str) -> None:
    import boto3

    s3 = boto3.client("s3", region_name=aws_region())
    bucket, prefix = parse_s3_uri(s3_uri)
    n = 0
    for root, _, files in os.walk(local_dir):
        for fn in files:
            lp = os.path.join(root, fn)
            rel = os.path.relpath(lp, local_dir)
            key = f"{prefix}/{rel}" if prefix else rel
            s3.upload_file(lp, bucket, key)
            n += 1
    LOG.info("uploaded %d files: %s -> %s", n, local_dir, s3_uri)


def s3_download_prefix(s3_uri: str, local_dir: str) -> None:
    """Recursively download everything under s3_uri -> local_dir (exact-dir match)."""
    import boto3

    s3 = boto3.client("s3", region_name=aws_region())
    bucket, prefix = parse_s3_uri(s3_uri)
    # Trailing slash => "a/b" lists only "a/b/*", never the sibling "a/b_old/*".
    list_prefix = prefix.rstrip("/") + "/" if prefix else ""
    os.makedirs(local_dir, exist_ok=True)
    paginator = s3.get_paginator("list_objects_v2")
    n = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=list_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(list_prefix):]
            if not rel or rel.endswith("/"):
                continue
            dest = os.path.join(local_dir, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            s3.download_file(bucket, key, dest)
            n += 1
    LOG.info("downloaded %d files: %s -> %s", n, s3_uri, local_dir)


def resolve_model_input(uri_or_id: str, local_dir: str) -> str:
    """S3 prefix -> download and return local_dir; anything else -> return unchanged
    (HF repo id or local path), which enables smoke-testing against a stock model."""
    if uri_or_id.startswith("s3://"):
        s3_download_prefix(uri_or_id, local_dir)
        return local_dir
    LOG.info("'%s' is not an S3 URI — treating as a model id/path (no download).", uri_or_id)
    return uri_or_id


# ---------------------------------------------------------------------------
# TRL config compatibility — keep only kwargs the installed version accepts,
# remap known renames, drop the rest with a log (instead of a bare TypeError
# that costs a full model re-download to discover).
# ---------------------------------------------------------------------------
_ALIASES = {
    "max_seq_length": ("max_length", "max_seq_len"),
    "max_length": ("max_seq_length",),
    "eval_strategy": ("evaluation_strategy",),
    "evaluation_strategy": ("eval_strategy",),
}


def _accepted_fields(cls) -> set | None:
    import inspect

    names: set = set()
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return None
    for pname, p in sig.parameters.items():
        if pname == "self":
            continue
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            return None
        names.add(pname)
    for attr in ("__dataclass_fields__", "model_fields"):
        container = getattr(cls, attr, None)
        if isinstance(container, dict):
            names |= set(container)
    return names


def make_trl_config(cls, desired: dict):
    """Build a TRL config (SFTConfig/GRPOConfig) keeping only supported kwargs."""
    try:
        import trl

        LOG.info("trl version: %s", getattr(trl, "__version__", "unknown"))
    except Exception:  # noqa: BLE001
        pass

    accepted = _accepted_fields(cls)
    if accepted is None:
        return cls(**desired)

    kept, renamed, dropped = {}, {}, []
    for key, value in desired.items():
        if key in accepted:
            kept[key] = value
            continue
        alias = next((a for a in _ALIASES.get(key, ()) if a in accepted), None)
        if alias:
            kept[alias] = value
            renamed[key] = alias
        else:
            dropped.append(key)
    if renamed:
        LOG.info("%s: remapped %s", cls.__name__, renamed)
    if dropped:
        LOG.warning("%s does not accept %s in this TRL build — dropping.",
                    cls.__name__, sorted(dropped))
    return cls(**kept)


def run(cmd: list[str], env: dict | None = None, check: bool = True) -> int:
    LOG.info("exec: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, env={**os.environ, **(env or {})})
    proc.wait()
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc.returncode
