# sagemaker_code/flops_meter.py
"""
FLOPs Meter for LLM finetuning according o EU AI act

This module provides comprehensive FLOPs (Floating Point Operations) tracking for ML training:
1. Analytical FLOPs calculation: F_ft ≈ (4*N_total + 2*N_trainable) * tokens_processed
2. Hardware-based upper bound via NVML GPU monitoring
3. Token counting across distributed training
4. Comparison with pretraining FLOPs

Output: flops_meter.json with metrics uploaded to S3 and stored in DynamoDB


See act details:
https://digital-strategy.ec.europa.eu/en/library/guidelines-scope-obligations-providers-general-purpose-ai-models-under-ai-act
https://ec.europa.eu/newsroom/dae/redirection/document/118340
"""
import json
import os
import time
import threading
import yaml
import pynvml
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

import torch
from torch import distributed as dist
import boto3
from transformers import (
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
)
from transformers import Trainer
from trl import SFTTrainer


# ---------- Shared FLOPs Calculation Utilities ----------


def calculate_finetuning_flops(
    N_total: int, N_trainable: int, tokens_processed: int
) -> float:
    """Calculate fine-tuning FLOPs using analytical formula.

    Formula: F_ft = (4*N_total + 2*N_trainable) * tokens_processed

    Breakdown:
    - Forward pass: 2*N_total (all parameters)
    - Backward pass gradient computation:
      - Gradients w.r.t. activations: 2*N_total (through ALL layers, even frozen)
      - Gradients w.r.t. weights: 2*N_trainable (only trainable parameters)

    For full fine-tuning (N_trainable = N_total):
    - F_ft = 4*N + 2*N = 6*N*D (matches EU AI Act formula C ≈ 6·P·D)

    Args:
        N_total: Total model parameters
        N_trainable: Trainable parameters
        tokens_processed: Non-padding tokens processed during training

    Returns:
        Fine-tuning FLOPs
    """
    return (4.0 * N_total + 2.0 * N_trainable) * float(tokens_processed)


def determine_compliance_threshold(pretrain_flops: Optional[float]) -> tuple:
    """Determine EU AI Act compliance threshold.

    Per EU AI Act guidelines:
    - If pretraining FLOPs unknown or < 10^23: use default 3.3×10^22 threshold
    - If pretraining FLOPs >= 10^23: use 30% of actual pretraining compute

    Args:
        pretrain_flops: Pretraining FLOPs (optional)

    Returns:
        Tuple of (threshold_value, threshold_type, use_default)
        - threshold_value: The threshold to compare against
        - threshold_type: "default_3.3e22" or "30pct_of_actual_pretraining"
        - use_default: Boolean indicating if default threshold is used
    """
    EU_AI_ACT_GPAI_THRESHOLD = 1e23  # 10^23 FLOPs - GPAI classification threshold
    EU_AI_ACT_DEFAULT_THRESHOLD = (
        3.3e22  # One-third of 10^23 - default when pretraining unknown
    )

    if (
        not pretrain_flops
        or pretrain_flops <= 0
        or pretrain_flops < EU_AI_ACT_GPAI_THRESHOLD
    ):
        return EU_AI_ACT_DEFAULT_THRESHOLD, "default_3.3e22", True
    else:
        return pretrain_flops, "30pct_of_actual_pretraining", False


# ---------- Helpers ----------


def is_dist():
    return dist.is_available() and dist.is_initialized()


def allreduce_scalar(value: float, op=dist.ReduceOp.SUM) -> float:
    if not is_dist():
        return float(value)
    t = torch.tensor(
        [value],
        dtype=torch.float64,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    dist.all_reduce(t, op=op)
    return float(t.item())


def get_rank():
    return dist.get_rank() if is_dist() else 0


def get_world_size():
    return dist.get_world_size() if is_dist() else 1


def now_s():
    return time.time()


def detect_gpu_peak_tflops(gpu_name: str) -> float:
    """Detect peak TFLOPS (FP16 tensor cores) based on GPU name.

    Args:
        gpu_name: GPU device name (e.g., "NVIDIA A100-SXM4-80GB")

    Returns:
        Peak TFLOPS for the GPU (FP16 with tensor cores)

    Fallback order:
    1. Known GPU patterns (H100, A100, L40, etc.)
    2. PEAK_TFLOPS_PER_GPU environment variable
    3. Conservative default of 150.0 TFLOPS
    """
    low = gpu_name.lower()
    if "h100" in low or "h800" in low:
        return 989.0  # H100 SXM
    elif "a100" in low or "a800" in low:
        return 312.0  # A100 SXM
    elif "l40" in low:
        return 181.05
    elif "l20" in low:
        return 119.5
    elif "h20" in low:
        return 148.0
    elif "910b" in low:
        return 354.0
    elif "a10" in low:
        return 125.0
    else:
        # Fallback: env var or conservative default
        env_peak = os.getenv("PEAK_TFLOPS_PER_GPU")
        if env_peak:
            try:
                return float(env_peak)
            except ValueError:
                pass
        return 150.0  # Conservative default for unknown GPUs


# ---------- NVML Sampler (hardware-based upper bound) ----------


class NvmlSampler:
    def __init__(self, sample_period_s: float = 1.0):
        self.sample_period_s = sample_period_s
        self._stop = threading.Event()
        self._thread = None
        self.samples = []  # list of dicts per sample
        self.start_ts = None
        self.end_ts = None

    def start(self):
        try:
            pynvml.nvmlInit()
        except Exception:
            return
        self.start_ts = now_s()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.end_ts = now_s()
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    def _run(self):
        device_count = pynvml.nvmlDeviceGetCount()
        while not self._stop.is_set():
            rec = {"ts": now_s(), "gpus": []}
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8")
                rec["gpus"].append(
                    {
                        "index": i,
                        "name": name,
                        "sm_util": util.gpu,
                        "mem_util": util.memory,
                    }
                )
            self.samples.append(rec)
            time.sleep(self.sample_period_s)

    def summarize(self) -> Dict[str, Any]:
        if not self.samples:
            return {"enabled": False}
        # Average SM util per GPU
        gpu_names = {}
        per_gpu_utils = {}
        counts = {}
        for rec in self.samples:
            for g in rec["gpus"]:
                idx = g["index"]
                per_gpu_utils[idx] = per_gpu_utils.get(idx, 0.0) + g["sm_util"]
                counts[idx] = counts.get(idx, 0) + 1
                gpu_names[idx] = g["name"]
        avg_utils = {
            idx: per_gpu_utils[idx] / counts[idx] / 100.0 for idx in per_gpu_utils
        }  # to fraction
        duration = (self.end_ts or now_s()) - (self.start_ts or now_s())
        # Auto-detect peak TFLOPS from GPU name
        peaks = {idx: detect_gpu_peak_tflops(name) for idx, name in gpu_names.items()}
        # Upper bound FLOPs = sum_gpus(peak * 1e12 * avg_util * duration)
        f_upper = 0.0
        per_gpu_upper = {}
        for idx, util in avg_utils.items():
            f = peaks[idx] * 1e12 * util * duration
            f_upper += f
            per_gpu_upper[idx] = {
                "name": gpu_names[idx],
                "avg_sm_util": util,
                "peak_tflops": peaks[idx],
                "flops_upper": f,
            }
        return {
            "enabled": True,
            "duration_s": duration,
            "per_gpu": per_gpu_upper,
            "F_upper": f_upper,
        }


# ---------- HF Trainer Callback (analytical) ----------


@dataclass
class FlopsAnalytics:
    N_total: int = 0
    N_trainable: int = 0
    tokens_processed: int = 0
    Flops_architecture: float = 0.0
    Flops_hardware: Optional[float] = None
    pct_of_pretrain: Optional[float] = None
    exceeds_30pct: Optional[bool] = None


class FlopsMeterCallback(TrainerCallback):
    """
    FLOPs Meter Callback for HuggingFace Trainer

    Tracks training FLOPs using two methods:
    1. Analytical: Flops_architecture ≈ (4*N_total + 2*N_trainable) * tokens_processed
       - 4*N_total: forward pass operations
       - 2*N_trainable: backward pass operations (gradient computation)

    2. Hardware upper bound: NVML-based GPU utilization monitoring
       - Flops_hardware = sum(peak_tflops * avg_utilization * duration)

    Output fields in flops_meter.json:
    - N_total, N_trainable: model parameter counts
    - tokens_processed: non-padding tokens processed
    - Flops_architecture: analytical FLOPs
    - Flops_hardware: hardware upper bound FLOPs
    - Flops_original: pretraining FLOPs for comparison
    - pct_of_pretrain: Flops_architecture / threshold ratio
    - exceeds_30pct: whether fine-tuning exceeds 30% of threshold
    - gpu_name, instance_type: hardware metadata
    - training_job_name, training_id: job identifiers

    Example output:
        {
        "Flops_architecture": "1.45e+13",
        "Flops_hardware": "1.52e+15",
        "Flops_original": "8.70e+22",
        "N_total": 1585294704,
        "N_trainable": 680094720,
        "threshold_type": "default_3.3e22",
        "threshold_value": "3.30e+22",
        "tokens_processed": 2150,
        }
    """

    def __init__(
        self,
        pad_token_id: int,
        pretrain_flops: Optional[float] = None,
        sample_nvml: bool = True,
        n_total: int = 0,
        n_trainable: int = 0,
        model_name: str = "",
        num_epochs: float = 0.0,
    ):
        self.pad_token_id = pad_token_id
        self.pretrain_flops = pretrain_flops
        self.sample_nvml = sample_nvml and (os.getenv("FLOPS_METER_NVML", "1") == "1")
        self.m = FlopsAnalytics()
        self.m.N_total = n_total
        self.m.N_trainable = n_trainable
        self.model_name = model_name
        self.num_epochs = num_epochs
        self._nvml = (
            NvmlSampler() if self.sample_nvml and torch.cuda.is_available() else None
        )
        self.start_time = None
        self.end_time = None
        self._last_inputs = None

    # Utility to sum tokens in a batch (supports dict or lists of dicts)
    def _count_tokens_in_batch(self, inputs) -> int:
        # Expect inputs["input_ids"] shape [bsz, seqlen]; support tuples/lists-of-dicts
        ids = (
            inputs.get("input_ids")
            if isinstance(inputs, dict)
            else (
                inputs[0].get("input_ids")
                if isinstance(inputs, (list, tuple)) and isinstance(inputs[0], dict)
                else None
            )
        )
        if isinstance(ids, torch.Tensor):
            # move to CPU for counting to avoid device mismatches in DDP
            return int((ids.detach().to("cpu") != self.pad_token_id).sum().item())
        # Fallback (huggingface datasets might deliver lists)
        if ids is not None:
            arr = torch.as_tensor(ids)
            return int((arr != self.pad_token_id).sum().item())
        return 0

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.start_time = time.time()
        if get_rank() == 0:
            print(
                f"[FLOPS] N_total={self.m.N_total}  N_trainable={self.m.N_trainable}"
            )
            print(
                f"[FLOPS DEBUG] on_train_begin called, pad_token_id={self.pad_token_id}"
            )

        if self._nvml:
            self._nvml.start()

    def on_substep_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # Called after each forward/backward pass (even during gradient accumulation)
        # Try to get inputs from kwargs
        inputs = kwargs.get("inputs")
        if inputs is not None:
            try:
                batch_tokens_local = self._count_tokens_in_batch(inputs)
                batch_tokens = (
                    allreduce_scalar(batch_tokens_local, op=dist.ReduceOp.SUM)
                    if is_dist()
                    else batch_tokens_local
                )
                self.m.tokens_processed += int(batch_tokens)
                if get_rank() == 0 and state.global_step <= 3:
                    print(
                        f"[FLOPS DEBUG] on_substep_end step={state.global_step}, "
                        f"batch_tokens={batch_tokens}, total={self.m.tokens_processed}"
                    )
            except Exception as e:
                if get_rank() == 0 and state.global_step <= 3:
                    print(f"[FLOPS WARN] token count failed: {e}")
        else:
            if get_rank() == 0 and state.global_step <= 3:
                print(
                    f"[FLOPS DEBUG] on_substep_end step={state.global_step}, NO INPUTS in kwargs"
                )
        return control

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # Debug logging
        if get_rank() == 0 and state.global_step <= 3:
            print(
                f"[FLOPS DEBUG] on_step_end step={state.global_step}, tokens_processed={self.m.tokens_processed}"
            )
        return control

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Called at the end of training - compute and save FLOPs metrics."""
        self.end_time = time.time()

        if get_rank() == 0:
            print("\n" + "=" * 60)
            print("FLOPs Meter - Final Calculation")
            print("=" * 60)

        # NVML stop & summarize (hardware upper bound)
        if self._nvml:
            self._nvml.stop()
            nvml_summary = self._nvml.summarize()
            self.m.Flops_hardware = nvml_summary.get("F_upper", None)
        # Analytical FLOPs using shared calculation
        self.m.Flops_architecture = calculate_finetuning_flops(
            self.m.N_total, self.m.N_trainable, self.m.tokens_processed
        )

        # Determine compliance threshold using shared logic
        comparison_threshold, threshold_type, use_default_threshold = (
            determine_compliance_threshold(self.pretrain_flops)
        )

        if get_rank() == 0:
            print("\nPretraining Comparison:")
            print(
                f"  Flops_original (pretraining): {self.pretrain_flops: .2e}"
                if self.pretrain_flops
                else "  Flops_original: Not provided"
            )
            print(f"  Flops_architecture: {self.m.Flops_architecture: .2e}")

            if use_default_threshold:
                print(
                    f"  Using EU AI Act default threshold: {comparison_threshold: .2e}"
                )
            else:
                print(
                    f"  Flops_original ({self.pretrain_flops: .2e}) >= GPAI threshold (1e23)"
                )
                print(
                    f"  Using 30% of actual pretraining per EU AI Act: {comparison_threshold * 0.30: .2e}"
                )

        if comparison_threshold and comparison_threshold > 0:
            if use_default_threshold:
                # For default threshold, compare directly (already represents 30% of 10^23)
                self.m.pct_of_pretrain = (
                    self.m.Flops_architecture / comparison_threshold
                )
                self.m.exceeds_30pct = bool(
                    self.m.Flops_architecture > comparison_threshold
                )
            else:
                # For actual pretraining, calculate percentage and compare to 30%
                self.m.pct_of_pretrain = (
                    self.m.Flops_architecture / comparison_threshold
                )
                self.m.exceeds_30pct = bool(self.m.pct_of_pretrain > 0.30)

            if get_rank() == 0:
                if use_default_threshold:
                    print(f"  Ratio: {self.m.pct_of_pretrain: .6%} of default threshold")
                    print(f"  Exceeds threshold: {self.m.exceeds_30pct}")
                else:
                    print(
                        f"  Ratio: {self.m.pct_of_pretrain: .6%} of actual pretraining"
                    )
                    print(f"  Exceeds 30% threshold: {self.m.exceeds_30pct}")
        else:
            self.m.pct_of_pretrain = None
            self.m.exceeds_30pct = None
            if get_rank() == 0:
                print("  Ratio: N/A (no valid threshold)")

        # Persist
        out = asdict(self.m)
        # Format Flops_architecture and Flops_hardware in scientific notation
        if out["Flops_architecture"] > 0:
            out["Flops_architecture"] = f"{out['Flops_architecture']: .2e}"
        if out["Flops_hardware"] is not None:
            out["Flops_hardware"] = f"{out['Flops_hardware']: .2e}"

        # Add metadata
        out["model_name"] = self.model_name
        out["num_epochs"] = self.num_epochs
        out["training_duration_seconds"] = (
            round(self.end_time - self.start_time, 2) if self.start_time else 0
        )
        out["gpu_size"] = get_world_size()
        out["gpu_rank"] = get_rank()
        out["Flops_original"] = (
            f"{self.pretrain_flops: .2e}" if self.pretrain_flops else None
        )
        out["threshold_type"] = threshold_type
        out["threshold_value"] = (
            f"{comparison_threshold: .2e}" if comparison_threshold else None
        )

        # Add job identifiers
        training_job_name = os.getenv("TRAINING_JOB_NAME", "")
        training_id = training_job_name.split("-")[-1] if training_job_name else ""
        out["training_job_name"] = training_job_name
        out["training_id"] = training_id

        # Add recipe config filename and full details
        # Read from hyperparameters.json (SageMaker training job config)
        recipe_config_path = ""
        hyperparams_path = "/opt/ml/input/config/hyperparameters.json"
        if os.path.exists(hyperparams_path):
            try:
                with open(hyperparams_path, "r") as f:
                    hyperparams = json.load(f)
                    recipe_config_path = hyperparams.get("config", "")
            except Exception as e:
                if get_rank() == 0:
                    print(f"  Warning: Failed to read hyperparameters.json: {e}")

        out["recipe_config"] = recipe_config_path

        # Load full recipe config YAML content
        recipe_config_details = None
        if recipe_config_path:
            try:
                # Strip surrounding quotes if present
                recipe_config_path = recipe_config_path.strip('"').strip("'")

                # Try to find the file in common locations
                possible_paths = [
                    os.path.join(
                        "/opt/ml/code", recipe_config_path
                    ),  # SageMaker code dir with full path
                    recipe_config_path,  # Direct path
                    os.path.join(os.getcwd(), recipe_config_path),  # Current dir
                ]

                for path in possible_paths:
                    if os.path.exists(path):
                        with open(path, "r") as f:
                            recipe_config_details = yaml.safe_load(f)
                        if get_rank() == 0:
                            print(f"  Recipe config loaded from: {path}")
                        break

                if recipe_config_details is None and get_rank() == 0:
                    print(
                        f"  Warning: Recipe config file not found: {recipe_config_path}"
                    )
                    print(f"  Searched paths: {possible_paths}")
            except Exception as e:
                if get_rank() == 0:
                    print(f"  Warning: Failed to load recipe config: {e}")

        out["recipe_config_details"] = recipe_config_details

        # Add GPU and instance info
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            out["gpu_name"] = gpu_name
            if get_rank() == 0:
                print("\nHardware Info:")
                print(f"  GPU: {gpu_name}")
                print(f"  Peak TFLOPS: {detect_gpu_peak_tflops(gpu_name): .2f}")
        else:
            out["gpu_name"] = None

        instance_type = os.getenv("TRAINING_INSTANCE_TYPE", None)
        out["instance_type"] = instance_type
        if get_rank() == 0 and instance_type:
            print(f"  Instance: {instance_type}")

        # Add model package group name for model registry traceability
        model_package_group_name = os.getenv("MODEL_PACKAGE_GROUP_NAME", None)
        out["model_package_group_name"] = model_package_group_name
        if get_rank() == 0 and model_package_group_name:
            print(f"  Model Package Group: {model_package_group_name}")

        # Write locally and to S3
        is_sagemaker = os.path.exists("/opt/ml/output") or "SM_MODEL_DIR" in os.environ
        default_path = (
            "/opt/ml/output/flops_meter.json" if is_sagemaker else "./flops_meter.json"
        )
        out_path = os.getenv("FLOPS_METER_OUT", default_path)

        if get_rank() == 0:
            # Save locally
            out_dir = os.path.dirname(out_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)

            print("\nOutput Summary:")
            print(f"  Tokens processed: {self.m.tokens_processed}")
            print(f"  Flops_architecture (analytical): {out['Flops_architecture']}")
            print(f"  Flops_hardware: {out['Flops_hardware']}")
            print(f"  Training duration: {out['training_duration_seconds']: .2f}s")
            print(f"  Local file: {out_path}")

            # Upload to S3 using training ID
            s3_base = os.getenv("TRAINING_PIPELINE_OUTPUT_S3_BASE")
            s3_uri = f"{s3_base}/{training_id}/evaluation_results"
            try:
                s3 = boto3.client("s3")
                s3_path = s3_uri.replace("s3://", "")
                bucket, key = s3_path.split("/", 1)
                s3_key = f"{key}/flops_meter.json"
                s3.put_object(Bucket=bucket, Key=s3_key, Body=json.dumps(out, indent=2))
                print(f"  S3 upload: s3: //{bucket}/{s3_key}")
            except Exception as e:
                print(f"  S3 upload failed: {e}")

            print("=" * 60 + "\n")

        return control

    # ---- helper used by TokenCountingTrainer ----
    def add_tokens(self, n: int):
        if n and n > 0:
            if is_dist():
                n = int(allreduce_scalar(float(n), op=dist.ReduceOp.SUM))
            self.m.tokens_processed += int(n)


# ---------- Trainer subclass that counts tokens reliably ----------


class TokenCountingTrainer(Trainer):
    """
    Counts non-pad tokens directly inside training_step (always has inputs here),
    then forwards to Trainer.training_step.
    """

    def training_step(self, model, inputs, num_items_in_batch=None):
        # Count tokens on CPU for safety
        ids = inputs.get("input_ids", None)
        if ids is not None:
            import torch

            if isinstance(ids, torch.Tensor):
                n = int(
                    (ids.detach().to("cpu") != getattr(self.args, "pad_token_id", -100))
                    .sum()
                    .item()
                )
            else:
                t = torch.tensor(ids)
                n = int((t != getattr(self.args, "pad_token_id", -100)).sum().item())
            # notify FLOPs callback if present
            for cb in self.callback_handler.callbacks:
                if hasattr(cb, "add_tokens"):
                    cb.add_tokens(n)
        return super().training_step(model, inputs, num_items_in_batch)


class TokenCountingSFTTrainer(SFTTrainer):
    """
    SFTTrainer that counts non-pad tokens directly inside training_step.
    """

    def training_step(self, model, inputs, num_items_in_batch=None):
        # Count tokens on CPU for safety
        ids = inputs.get("input_ids", None)
        if ids is not None:
            if isinstance(ids, torch.Tensor):
                n = int(
                    (ids.detach().to("cpu") != getattr(self.args, "pad_token_id", -100))
                    .sum()
                    .item()
                )
            else:
                t = torch.tensor(ids)
                n = int((t != getattr(self.args, "pad_token_id", -100)).sum().item())
            # notify FLOPs callback if present
            for cb in self.callback_handler.callbacks:
                if hasattr(cb, "add_tokens"):
                    cb.add_tokens(n)
        return super().training_step(model, inputs, num_items_in_batch)
