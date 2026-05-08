#!/usr/bin/env python3
"""
SAMA SageMaker Job MCP Server
Launch SFT and GRPO training jobs on SageMaker, and monitor their status.
"""

import io
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from typing import Any, Dict, Optional

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)

_orig = sys.stdout
sys.stdout = io.StringIO()
try:
    import boto3
    from botocore.exceptions import ClientError
except Exception:
    pass
finally:
    sys.stdout = _orig

from fastmcp import FastMCP

# Suppress sagemaker.config INFO messages that pollute stdout and break MCP transport
logging.getLogger("sagemaker.config").setLevel(logging.CRITICAL)
logging.getLogger("sagemaker").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

mcp = FastMCP("sama-sagemaker-job-mcp-server")


def _suppress_stdout():
    """Context manager that redirects fd 1 (real stdout) to /dev/null."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        fd = sys.__stdout__.fileno()
        old_fd = os.dup(fd)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, fd)
        os.close(devnull)
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            yield
        finally:
            sys.stdout = old_stdout
            os.dup2(old_fd, fd)
            os.close(old_fd)

    return _ctx()


def _import_sagemaker():
    """Lazy import sagemaker to avoid stdout pollution at module load."""
    # Silence the sagemaker.config logger which writes to stdout via basicConfig
    sm_config_logger = logging.getLogger("sagemaker.config")
    sm_config_logger.setLevel(logging.CRITICAL)

    _o = sys.stdout
    sys.stdout = io.StringIO()
    try:
        import sagemaker
        return sagemaker
    finally:
        sys.stdout = _o


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_session():
    sagemaker = _import_sagemaker()
    region = boto3.Session().region_name
    return sagemaker.Session(boto3.Session(region_name=region))


def _get_training_image(region: str, instance_type: str) -> str:
    sagemaker = _import_sagemaker()
    return sagemaker.image_uris.retrieve(
        framework="pytorch",
        region=region,
        version="2.8.0",
        instance_type=instance_type,
        image_scope="training",
    )


# ---------------------------------------------------------------------------
# Job Launcher
# ---------------------------------------------------------------------------

@mcp.tool()
def launch_sft_training_job(
    model_id: str,
    recipe_path: str,
    training_data_s3_uri: str,
    instance_type: str = "ml.g5.2xlarge",
    instance_count: int = 1,
    hf_token: Optional[str] = None,
    mlflow_tracking_arn: Optional[str] = None,
    source_dir: str = "../supervised_finetuning/sagemaker_code",
    volume_size_gb: int = 300,
    max_runtime_seconds: int = 18000,
) -> Dict[str, Any]:
    """
    Launch an SFT training job on SageMaker.

    This is the FINAL STEP of the SFT workflow. It creates a ModelTrainer
    and starts the job using the recipe YAML and training data on S3.

    Args:
        model_id: HuggingFace model identifier.
        recipe_path: Relative path to the recipe YAML (e.g. 'hf_recipes/Qwen/Qwen3-4B-vanilla-peft-qlora.yaml').
        training_data_s3_uri: S3 URI of the training dataset.
        instance_type: SageMaker instance type.
        instance_count: Number of instances.
        hf_token: HuggingFace token for gated models (optional).
        mlflow_tracking_arn: MLflow tracking server ARN (optional).
        source_dir: Path to the sagemaker_code directory.
        volume_size_gb: EBS volume size in GB.
        max_runtime_seconds: Max job runtime.

    Returns:
        Job name, status, and monitoring info.
    """
    try:
        sagemaker = _import_sagemaker()
        from sagemaker.modules.configs import (
            CheckpointConfig, Compute, OutputDataConfig,
            SourceCode, StoppingCondition,
        )
        from sagemaker.modules.configs import InputData
        from sagemaker.modules.train import ModelTrainer

        # Redirect stdout for the entire SageMaker interaction — the SDK
        # prints config warnings and progress to stdout which kills MCP transport.
        with _suppress_stdout():
            sess = _get_session()
            role = sagemaker.get_execution_role()
            region = sess.boto_region_name

            job_name = model_id.replace("/", "--").replace(".", "-")
            args = ["--config", recipe_path]

            env = {}
            if hf_token:
                env["HF_TOKEN"] = hf_token
            env["NCCL_DEBUG"] = "INFO"
            env["FI_EFA_USE_DEVICE_RDMA"] = "1"
            env["NCCL_SOCKET_IFNAME"] = "eth0"
            env["FI_PROVIDER"] = "efa"

            if mlflow_tracking_arn:
                env["MLFLOW_EXPERIMENT_NAME"] = f"{job_name}-exp"
                env["MLFLOW_TRACKING_URI"] = mlflow_tracking_arn
                env["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
                env["MLFLOW_TAGS"] = json.dumps({
                    "source.job": "sm-training-jobs",
                    "source.type": "sft",
                    "source.framework": "pytorch",
                })

            image_uri = _get_training_image(region, instance_type)
            base_job_name = f"{job_name}-sft"
            output_path = f"s3://{sess.default_bucket()}/{base_job_name}"

            trainer = ModelTrainer(
                training_image=image_uri,
                source_code=SourceCode(
                    source_dir=source_dir,
                    command=f"bash sm_accelerate_train.sh {' '.join(args)}",
                ),
                base_job_name=base_job_name,
                compute=Compute(
                    instance_type=instance_type,
                    instance_count=instance_count,
                    keep_alive_period_in_seconds=1800,
                    volume_size_in_gb=volume_size_gb,
                ),
                stopping_condition=StoppingCondition(max_runtime_in_seconds=max_runtime_seconds),
                output_data_config=OutputDataConfig(s3_output_path=output_path),
                checkpoint_config=CheckpointConfig(
                    s3_uri=os.path.join(output_path, job_name, "checkpoints"),
                    local_path="/opt/ml/checkpoints",
                ),
                role=role,
                environment=env,
            )

            trainer.train(
                input_data_config=[InputData(channel_name="training", data_source=training_data_s3_uri)],
                wait=False,
            )
            launched_job_name = trainer._latest_training_job.training_job_name

        return {
            "status": "success",
            "message": f"SFT training job launched for {model_id}",
            "job_name": launched_job_name,
            "base_job_name": base_job_name,
            "instance_type": instance_type,
            "training_data_s3_uri": training_data_s3_uri,
            "output_path": output_path,
            "next_step": "Use monitor_training_job with the base_job_name to track progress.",
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to launch SFT job: {e}"}


@mcp.tool()
def launch_grpo_training_job(
    model_id: str,
    recipe_path: str,
    training_data_s3_uri: str,
    tools_script: str = "tools_funcs/financial_tools_complex.py",
    reward_fn: str = "rewards/financial_tools_reward.py",
    zero_stage: int = 2,
    instance_type: str = "ml.g6e.12xlarge",
    instance_count: int = 1,
    hf_token: Optional[str] = None,
    mlflow_tracking_arn: Optional[str] = None,
    source_dir: str = "../preference_optimization/grpo_rlvr/sagemaker_code",
    volume_size_gb: int = 450,
    max_runtime_seconds: int = 36000,
) -> Dict[str, Any]:
    """
    Launch a GRPO RLVR training job on SageMaker.

    This is the FINAL STEP of the GRPO workflow. Uses sm_accelerate_grpo_train.sh
    with tools_script and reward_fn arguments.

    Args:
        model_id: HuggingFace model identifier.
        recipe_path: Relative path to the GRPO recipe YAML.
        training_data_s3_uri: S3 URI of the training dataset.
        tools_script: Path to tool functions script (relative to source_dir).
        reward_fn: Path to reward function script (relative to source_dir).
        zero_stage: DeepSpeed ZeRO stage (2 or 3, default 2).
        instance_type: SageMaker instance type.
        instance_count: Number of instances.
        hf_token: HuggingFace token (optional).
        mlflow_tracking_arn: MLflow tracking server ARN (optional).
        source_dir: Path to the grpo sagemaker_code directory.
        volume_size_gb: EBS volume size (default 450, GRPO needs more).
        max_runtime_seconds: Max runtime (default 36000, GRPO runs longer).

    Returns:
        Job name, status, and monitoring info.
    """
    try:
        sagemaker = _import_sagemaker()
        from sagemaker.modules.configs import (
            CheckpointConfig, Compute, OutputDataConfig,
            SourceCode, StoppingCondition,
        )
        from sagemaker.modules.configs import InputData
        from sagemaker.modules.train import ModelTrainer

        # Redirect stdout for the entire SageMaker interaction
        with _suppress_stdout():
            sess = _get_session()
            role = sagemaker.get_execution_role()
            region = sess.boto_region_name

            job_name = model_id.replace("/", "--").replace(".", "-")
            args = [
                "--config", recipe_path,
                "--tools_script", tools_script,
                "--reward_fn", reward_fn,
                "--zero_stage", str(zero_stage),
            ]

            env = {"NCCL_DEBUG": "INFO"}
            if hf_token:
                env["HF_TOKEN"] = hf_token
            if mlflow_tracking_arn:
                env["MLFLOW_EXPERIMENT_NAME"] = job_name
                env["MLFLOW_TRACKING_URI"] = mlflow_tracking_arn
                env["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
                env["MLFLOW_TAGS"] = json.dumps({
                    "source.job": "sm-training-jobs",
                    "source.type": "trl-grpo-rlvr",
                    "source.framework": "pytorch",
                })

            image_uri = _get_training_image(region, instance_type)
            base_job_name = f"{job_name}-grpo"
            output_path = f"s3://{sess.default_bucket()}/{base_job_name}"

            trainer = ModelTrainer(
                training_image=image_uri,
                source_code=SourceCode(
                    source_dir=source_dir,
                    command=f"bash sm_accelerate_grpo_train.sh {' '.join(args)}",
                ),
                base_job_name=base_job_name,
                compute=Compute(
                    instance_type=instance_type,
                    instance_count=instance_count,
                    keep_alive_period_in_seconds=1800,
                    volume_size_in_gb=volume_size_gb,
                ),
                stopping_condition=StoppingCondition(max_runtime_in_seconds=max_runtime_seconds),
                output_data_config=OutputDataConfig(s3_output_path=output_path),
                checkpoint_config=CheckpointConfig(
                    s3_uri=os.path.join(output_path, job_name, "checkpoints"),
                    local_path="/opt/ml/checkpoints",
                ),
                role=role,
                environment=env,
            )

            trainer.train(
                input_data_config=[InputData(channel_name="training", data_source=training_data_s3_uri)],
                wait=False,
            )
            launched_job_name = trainer._latest_training_job.training_job_name

        return {
            "status": "success",
            "message": f"GRPO training job launched for {model_id}",
            "job_name": launched_job_name,
            "base_job_name": base_job_name,
            "instance_type": instance_type,
            "tools_script": tools_script,
            "reward_fn": reward_fn,
            "output_path": output_path,
            "next_step": "Use monitor_training_job with the base_job_name to track progress.",
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to launch GRPO job: {e}"}


# ---------------------------------------------------------------------------
# Job Monitor
# ---------------------------------------------------------------------------

@mcp.tool()
def monitor_training_job(
    job_name: str,
    region: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Monitor the status of a SageMaker training job.

    Args:
        job_name: SageMaker training job name.
        region: AWS region (auto-detected if not provided).

    Returns:
        Job status, timing info, failure reason if applicable, model artifacts location.
    """
    try:
        if not region:
            region = boto3.Session().region_name or "us-east-1"

        client = boto3.client("sagemaker", region_name=region)
        resp = client.describe_training_job(TrainingJobName=job_name)

        result = {
            "status": "success",
            "job_name": job_name,
            "training_status": resp["TrainingJobStatus"],
            "creation_time": resp["CreationTime"].isoformat(),
        }

        if "TrainingStartTime" in resp:
            result["start_time"] = resp["TrainingStartTime"].isoformat()
        if "TrainingEndTime" in resp:
            result["end_time"] = resp["TrainingEndTime"].isoformat()
        if resp["TrainingJobStatus"] == "Failed" and "FailureReason" in resp:
            result["failure_reason"] = resp["FailureReason"]
        if resp["TrainingJobStatus"] == "Completed" and "ModelArtifacts" in resp:
            result["model_artifacts_s3"] = resp["ModelArtifacts"]["S3ModelArtifacts"]
        if "ResourceConfig" in resp:
            rc = resp["ResourceConfig"]
            result["instance_type"] = rc.get("InstanceType")
            result["instance_count"] = rc.get("InstanceCount")

        return result

    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "ValidationException":
            return {"status": "error", "message": f"Job '{job_name}' not found in region '{region}'."}
        return {"status": "error", "message": f"AWS error: {e}"}
    except Exception as e:
        return {"status": "error", "message": f"Monitor failed: {e}"}


@mcp.tool()
def list_recent_training_jobs(
    name_contains: Optional[str] = None,
    max_results: int = 10,
    region: Optional[str] = None,
) -> Dict[str, Any]:
    """
    List recent SageMaker training jobs, optionally filtered by name.

    Args:
        name_contains: Filter jobs whose name contains this string.
        max_results: Max jobs to return (default 10).
        region: AWS region.

    Returns:
        List of recent training jobs with name, status, and creation time.
    """
    try:
        if not region:
            region = boto3.Session().region_name or "us-east-1"

        client = boto3.client("sagemaker", region_name=region)
        kwargs = {
            "SortBy": "CreationTime",
            "SortOrder": "Descending",
            "MaxResults": max_results,
        }
        if name_contains:
            kwargs["NameContains"] = name_contains

        resp = client.list_training_jobs(**kwargs)
        jobs = []
        for j in resp.get("TrainingJobSummaries", []):
            jobs.append({
                "job_name": j["TrainingJobName"],
                "status": j["TrainingJobStatus"],
                "creation_time": j["CreationTime"].isoformat(),
            })

        return {"status": "success", "total": len(jobs), "jobs": jobs}
    except Exception as e:
        return {"status": "error", "message": f"Failed to list jobs: {e}"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    mcp.run(show_banner=False)

if __name__ == "__main__":
    main()
