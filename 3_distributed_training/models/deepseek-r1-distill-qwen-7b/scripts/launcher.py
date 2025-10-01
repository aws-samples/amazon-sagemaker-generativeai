# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ray distributed workload launcher for SageMaker training jobs.

This script serves as an entrypoint for SageMaker training jobs and handles both
single-node and multi-node distributed workload scenarios using Ray.

Supports both Python (.py) and Bash (.sh) scripts as entrypoints.
"""
from __future__ import absolute_import
import argparse
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import importlib
import logging
import os
import requests
import sagemaker_training.environment
import shlex
import signal
import subprocess
import sys
import time
import tarfile
from typing import Dict, List, Optional, Any, Tuple
import ray


# Configure logger
def get_logger():
    """Get configured logger for the launcher."""
    logger = logging.getLogger(__name__)

    # Prevent duplicate handlers in distributed environments
    if logger.handlers:
        return logger

    # Only add handler if none exist and we haven't already configured this logger
    if not hasattr(logger, "_configured"):
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        # Mark this logger as configured to prevent re-configuration
        logger._configured = True

        # Prevent propagation to avoid duplicate messages from parent loggers
        logger.propagate = False

    return logger


logger = get_logger()

# Exit codes
SUCCESS_EXIT_CODE = 0
DEFAULT_FAILURE_CODE = 1

# Ray configuration constants
DEFAULT_RAY_PORT = 6379
RAY_WORKER_POLL_INTERVAL = 10  # seconds
RAY_CONNECTION_TIMEOUT = 300  # seconds (5 minutes)

# Prometheus timer
PROMETHEUS_WAIT_SECONDS = 300

# Status and ready files
FAILURE_REASON_PATH = "/opt/ml/output/failure"

# Global variable to track Ray initialization status
ray_initialized = False
# Global variable to track if there was a failure
has_failure = False
# Global variable to track the Prometheus folder name
prometheus_folder_name = None


def signal_handler(signum: int, frame: Any) -> None:
    """Handle termination signals gracefully.

    Args:
        signum: Signal number
        frame: Current stack frame
    """
    global ray_initialized, has_failure
    signal_name = signal.Signals(signum).name
    logger.info("Received %s signal, initiating graceful shutdown...", signal_name)

    if ray_initialized:
        try:
            logger.info("Shutting down Ray...")
            ray.shutdown()
            ray_initialized = False
            logger.info("Ray shutdown completed successfully")
        except Exception as e:
            logger.warning("Error during Ray shutdown: %s", e)

    # Exit with failure code if there was a failure, otherwise success
    exit_code = DEFAULT_FAILURE_CODE if has_failure else SUCCESS_EXIT_CODE
    logger.info("Signal handler exiting with code: %s", exit_code)
    sys.exit(exit_code)


# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


def get_efa_supported_gpu_instances(region_name=None):
    """
    Dynamically fetch EFA-supported GPU instance types from AWS EC2.

    Args:
        region_name: AWS region name. If None, uses default region from environment/config

    Returns:
        List of SageMaker ML instance names (e.g., ['ml.p4d.24xlarge', 'ml.g5.12xlarge'])
    """
    try:
        # Use region from environment if not specified
        if region_name is None:
            region_name = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

        ec2_client = boto3.client("ec2", region_name=region_name)

        # Use paginator to ensure we get all results
        paginator = ec2_client.get_paginator("describe_instance_types")

        page_iterator = paginator.paginate(
            Filters=[{"Name": "network-info.efa-supported", "Values": ["true"]}]
        )

        instance_names = []

        for page in page_iterator:
            for instance_type in page["InstanceTypes"]:
                # Only include instances that have GPU info
                if "GpuInfo" in instance_type:
                    ml_instance_name = f"ml.{instance_type['InstanceType']}"
                    instance_names.append(ml_instance_name)

        logger.info(
            f"Found {len(instance_names)} EFA-supported GPU instance types in {region_name}"
        )
        logger.debug(f"EFA-supported GPU instances: {sorted(instance_names)}")

        if len(instance_names) > 0:
            return sorted(instance_names)
        else:
            return SM_EFA_NCCL_INSTANCES_FALLBACK

    except NoCredentialsError:
        logger.warning(
            "AWS credentials not found. Using fallback static EFA instance list."
        )
        return SM_EFA_NCCL_INSTANCES_FALLBACK
    except ClientError as e:
        logger.warning(
            f"AWS API error when fetching EFA instances: {e}. Using fallback static list."
        )
        return SM_EFA_NCCL_INSTANCES_FALLBACK
    except Exception as e:
        logger.warning(
            f"Unexpected error when fetching EFA instances: {e}. Using fallback static list."
        )
        return SM_EFA_NCCL_INSTANCES_FALLBACK


# Fallback static lists (your current lists as backup)
SM_EFA_NCCL_INSTANCES_FALLBACK = [
    "ml.g6e.48xlarge",
    "ml.g6.24xlarge",
    "ml.g6e.8xlarge",
    "ml.g6e.16xlarge",
    "ml.g6.48xlarge",
    "ml.g5.24xlarge",
    "ml.g6.8xlarge",
    "ml.g4dn.12xlarge",
    "ml.p4de.24xlarge",
    "ml.g6e.12xlarge",
    "ml.g5.12xlarge",
    "ml.p6-b200.48xlarge",
    "ml.p4d.24xlarge",
    "ml.p5.48xlarge",
    "ml.p5.4xlarge",
    "ml.g5.16xlarge",
    "ml.p5en.48xlarge",
    "ml.g6.16xlarge",
    "ml.g6.12xlarge",
    "ml.dl1.24xlarge",
    "ml.g4dn.16xlarge",
    "ml.gr6.8xlarge",
    "ml.g6e.24xlarge",
    "ml.g5.8xlarge",
    "ml.g5.48xlarge",
    "ml.p3dn.24xlarge",
    "ml.g4dn.8xlarge",
]

SM_EFA_RDMA_INSTANCES = [
    "ml.p4d.24xlarge",
    "ml.p4de.24xlarge",
    "ml.trn1.32xlarge",
]

# Initialize dynamic lists at module level (cached)
try:
    SM_EFA_NCCL_INSTANCES = get_efa_supported_gpu_instances()
except Exception as e:
    logger.warning(
        f"Failed to initialize dynamic EFA instance lists: {e}. Using static fallback lists."
    )
    SM_EFA_NCCL_INSTANCES = SM_EFA_NCCL_INSTANCES_FALLBACK


def _parse_args():
    """Parse CLI arguments.

    Args:
        sys_args: Command line arguments

    Returns:
        Parsed arguments and unknown arguments
    """
    parser = argparse.ArgumentParser(
        description="SageMaker Ray distributed workload launcher", allow_abbrev=False
    )

    parser.add_argument(
        "-e",
        "--entrypoint",
        type=str,
        help="Entry point script path (e.g., training/train.py, ./training/train.py, or training/train.sh)",
    )

    parser.add_argument(
        "--head-instance-group",
        type=str,
        default=None,
        help="Instance group that should act as head node",
    )

    parser.add_argument(
        "--head-num-cpus",
        type=int,
        default=None,
        help="Number of CPUs to reserve to the head node",
    )

    parser.add_argument(
        "--head-num-gpus",
        type=int,
        default=None,
        help="Number of GPUs to reserve to the head node",
    )

    parser.add_argument(
        "--include-dashboard",
        type=bool,
        default=True,
        help="Include ray dashboard",
    )

    parser.add_argument(
        "--launch-prometheus",
        type=bool,
        default=False,
        help="Number of seconds to wait before shutting down Ray server",
    )

    parser.add_argument(
        "--prometheus-path",
        type=str,
        default=None,
        help="Path to the prometheus binary tar.gz file to copy to /opt/ml/code",
    )

    parser.add_argument(
        "--wait-shutdown",
        type=int,
        default=None,
        help="Number of seconds to wait before shutting down Ray server",
    )

    args, unknown = parser.parse_known_args()

    logger.info(f"Arguments: {args}")

    if unknown:
        logger.info(f"Ignoring unknown arguments: {unknown}")

    # Handle wait_shutdown parameter from environment variable if not provided as argument
    if args.wait_shutdown is None:
        env_wait_shutdown = os.environ.get("wait_shutdown")
        if env_wait_shutdown is not None:
            try:
                args.wait_shutdown = int(env_wait_shutdown)
                logger.info(
                    "Using wait_shutdown from environment variable: %s",
                    args.wait_shutdown,
                )
            except ValueError:
                logger.warning(
                    "Invalid wait_shutdown environment variable value: %s. Must be an integer.",
                    env_wait_shutdown,
                )

    # Handle head_instance_group parameter from environment variable if not provided as argument
    if args.head_instance_group is None:
        env_head_instance_group = os.environ.get("head_instance_group")
        if env_head_instance_group is not None:
            args.head_instance_group = env_head_instance_group
            logger.info(
                "Using head_instance_group from environment variable: %s",
                args.head_instance_group,
            )

    # Handle head_num_cpus parameter from environment variable if not provided as argument
    if args.head_num_cpus is None:
        env_head_num_cpus = os.environ.get("head_num_cpus")
        if env_head_num_cpus is not None:
            try:
                args.head_num_cpus = int(env_head_num_cpus)
                logger.info(
                    "Using head_num_cpus from environment variable: %s",
                    args.head_num_cpus,
                )
            except ValueError:
                logger.warning(
                    "Invalid head_num_cpus environment variable value: %s. Must be an integer.",
                    env_head_num_cpus,
                )

    # Handle head_num_gpus parameter from environment variable if not provided as argument
    if args.head_num_gpus is None:
        env_head_num_gpus = os.environ.get("head_num_gpus")
        if env_head_num_gpus is not None:
            try:
                args.head_num_gpus = int(env_head_num_gpus)
                logger.info(
                    "Using head_num_gpus from environment variable: %s",
                    args.head_num_gpus,
                )
            except ValueError:
                logger.warning(
                    "Invalid head_num_gpus environment variable value: %s. Must be an integer.",
                    env_head_num_gpus,
                )

    # Handle launch_prometheus parameter from environment variable if not provided as argument
    env_launch_prometheus = os.environ.get("launch_prometheus")
    if env_launch_prometheus is not None:
        try:
            # Convert string to boolean (handle common boolean representations)
            if env_launch_prometheus.lower() in ("true", "1", "yes", "on"):
                args.launch_prometheus = True
            elif env_launch_prometheus.lower() in ("false", "0", "no", "off"):
                args.launch_prometheus = False
            else:
                raise ValueError(f"Invalid boolean value: {env_launch_prometheus}")

            logger.info(
                "Using launch_prometheus from environment variable: %s",
                args.launch_prometheus,
            )
        except ValueError as e:
            logger.warning(
                "Invalid launch_prometheus environment variable value: %s. Must be a boolean (true/false, 1/0, yes/no, on/off). Error: %s",
                env_launch_prometheus,
                e,
            )

    # Handle prometheus_path parameter from environment variable if not provided as argument
    if args.prometheus_path is None:
        env_prometheus_path = os.environ.get("prometheus_path")
        if env_prometheus_path is not None:
            args.prometheus_path = env_prometheus_path
            logger.info(
                "Using prometheus_path from environment variable: %s",
                args.prometheus_path,
            )

    # If entrypoint is provided, parse it and set environment variables
    if args.entrypoint:
        entrypoint_path = args.entrypoint

        # Remove leading "./" if present
        if entrypoint_path.startswith("./"):
            entrypoint_path = entrypoint_path[2:]

        # Split the path to get source_dir and entry_script
        if "/" in entrypoint_path:
            # Split into directory and filename
            path_parts = entrypoint_path.split("/")
            source_dir = "/".join(path_parts[:-1])  # All parts except the last
            entry_script = path_parts[-1]  # Last part (filename)
        else:
            # No directory, just filename
            source_dir = ""
            entry_script = entrypoint_path

        # Set environment variables
        os.environ["source_dir"] = source_dir
        os.environ["entry_script"] = entry_script

        logger.info("Entrypoint argument provided: %s", args.entrypoint)
        logger.info("Set source_dir=%s, entry_script=%s", source_dir, entry_script)

    return args, unknown


def _build_prometheus_command(prometheus_folder_name: str) -> str:
    """Build a safe prometheus command string.

    Args:
        prometheus_folder_name: Name of the prometheus folder

    Returns:
        Safely constructed prometheus command string
    """
    prometheus_binary_path = f"./{prometheus_folder_name}/prometheus"
    config_file_path = "/tmp/ray/session_latest/metrics/prometheus/prometheus.yml"
    return f"{shlex.quote(prometheus_binary_path)} --config.file={shlex.quote(config_file_path)}"


def _copy_prometheus_binary(prometheus_path: str) -> str:
    """Copy prometheus tar.gz file to /opt/ml/code directory and extract it.

    Args:
        prometheus_path: Path to the prometheus binary tar.gz file

    Returns:
        The name of the extracted folder

    Raises:
        FileNotFoundError: If the prometheus file doesn't exist
        Exception: If there are errors during file copy or extraction
    """
    import shutil
    import tarfile

    if not os.path.exists(prometheus_path):
        raise FileNotFoundError(f"Prometheus binary file not found: {prometheus_path}")

    # Ensure the destination directory exists
    destination_dir = "/opt/ml/code"
    os.makedirs(destination_dir, exist_ok=True)

    # Get the filename from the path
    filename = os.path.basename(prometheus_path)
    destination_path = os.path.join(destination_dir, filename)

    try:
        logger.info(
            "Copying prometheus binary from %s to %s", prometheus_path, destination_path
        )
        shutil.copy2(prometheus_path, destination_path)
        logger.info("Successfully copied prometheus binary to %s", destination_path)

        # Verify the file was copied successfully
        if os.path.exists(destination_path):
            original_size = os.path.getsize(prometheus_path)
            copied_size = os.path.getsize(destination_path)
            if original_size == copied_size:
                logger.info(
                    "File copy verification successful (size: %s bytes)", copied_size
                )
            else:
                logger.warning(
                    "File size mismatch: original=%s, copied=%s",
                    original_size,
                    copied_size,
                )
        else:
            raise Exception(
                f"File copy failed: destination file not found at {destination_path}"
            )

        # Extract the tar.gz file
        logger.info("Extracting prometheus binary from %s", destination_path)

        # Determine the folder name from the filename
        # Remove .tar.gz extension to get the folder name
        if filename.endswith(".tar.gz"):
            folder_name = filename[:-7]  # Remove '.tar.gz'
        elif filename.endswith(".tgz"):
            folder_name = filename[:-4]  # Remove '.tgz'
        else:
            # Fallback: remove common archive extensions
            folder_name = filename.rsplit(".", 1)[0]

        logger.info("Expected folder name after extraction: %s", folder_name)

        # Extract the tar.gz file in the destination directory with security validation
        with tarfile.open(destination_path, "r:gz") as tar:
            # Validate and extract members safely
            _safe_extract_all(tar, destination_dir)

        # Verify the extraction was successful
        extracted_folder_path = os.path.join(destination_dir, folder_name)
        if os.path.exists(extracted_folder_path):
            logger.info(
                "Successfully extracted prometheus binary to %s", extracted_folder_path
            )

            # Check if prometheus binary exists in the extracted folder
            prometheus_binary_path = os.path.join(extracted_folder_path, "prometheus")
            if os.path.exists(prometheus_binary_path):
                logger.info("Prometheus binary found at %s", prometheus_binary_path)
            else:
                logger.warning(
                    "Prometheus binary not found at expected path: %s",
                    prometheus_binary_path,
                )
        else:
            raise Exception(
                f"Extraction failed: folder not found at {extracted_folder_path}"
            )

        return folder_name

    except Exception as e:
        logger.error("Error copying or extracting prometheus binary: %s", e)
        raise


def _create_runtime_environment(args: argparse.Namespace, env: Any) -> Dict[str, Any]:
    """
    Create the Ray runtime environment configuration based on instance type.
    Includes ALL current environment variables to ensure complete environment is available to Ray workers.

    Args:
        env: SageMaker environment object

    Returns:
        Dict containing ALL environment variables plus Ray-specific configurations
    """
    # Start with ALL current environment variables
    runtime_env = dict(os.environ)

    # Get the source_dir directory path for Ray workers
    source_dir = os.environ.get("source_dir", "")
    current_dir = os.getcwd()
    absolute_source_dir = (
        os.path.join(current_dir, source_dir) if source_dir else current_dir
    )

    # Get current PYTHONPATH and add our source_dir directory
    current_pythonpath = runtime_env.get("PYTHONPATH", "")
    if current_pythonpath:
        new_pythonpath = f"{absolute_source_dir}:{current_pythonpath}"
    else:
        new_pythonpath = absolute_source_dir

    # Override/add specific Ray and networking environment variables
    runtime_env.update(
        {
            "NCCL_SOCKET_IFNAME": str(env.network_interface_name),
            "NCCL_PROTO": "simple",
            "PYTHONPATH": new_pythonpath,  # Add source_dir directory to PYTHONPATH for Ray workers
        }
    )

    # Configure EFA if supported by the instance type
    if env.current_instance_type in SM_EFA_NCCL_INSTANCES:
        runtime_env["FI_PROVIDER"] = "efa"

    # Configure RDMA if supported by the instance type
    if env.current_instance_type in SM_EFA_RDMA_INSTANCES:
        runtime_env["FI_EFA_USE_DEVICE_RDMA"] = "1"
        runtime_env["RDMAV_FORK_SAFE"] = "1"

    if args.launch_prometheus:
        # Configure Prometheus host - Ray Dashboard connects to Prometheus via this URL
        runtime_env["RAY_PROMETHEUS_HOST"] = "http://127.0.0.1:9090"
    else:
        if os.environ.get("RAY_PROMETHEUS_HOST") is not None:
            runtime_env["RAY_PROMETHEUS_HOST"] = os.environ.get("RAY_PROMETHEUS_HOST")

    if os.environ.get("RAY_PROMETHEUS_NAME") is not None:
        runtime_env["RAY_PROMETHEUS_NAME"] = os.environ.get("RAY_PROMETHEUS_NAME")

    # Configure Grafana environment variables for Ray Dashboard integration
    if os.environ.get("RAY_GRAFANA_HOST") is not None:
        runtime_env["RAY_GRAFANA_HOST"] = os.environ.get("RAY_GRAFANA_HOST")

    # RAY_GRAFANA_IFRAME_HOST: Used by browser to fetch Grafana panels
    if os.environ.get("RAY_GRAFANA_IFRAME_HOST") is not None:
        runtime_env["RAY_GRAFANA_IFRAME_HOST"] = os.environ.get(
            "RAY_GRAFANA_IFRAME_HOST"
        )
    elif os.environ.get("RAY_GRAFANA_HOST") is not None:
        runtime_env["RAY_GRAFANA_IFRAME_HOST"] = os.environ.get("RAY_GRAFANA_HOST")

    if runtime_env.get("RAY_PROMETHEUS_HOST") is not None:
        logger.info(
            "Configured Prometheus host: %s", runtime_env.get("RAY_PROMETHEUS_HOST")
        )

    if runtime_env.get("RAY_GRAFANA_HOST") is not None:
        logger.info("Configured Grafana host: %s", os.environ.get("RAY_GRAFANA_HOST"))

    logger.info(
        "Ray runtime environment contains %d total environment variables",
        len(runtime_env),
    )
    logger.info("Ray runtime environment: %s", runtime_env)

    logger.info("source_dir directory added to PYTHONPATH: %s", absolute_source_dir)
    return runtime_env


def _execute_entry_script(
    runtime_env: Dict[str, Any],
) -> None:
    """Execute the entry script dynamically based on environment variables.

    This function loads and executes the script based on its file extension:
    - .py files: Loaded as Python modules and executed
    - .sh files: Executed as bash scripts

    Args:
        runtime_env: Ray runtime environment configuration

    Raises:
        ValueError: If required environment variables are not set or unsupported file type
        FileNotFoundError: If the script file cannot be found
        Exception: If there are errors during script execution
    """
    global has_failure
    source_dir = os.environ.get("source_dir", "")
    entry_script = os.environ.get("entry_script")

    # source_dir can be empty if the script is in the same directory as launcher
    # but entry_script is always required
    if not entry_script:
        raise ValueError("entry_script environment variable is required")

    logger.info("Raw source_dir from env: '%s'", source_dir)
    logger.info("Raw entry_script from env: '%s'", entry_script)

    # Get current working directory and construct absolute paths
    current_dir = os.getcwd()  # This should be /opt/ml/input/data/code

    # Handle empty source_dir (script in same directory as launcher)
    if source_dir:
        absolute_source_dir = os.path.join(current_dir, source_dir)
    else:
        absolute_source_dir = current_dir

    script_path = os.path.join(absolute_source_dir, entry_script)

    logger.info("Current working directory: %s", current_dir)
    logger.info("Absolute source directory: %s", absolute_source_dir)
    logger.info("Script path: %s", script_path)

    # Debug: List contents of directories
    if os.path.exists(current_dir):
        logger.info("Contents of %s: %s", current_dir, os.listdir(current_dir))
    if os.path.exists(absolute_source_dir):
        logger.info(
            "Contents of %s: %s", absolute_source_dir, os.listdir(absolute_source_dir)
        )

    logger.info("Script path exists: %s", os.path.exists(script_path))

    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Entry script not found: {script_path}")

    # Determine script type based on file extension
    script_extension = os.path.splitext(entry_script)[1].lower()

    try:
        if script_extension == ".py":
            _execute_python_script(script_path, absolute_source_dir)
        elif script_extension == ".sh":
            _execute_bash_script(script_path, absolute_source_dir, runtime_env)
        else:
            raise ValueError(
                f"Unsupported script type: {script_extension}. Only .py and .sh files are supported."
            )

        logger.info("Script execution completed successfully")

    except Exception as e:
        has_failure = True
        logger.error("Error executing entry script %s: %s", script_path, e)
        logger.error("Exception type: %s", type(e).__name__)
        import traceback

        logger.error("Traceback: %s", traceback.format_exc())
        raise
    finally:
        # Restore original working directory
        os.chdir(current_dir)
        logger.info("Restored working directory to: %s", current_dir)


def _execute_python_script(script_path: str, absolute_source_dir: str) -> None:
    """Execute a Python script using importlib.

    Args:
        script_path: Full path to the Python script
        absolute_source_dir: Absolute path to the source directory
    """
    # Change to the absolute source directory so relative imports work
    os.chdir(absolute_source_dir)
    logger.info("Changed working directory to: %s", absolute_source_dir)
    logger.info("Current working directory after change: %s", os.getcwd())
    logger.info("Contents of current directory: %s", os.listdir("."))

    # Add the absolute source directory to Python path if not already there
    if absolute_source_dir not in sys.path:
        sys.path.insert(0, absolute_source_dir)
        logger.info("Added %s to sys.path", absolute_source_dir)

    # Use importlib to load and execute the script
    # We are calling the module as __main__ to ensure it runs as a script
    logger.info("Loading and executing Python script using importlib...")

    spec = importlib.util.spec_from_file_location("__main__", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def _execute_bash_script(
    script_path: str,
    absolute_source_dir: str,
    runtime_env: Dict[str, Any],
) -> None:
    """Execute a bash script using subprocess.

    Args:
        script_path: Full path to the bash script
        absolute_source_dir: Absolute path to the source directory
        runtime_env: Ray runtime environment configuration
    """
    # Change to the absolute source directory so relative paths in the script work
    os.chdir(absolute_source_dir)
    logger.info("Changed working directory to: %s", absolute_source_dir)
    logger.info("Current working directory after change: %s", os.getcwd())
    logger.info("Contents of current directory: %s", os.listdir("."))

    # Execute the bash script
    logger.info("Executing bash script: %s", script_path)

    # Use subprocess to run the bash script
    try:
        result = subprocess.run(
            ["bash", script_path],
            check=True,
            capture_output=True,
            text=True,
            env=runtime_env,
            cwd=absolute_source_dir,
        )

        # Log the output
        if result.stdout:
            logger.info("Script stdout:\n%s", result.stdout)
        if result.stderr:
            logger.info("Script stderr:\n%s", result.stderr)

        logger.info("Bash script completed with return code: %s", result.returncode)

    except subprocess.CalledProcessError as e:
        logger.error("Bash script failed with return code: %s", e.returncode)
        if e.stdout:
            logger.error("Script stdout:\n%s", e.stdout)
        if e.stderr:
            logger.error("Script stderr:\n%s", e.stderr)
        raise


def _get_ip_from_host(host):
    """Get the IP address from the current host."""
    import socket

    ip_wait_time = 200
    counter = 0
    ip = ""

    while counter < ip_wait_time and ip == "":
        try:
            ip = socket.gethostbyname(host)
            break
        except:
            counter += 1
            time.sleep(5)

    if counter == ip_wait_time and ip == "":
        raise Exception(
            "Exceeded max wait time of %ss for hostname resolution" % ip_wait_time
        )

    logger.info("IP address for %s is %s", host, ip)
    return ip


def _log_environment_debug_info() -> None:
    """Log environment variables for debugging."""
    source_dir = os.environ.get("source_dir", "Not set")
    entry_script = os.environ.get("entry_script", "Not set")
    logger.info("source_dir: %s", source_dir)
    logger.info("entry_script: %s", entry_script)


def _read_and_log_prometheus_logs(log_file_path: str) -> None:
    """Read and log the contents of a Prometheus log file.

    Args:
        log_file_path: Path to the log file to read and log
    """
    try:
        if not os.path.exists(log_file_path):
            logger.warning("Log file not found: %s", log_file_path)
            return

        with open(log_file_path, "r") as f:
            content = f.read()

        if content.strip():
            logger.info("Contents of %s:\n%s", log_file_path, content)
        else:
            logger.info("Log file %s is empty", log_file_path)

    except Exception as e:
        logger.error("Error reading log file %s: %s", log_file_path, e)


def _run_script(
    runtime_env: Dict[str, Any],
) -> None:
    """
    Execute the dynamically loaded entry script.

    Args:
        runtime_env: Ray runtime environment configuration
    """
    global has_failure
    try:
        _execute_entry_script(runtime_env)
        logger.info("Entry script execution complete")
    except Exception as e:
        has_failure = True
        logger.error("Error executing entry script: %s", e)
        raise


def _validate_command(args: List[str]) -> None:
    """Validate that command uses only allowed executables.

    Args:
        args: Command arguments list

    Raises:
        ValueError: If command is not allowed
    """
    if not args:
        raise ValueError("Empty command not allowed")

    executable = args[0]
    if executable in ["ray", "bash"] or executable.startswith("./prometheus-"):
        return

    raise ValueError(f"Command not allowed: {executable}")


def _run_subprocess_command_with_env(
    command: str, env_vars: Dict[str, str], check: bool = True
) -> Tuple[int, str, str]:
    """
    Run a shell command with custom environment variables and return the result.

    Args:
        command: Shell command to execute (will be parsed safely)
        env_vars: Dictionary of environment variables to set for the process
        check: Whether to raise an exception on non-zero exit status

    Returns:
        Tuple of (return_code, stdout, stderr)

    Raises:
        subprocess.CalledProcessError: If check is True and the command returns non-zero exit status
    """
    try:
        # Parse command string into arguments to avoid shell injection
        args = shlex.split(command)
        _validate_command(args)

        result = subprocess.run(
            args, shell=False, check=check, capture_output=True, text=True, env=env_vars
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        logger.error("Command '%s' failed with exit status %s", command, e.returncode)
        logger.error("STDOUT: %s", e.stdout)
        logger.error("STDERR: %s", e.stderr)
        if check:
            raise
        return e.returncode, e.stdout, e.stderr


def _run_subprocess_command(command: str, check: bool = True) -> Tuple[int, str, str]:
    """
    Run a shell command and return the result.

    Args:
        command: Shell command to execute (will be parsed safely)
        check: Whether to raise an exception on non-zero exit status

    Returns:
        Tuple of (return_code, stdout, stderr)

    Raises:
        subprocess.CalledProcessError: If check is True and the command returns non-zero exit status
    """
    try:
        # Parse command string into arguments to avoid shell injection
        args = shlex.split(command)
        _validate_command(args)

        result = subprocess.run(
            args, shell=False, check=check, capture_output=True, text=True
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        logger.error("Command '%s' failed with exit status %s", command, e.returncode)
        logger.error("STDOUT: %s", e.stdout)
        logger.error("STDERR: %s", e.stderr)
        if check:
            raise
        return e.returncode, e.stdout, e.stderr


def _run_subprocess_command_async(
    command: str,
    stdout_file: Optional[str] = None,
    stderr_file: Optional[str] = None,
    env_vars: Optional[Dict[str, str]] = None,
    wait_in_seconds: int = 0,
) -> subprocess.Popen:
    """
    Run a shell command asynchronously using subprocess.Popen without waiting for completion.

    Args:
        command: Shell command to execute (will be parsed safely)
        stdout_file: Optional file path to redirect stdout (defaults to subprocess.PIPE)
        stderr_file: Optional file path to redirect stderr (defaults to subprocess.PIPE)
        env_vars: Optional dictionary of environment variables to set for the process
        wait_in_seconds: Number of seconds to wait after starting the process (0 = no wait)

    Returns:
        subprocess.Popen object representing the running process

    Example:
        # Start a long-running process without waiting
        process = _run_subprocess_command_async("sleep 60")
        logger.info("Process started with PID: %s", process.pid)

        # Start a process and wait 5 seconds before returning
        process = _run_subprocess_command_async("ray start --head", wait_in_seconds=5)

        # You can check if it's still running
        if process.poll() is None:
            logger.info("Process is still running")

        # Or terminate it later if needed
        # process.terminate()
    """
    try:
        # Parse command string into arguments to avoid shell injection
        args = shlex.split(command)
        _validate_command(args)

        # Validate and set up stdout redirection
        if stdout_file:
            # Validate file path to prevent directory traversal
            if ".." in stdout_file or stdout_file.startswith("/"):
                if not stdout_file.startswith("/tmp/"):
                    raise ValueError(f"Invalid stdout file path: {stdout_file}")
            stdout = open(stdout_file, "w")
        else:
            stdout = subprocess.PIPE

        # Validate and set up stderr redirection
        if stderr_file:
            # Validate file path to prevent directory traversal
            if ".." in stderr_file or stderr_file.startswith("/"):
                if not stderr_file.startswith("/tmp/"):
                    raise ValueError(f"Invalid stderr file path: {stderr_file}")
            stderr = open(stderr_file, "w")
        else:
            stderr = subprocess.PIPE

        # Set up environment variables
        process_env = os.environ.copy()
        if env_vars:
            process_env.update(env_vars)

        logger.info("Starting async command: %s", command)

        # Start the process without waiting for completion
        process = subprocess.Popen(
            args,
            shell=False,
            stdout=stdout,
            stderr=stderr,
            env=process_env,
            text=True,
        )

        logger.info("Async process started with PID: %s", process.pid)

        # Close file handles immediately after Popen since they're now owned by the subprocess
        if stdout_file and stdout != subprocess.PIPE:
            stdout.close()
        if stderr_file and stderr != subprocess.PIPE:
            stderr.close()

        # Wait for specified seconds if requested
        if wait_in_seconds > 0:
            logger.info("Waiting %s seconds after starting process...", wait_in_seconds)
            time.sleep(wait_in_seconds)

            # Check if process is still running after wait
            if process.poll() is None:
                logger.info(
                    "Process is still running after %s seconds wait", wait_in_seconds
                )
            else:
                logger.info(
                    "Process completed during %s seconds wait with return code: %s",
                    wait_in_seconds,
                    process.returncode,
                )

        return process

    except Exception as e:
        logger.error("Error starting async command '%s': %s", command, e)
        # Clean up file handles if they were opened
        if stdout_file and "stdout" in locals() and hasattr(stdout, "close"):
            stdout.close()
        if stderr_file and "stderr" in locals() and hasattr(stderr, "close"):
            stderr.close()
        raise


def _safe_extract_all(tar: tarfile.TarFile, path: str) -> None:
    """Safely extract all members from a tar file, preventing directory traversal attacks.

    Args:
        tar: The TarFile object to extract from
        path: The destination directory path

    Raises:
        ValueError: If a member path is unsafe (contains directory traversal)
        Exception: If extraction fails for any other reason
    """

    def _is_safe_path(member_path: str, destination: str) -> bool:
        """Check if the member path is safe for extraction."""
        # Resolve the full path and check if it's within the destination directory
        full_path = os.path.realpath(os.path.join(destination, member_path))
        destination_path = os.path.realpath(destination)
        return (
            full_path.startswith(destination_path + os.sep)
            or full_path == destination_path
        )

    def _is_safe_member(member: tarfile.TarInfo) -> bool:
        """Check if a tar member is safe to extract."""
        # Check for absolute paths
        if os.path.isabs(member.name):
            return False

        # Check for directory traversal patterns
        if ".." in member.name:
            return False

        # Check for null bytes
        if "\x00" in member.name:
            return False

        # Additional check using path resolution
        return _is_safe_path(member.name, path)

    # Validate all members before extraction
    unsafe_members = []
    for member in tar.getmembers():
        if not _is_safe_member(member):
            unsafe_members.append(member.name)

    if unsafe_members:
        raise ValueError(
            f"Unsafe tar members detected (potential directory traversal): {unsafe_members}"
        )

    # Extract all members (now that we've validated they're safe)
    for member in tar.getmembers():
        try:
            tar.extract(member, path)
        except Exception as e:
            logger.error("Failed to extract member %s: %s", member.name, e)
            raise


def _setup_head_node(
    hosts: List[str],
    runtime_env: Dict[str, Any],
    args: argparse.Namespace,
    env: Any,
) -> int:
    """Configure and run the Ray head node.

    Args:
        hosts: List of all hosts in the cluster
        runtime_env: Ray runtime environment configuration
        args: Command line arguments
        env: SageMaker environment variables object

    Returns:
        Return code from Ray stop command
    """
    global ray_initialized, has_failure, prometheus_folder_name

    try:
        num_cpus = (
            args.head_num_cpus if args.head_num_cpus is not None else env.num_cpus
        )
        num_gpus = (
            args.head_num_gpus if args.head_num_gpus is not None else env.num_gpus
        )

        logger.info("CPUs for the head node: %s", num_cpus)
        logger.info("GPUs for the head node: %s", num_gpus)

        # Build Ray start command safely (no runtime-env option available in CLI)
        ray_cmd = f"ray start --head --num-cpus={shlex.quote(str(num_cpus))} --num-gpus={shlex.quote(str(num_gpus))} --port={DEFAULT_RAY_PORT}"
        ray_init_kwargs = {
            "address": "auto",
            "include_dashboard": args.include_dashboard,
            "runtime_env": {"env_vars": runtime_env},
        }

        if args.include_dashboard:
            ray_cmd += " --dashboard-host=0.0.0.0 --dashboard-port=8265 --metrics-export-port=8080"
            ray_init_kwargs["dashboard_host"] = "0.0.0.0"
            ray_init_kwargs["dashboard_port"] = 8265

        # Set environment variables for the Ray process
        env_for_ray = os.environ.copy()
        env_for_ray.update(runtime_env)
        _run_subprocess_command_with_env(ray_cmd, env_for_ray, check=True)

        ray.init(**ray_init_kwargs)

        if args.include_dashboard and args.launch_prometheus:
            logger.info("Launching prometheus")
            if args.prometheus_path and prometheus_folder_name:
                # Launch custom Prometheus binary with output capture
                prometheus_cmd = _build_prometheus_command(prometheus_folder_name)
                logger.info(
                    "Starting custom Prometheus with command: %s", prometheus_cmd
                )
                _run_subprocess_command_async(
                    prometheus_cmd,
                    wait_in_seconds=0,
                    stdout_file="/tmp/prometheus_stdout.log",
                    stderr_file="/tmp/prometheus_stderr.log",
                )
            else:
                # Launch Prometheus with output capture
                _run_subprocess_command_async(
                    "ray metrics launch-prometheus",
                    wait_in_seconds=0,
                    stdout_file="/tmp/prometheus_stdout.log",
                    stderr_file="/tmp/prometheus_stderr.log",
                )

            # Poll for Prometheus readiness
            start_time = time.time()
            prometheus_ready = False

            logger.info(
                "Waiting for Prometheus to become ready (max %s seconds)...",
                PROMETHEUS_WAIT_SECONDS,
            )

            while time.time() - start_time < PROMETHEUS_WAIT_SECONDS:
                try:
                    response = requests.get(
                        f"{ray_init_kwargs['runtime_env']['RAY_PROMETHEUS_HOST']}/-/healthy",
                        timeout=5,
                    )
                    if response.status_code == 200:
                        elapsed_time = time.time() - start_time
                        logger.info(
                            "Prometheus health check passed after %.1f seconds",
                            elapsed_time,
                        )
                        prometheus_ready = True
                        break
                except Exception as e:
                    logger.warning(str(e))
                    # Prometheus not ready yet, continue polling
                    pass

                # Wait 2 seconds before next check
                time.sleep(2)

            if not prometheus_ready:
                logger.warning(
                    "Prometheus did not become ready within %s seconds",
                    PROMETHEUS_WAIT_SECONDS,
                )

            # _read_and_log_prometheus_logs("/tmp/prometheus_stdout.log")

        ray_initialized = True

        # Wait for all worker nodes to connect
        cluster_size = len(hosts)
        connected_nodes = 1
        start_time = time.time()

        while connected_nodes < cluster_size:
            if time.time() - start_time > RAY_CONNECTION_TIMEOUT:
                logger.warning(
                    "Timed out waiting for all nodes to connect after %s seconds",
                    RAY_CONNECTION_TIMEOUT,
                )
                logger.warning(
                    "Proceeding with %s/%s nodes", connected_nodes, cluster_size
                )
                break

            time.sleep(1)
            resources = ray.available_resources().keys()
            curr_nodes = [r for r in resources if r.startswith("node:")]
            connected_nodes = len(curr_nodes)

            if connected_nodes < cluster_size and (time.time() - start_time) % 30 < 1:
                logger.info(
                    "Waiting for nodes to connect: %s/%s", connected_nodes, cluster_size
                )
                logger.info("Currently connected nodes: %s", curr_nodes)

        logger.info("All nodes connected to the Ray cluster!")
        _run_script(runtime_env)

    except Exception as e:
        has_failure = True
        logger.error("Error in head node setup or script execution: %s", e)
        raise
    finally:
        if not has_failure:
            _wait_before_shutdown(args.wait_shutdown)

        if args.include_dashboard and args.launch_prometheus:
            logger.info("Shutting down prometheus")
            _run_subprocess_command("ray metrics shutdown-prometheus", check=False)

        _shutdown_ray_safely()
        returncode, _, _ = _run_subprocess_command("ray stop", check=False)
        return returncode


def _setup_worker_node(
    head: str,
    runtime_env: Dict[str, Any],
) -> int:
    """Configure and run a Ray worker node.

    Args:
        head: Hostname of the head node
        runtime_env: Ray runtime environment configuration

    Returns:
        Return code from Ray stop command
    """
    master_ip = _get_ip_from_host(head)
    # Connect to the head node - construct command safely (no runtime-env option available in CLI)
    ray_address = f"{master_ip}:{DEFAULT_RAY_PORT}"
    ray_start_cmd = f"ray start --address={shlex.quote(ray_address)}"

    # Set environment variables for the Ray process
    env_for_ray = os.environ.copy()
    env_for_ray.update(runtime_env)
    _run_subprocess_command_with_env(ray_start_cmd, env_for_ray, check=True)

    # Keep worker node alive until head node completes
    poll_count = 0

    while _is_ray_alive():
        time.sleep(RAY_WORKER_POLL_INTERVAL)
        poll_count += 1
        if poll_count % 6 == 0:  # Log every ~60 seconds
            logger.info("Worker node still connected to Ray cluster")

    logger.info("Head node is down, shutting down worker node")
    returncode, _, _ = _run_subprocess_command("ray stop", check=False)
    return returncode


def _is_ray_alive() -> bool:
    """
    Check if the Ray cluster is still running.

    Returns:
        True if Ray is running, False otherwise
    """
    try:
        returncode, _, _ = _run_subprocess_command("ray status", check=False)
        return returncode == 0
    except Exception as e:
        logger.warning("Error checking Ray status: %s", e)
        return False


def _get_cluster_configuration(
    args: argparse.Namespace, env: Any
) -> Tuple[List[str], str, int]:
    """
    Get cluster configuration for both homogeneous and heterogeneous setups.

    Args:
        args: Command line arguments
        env: SageMaker environment variables

    Returns:
        Tuple of (all_hosts, head_host, total_host_count)
    """
    if env.is_hetero:
        return _get_heterogeneous_cluster_config(args, env)
    else:
        return _get_homogeneous_cluster_config(env)


def _get_homogeneous_cluster_config(env: Any) -> Tuple[List[str], str, int]:
    """Get configuration for homogeneous cluster."""
    hosts = env.hosts
    head_host = hosts[0] if hosts else ""
    return hosts, head_host, len(hosts)


def _get_heterogeneous_cluster_config(
    args: argparse.Namespace, env: Any
) -> Tuple[List[str], str, int]:
    """Get configuration for heterogeneous cluster."""
    all_hosts = []
    head_host = ""
    total_host_count = 0

    # Find head instance group and collect all hosts
    for instance_group in env.instance_groups_dict.values():
        group_hosts = instance_group["hosts"]
        all_hosts.extend(group_hosts)

        if instance_group["instance_group_name"] == args.head_instance_group:
            head_host = group_hosts[0]  # First host in head group becomes head node

            # Check if head node should participate in computation
            if args.head_num_cpus is not None:
                head_num_cpus = args.head_num_cpus
            else:
                head_num_cpus = env.num_cpus

            if args.head_num_gpus is not None:
                head_num_gpus = args.head_num_gpus
            else:
                head_num_gpus = env.num_gpus

            if head_num_cpus == 0 and head_num_gpus == 0:
                # Head node is coordinator only, exclude from worker count
                total_host_count += len(group_hosts) - 1
                logger.info("Head node configured as coordinator only (0 CPUs, 0 GPUs)")
            else:
                # Head node participates in computation
                total_host_count += len(group_hosts)
        else:
            # All hosts in non-head groups are workers
            total_host_count += len(group_hosts)

    if not head_host:
        raise ValueError(
            "Head instance group '%s' not found" % args.head_instance_group
        )

    return all_hosts, head_host, total_host_count


def _setup_single_node_ray(
    args: argparse.Namespace, runtime_env: Dict[str, Any]
) -> int:
    """
    Set up Ray for single-node execution.

    Args:
        args: Command line arguments
        runtime_env: Ray runtime environment configuration
    """
    global ray_initialized, has_failure, prometheus_folder_name

    logger.info("Found a single host, initializing Ray as a single node")
    try:
        # Build Ray start command (no runtime-env option available in CLI)
        ray_cmd = f"ray start --head --port={DEFAULT_RAY_PORT}"
        ray_init_kwargs = {
            "address": "auto",
            "include_dashboard": args.include_dashboard,
            "runtime_env": {"env_vars": runtime_env},
        }

        if args.include_dashboard:
            ray_cmd += " --dashboard-host=0.0.0.0 --dashboard-port=8265 --metrics-export-port=8080"
            ray_init_kwargs["dashboard_host"] = "0.0.0.0"
            ray_init_kwargs["dashboard_port"] = 8265

        # Set environment variables for the Ray process
        env_for_ray = os.environ.copy()
        env_for_ray.update(runtime_env)
        _run_subprocess_command_with_env(ray_cmd, env_for_ray, check=True)

        ray.init(**ray_init_kwargs)

        if args.include_dashboard and args.launch_prometheus:
            logger.info("Launching prometheus")
            if args.prometheus_path and prometheus_folder_name:
                # Launch custom Prometheus binary with output capture
                prometheus_cmd = _build_prometheus_command(prometheus_folder_name)
                logger.info(
                    "Starting custom Prometheus with command: %s", prometheus_cmd
                )
                _run_subprocess_command_async(
                    prometheus_cmd,
                    wait_in_seconds=0,
                    stdout_file="/tmp/prometheus_stdout.log",
                    stderr_file="/tmp/prometheus_stderr.log",
                )
            else:
                # Launch Prometheus with output capture
                _run_subprocess_command_async(
                    "ray metrics launch-prometheus",
                    wait_in_seconds=0,
                    stdout_file="/tmp/prometheus_stdout.log",
                    stderr_file="/tmp/prometheus_stderr.log",
                )

            # Poll for Prometheus readiness
            start_time = time.time()
            prometheus_ready = False

            logger.info(
                "Waiting for Prometheus to become ready (max %s seconds)...",
                PROMETHEUS_WAIT_SECONDS,
            )

            while time.time() - start_time < PROMETHEUS_WAIT_SECONDS:
                try:
                    response = requests.get(
                        f"{ray_init_kwargs['runtime_env']['RAY_PROMETHEUS_HOST']}/-/healthy",
                        timeout=5,
                    )
                    if response.status_code == 200:
                        elapsed_time = time.time() - start_time
                        logger.info(
                            "Prometheus health check passed after %.1f seconds",
                            elapsed_time,
                        )
                        prometheus_ready = True
                        break
                except Exception as e:
                    logger.warning(str(e))
                    # Prometheus not ready yet, continue polling
                    pass

                # Wait 2 seconds before next check
                time.sleep(2)

            if not prometheus_ready:
                logger.warning(
                    "Prometheus did not become ready within %s seconds",
                    PROMETHEUS_WAIT_SECONDS,
                )

            # _read_and_log_prometheus_logs("/tmp/prometheus_stdout.log")

        ray_initialized = True
        _run_script(runtime_env)
    except Exception as e:
        has_failure = True
        logger.error("Error in single-node setup or script execution: %s", e)
        raise
    finally:
        if not has_failure:
            _wait_before_shutdown(args.wait_shutdown)

        if args.include_dashboard and args.launch_prometheus:
            logger.info("Shutting down prometheus")
            _run_subprocess_command("ray metrics shutdown-prometheus", check=False)

        _shutdown_ray_safely()
        returncode, _, _ = _run_subprocess_command("ray stop", check=False)
        return returncode


def _setup_multi_node_ray(
    all_hosts: List[str],
    head_host: str,
    runtime_env: Dict[str, Any],
    args: argparse.Namespace,
    env: Any,
) -> int:
    """
    Set up Ray for multi-node execution.

    Args:
        all_hosts: List of all hosts in the cluster
        head_host: Hostname of the head node
        runtime_env: Ray runtime environment configuration
        args: Command line arguments
        env: SageMaker environment variables

    Returns:
        Return code from Ray stop command
    """
    logger.info("Found multiple hosts, initializing Ray as a multi-node cluster")
    logger.info("Head node: %s, Current host: %s", head_host, env.current_host)

    if env.current_host == head_host:
        return _setup_head_node(all_hosts, runtime_env, args, env)
    else:
        return _setup_worker_node(head_host, runtime_env)


def _wait_before_shutdown(wait_shutdown: Optional[int]) -> None:
    """Wait for the specified number of seconds before shutting down Ray.

    Args:
        wait_shutdown: Number of seconds to wait, or None to skip waiting
    """
    if wait_shutdown is not None and wait_shutdown > 0:
        logger.info(
            "Waiting %s seconds before shutting down Ray server...", wait_shutdown
        )
        time.sleep(wait_shutdown)
        logger.info("Wait period completed, proceeding with shutdown")


def _shutdown_ray_safely() -> None:
    """Safely shutdown Ray with proper error handling."""
    global ray_initialized

    if ray_initialized:
        try:
            logger.info("Shutting down Ray...")
            ray.shutdown()
            ray_initialized = False
            logger.info("Ray shutdown completed successfully")
        except Exception as e:
            logger.warning("Error during Ray shutdown: %s", e)


def _setup_ray_environment_homogeneous_cluster(
    args: argparse.Namespace,
    env: Any,
) -> Optional[int]:
    """
    Set up the Ray execution environment for homogeneous distributed workload.

    Args:
        args: Command line arguments
        env: SageMaker environment variables

    Returns:
        Return code from Ray stop command (for multi-node setups) or None
    """
    # Create runtime environment based on instance type
    runtime_env = _create_runtime_environment(args, env)

    # Log environment variables for debugging
    _log_environment_debug_info()

    # Get cluster configuration
    all_hosts, head_host, total_host_count = _get_cluster_configuration(args, env)

    logger.info("Homogeneous cluster configuration: %s total hosts", total_host_count)
    logger.info("All hosts: %s", all_hosts)

    # Single-node workload scenario
    if total_host_count == 1:
        return _setup_single_node_ray(args, runtime_env)

    # Multi-node workload scenario
    return _setup_multi_node_ray(all_hosts, head_host, runtime_env, args, env)


def _setup_ray_environment_heterogeneous_cluster(
    args: argparse.Namespace,
    env: Any,
) -> Optional[int]:
    """
    Set up the Ray execution environment for heterogeneous distributed workload.

    Args:
        args: Command line arguments
        env: SageMaker environment variables

    Returns:
        Return code from Ray stop command (for multi-node setups) or None
    """
    # Validate required arguments for heterogeneous setup
    if not args.head_instance_group:
        raise ValueError("--head-instance-group is required for heterogeneous clusters")

    # Create runtime environment based on instance type
    runtime_env = _create_runtime_environment(args, env)

    # Log environment variables for debugging
    _log_environment_debug_info()

    # Get cluster configuration
    all_hosts, head_host, total_host_count = _get_cluster_configuration(args, env)

    logger.info("Heterogeneous cluster configuration: %s total hosts", total_host_count)
    logger.info("Head instance group: %s", args.head_instance_group)
    logger.info("Head host: %s", head_host)
    logger.info("All hosts: %s", all_hosts)

    # Single-node workload scenario
    if total_host_count == 1:
        return _setup_single_node_ray(args, runtime_env)

    # Multi-node workload scenario
    return _setup_multi_node_ray(all_hosts, head_host, runtime_env, args, env)


def _write_failure_reason_file(failure_msg: str) -> None:
    """Create a file 'failure' with failure reason if Ray initialization failed.

    Args:
        failure_msg: The content of file to be written.
    """
    if not os.path.exists(FAILURE_REASON_PATH):
        with open(FAILURE_REASON_PATH, "w") as f:
            f.write("RayRuntimeError: " + failure_msg)


def main() -> int:
    """Main entry point for the launcher script.

    Args:
        sys_args: Command line arguments (defaults to sys.argv[1:])

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    global has_failure, prometheus_folder_name
    try:
        # Parse only the arguments we care about and ignore the rest
        args, unknown = _parse_args()

        if unknown:
            logger.info("Ignoring unknown arguments: %s", unknown)

        # Get SageMaker environment information
        env = sagemaker_training.environment.Environment()
        logger.info(
            "Found SageMaker environment with hosts: %s", env.instance_groups_dict
        )
        logger.info("Current host: %s", env.current_host)

        # Copy and extract prometheus binary if path is provided
        if args.prometheus_path:
            prometheus_folder_name = _copy_prometheus_binary(args.prometheus_path)
            logger.info("Prometheus folder name set to: %s", prometheus_folder_name)

        # Set up Ray environment and run the specified script
        if env.is_hetero:
            exit_code = (
                _setup_ray_environment_heterogeneous_cluster(args, env)
                or SUCCESS_EXIT_CODE
            )
        else:
            exit_code = (
                _setup_ray_environment_homogeneous_cluster(args, env)
                or SUCCESS_EXIT_CODE
            )

        # Check if the job failed and raise an exception to ensure SageMaker marks it as failed
        if exit_code == DEFAULT_FAILURE_CODE or has_failure:
            has_failure = True
            failure_reason = "Unknown failure"
            if os.path.exists(FAILURE_REASON_PATH):
                try:
                    with open(FAILURE_REASON_PATH, "r") as f:
                        failure_reason = f.read().strip()
                    logger.error("Job failed with reason: %s", failure_reason)
                except Exception as e:
                    logger.warning("Could not read failure reason file: %s", e)

            # Raise an exception to ensure SageMaker marks the job as failed
            raise RuntimeError("Training job failed: %s" % failure_reason)

        return exit_code

    except Exception as e:
        has_failure = True
        logger.exception("Error encountered while running Ray launcher: %s", e)
        _write_failure_reason_file(str(e))
        raise


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        logger.error("Fatal error in launcher: %s", e)
        sys.exit(DEFAULT_FAILURE_CODE)
