import os
import uuid
from pathlib import Path
from typing import Any, Dict

import yaml

from finetune.utils.logging_util import get_logger
from finetune.utils.training_utils.training_constants import TOKENIZERS_PARALLELISM

logger = get_logger(__name__)

# Resolve configs from the finetune package root (utils -> finetune -> configs)
CONFIGS_DIR: Path = (Path(__file__).resolve().parent.parent / "configs")

# Default base names used in the shipped config templates.
# If the user hasn't changed these, we append a machine-unique suffix automatically.
_DEFAULT_BASE_NAMES = {"finetune-blog"}
_DEFAULT_DEPLOYMENT_BUCKET_NAMES = {"finetune-blog-deployment"}


def _generate_unique_suffix() -> str:
    """
    Generate a short, deterministic, machine-unique suffix derived from the
    MAC address (uuid.getnode()).  The same machine always produces the same
    suffix, so S3 names stay stable across runs.
    """
    mac = uuid.getnode()
    return format(mac, "012x")[-12:]


def _apply_unique_bucket_name(configs: Dict[str, Dict[str, Any]]) -> None:
    """
    Ensure ``default_bucket`` in aws_setup is globally unique by appending a
    machine-derived suffix when it still carries the shipped default value.

    Users who manually set a custom bucket name in the YAML are not affected.
    """
    aws_cfg = configs.get("aws_setup", {})
    original_bucket = aws_cfg.get("default_bucket", "")

    if original_bucket not in _DEFAULT_BASE_NAMES:
        return  # User already customized the bucket name

    suffix = _generate_unique_suffix()
    new_bucket = f"{original_bucket}-{suffix}"
    aws_cfg["default_bucket"] = new_bucket
    logger.info(f"Auto-generated unique default_bucket: {new_bucket}")

    # Also apply to the deployment bucket if it still has the default name
    ft_cfg = configs.get("faster_transformer", {})
    deploy_bucket = ft_cfg.get("s3_faster_transformer_bucket", "")
    if deploy_bucket in _DEFAULT_DEPLOYMENT_BUCKET_NAMES:
        new_deploy_bucket = f"{deploy_bucket}-{suffix}"
        ft_cfg["s3_faster_transformer_bucket"] = new_deploy_bucket
        logger.info(f"Auto-generated unique s3_faster_transformer_bucket: {new_deploy_bucket}")


def _apply_data_defaults(configs: Dict[str, Dict[str, Any]]) -> None:
    """
    Apply default values for optional data config fields.

    ``s3_root_folder`` defaults to ``finetune-blog`` if not specified in data.yaml.
    Users can still override it by adding the key back to their YAML.
    """
    data_cfg = configs.get("data", {})
    if "s3_root_folder" not in data_cfg:
        data_cfg["s3_root_folder"] = "finetune-blog"
        logger.info("Using default s3_root_folder: finetune-blog")


def check_configs_and_files() -> None:
    """
    Verifies the existence of the 'configs' folder and required configuration files.

    Raises:
        FileNotFoundError: If the 'configs' folder or any required files are missing.
    """
    required_files: list[str] = ["aws_setup.yaml", "data.yaml"]

    if not CONFIGS_DIR.is_dir():
        logger.error(f"The '{CONFIGS_DIR}' folder does not exist.")
        raise FileNotFoundError(f"The '{CONFIGS_DIR}' folder does not exist.")

    missing_files: list[str] = [
        f for f in required_files if not (CONFIGS_DIR / f).is_file()
    ]

    if missing_files:
        missing_files_str: str = ", ".join(missing_files)
        logger.error(f"Missing required files in '{CONFIGS_DIR}': {missing_files_str}")
        raise FileNotFoundError(
            f"Missing required files in '{CONFIGS_DIR}': {missing_files_str}"
        )

    logger.info("All required files are present in the 'configs' folder.")


def load_yaml_file(yaml_file: Path) -> Dict[str, Any]:
    """
    Load a single YAML file and return its content as a dictionary.

    Args:
        yaml_file: The file path to the YAML file.

    Returns:
        Parsed YAML content.

    Raises:
        yaml.YAMLError: If there is an error during YAML parsing.
    """
    try:
        with yaml_file.open("r") as file:
            return yaml.safe_load(file)
    except yaml.YAMLError as exc:
        logger.error(f"Error loading {yaml_file}: {exc}")
        raise exc


def import_yaml_files() -> Dict[str, Dict[str, Any]]:
    """
    Load all YAML files from a given directory and return their contents in a dictionary.

    Returns:
        A dictionary where keys are file names (without extensions) and values are the contents of the YAML files.
    """
    all_configs: Dict[str, Dict[str, Any]] = {}

    if not CONFIGS_DIR.exists() or not CONFIGS_DIR.is_dir():
        logger.error(f"The provided path '{CONFIGS_DIR}' is not a valid directory.")
        return all_configs

    for yaml_file in CONFIGS_DIR.glob("*.yaml"):
        try:
            config_data: Dict[str, Any] = load_yaml_file(yaml_file)
            all_configs[yaml_file.stem] = config_data
        except yaml.YAMLError as e:
            # Continue to the next file if there's an error
            logger.error(f"Error loading YAML file '{yaml_file.name}': {e}")
            continue

    # Auto-generate unique S3 bucket name for users who haven't customized it
    _apply_unique_bucket_name(all_configs)

    # Apply defaults for optional data config fields
    _apply_data_defaults(all_configs)

    return all_configs


def load_env_config(configs: Dict[str, Dict[str, str]]) -> None:
    """
    Load environment configurations and set OS environment variables for parallelism and module directory.

    Args:
        configs: Configuration dictionary containing AWS credentials and input data paths.
    """
    # Set environment variable for tokenizer parallelism
    os.environ["TOKENIZERS_PARALLELISM"] = TOKENIZERS_PARALLELISM
    logger.info(f'Set TOKENIZERS_PARALLELISM to {os.environ["TOKENIZERS_PARALLELISM"]}')

    # Construct the SM_MODULE_DIR from the config and set the environment variable
    sm_module_dir: str = os.path.join(
        configs["aws_setup"]["default_bucket"],
        configs["data"]["s3_root_folder"],
        configs["data"]["s3_input_data_folder"],
    )
    os.environ["SM_MODULE_DIR"] = sm_module_dir
    logger.info(f'Set SM_MODULE_DIR to {os.environ["SM_MODULE_DIR"]}')
