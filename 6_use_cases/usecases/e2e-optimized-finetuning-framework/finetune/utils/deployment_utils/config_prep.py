import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict

from finetune.utils.deployment_utils.file_operations import (
    cleanup_existing_files,
    extract_tarfile,
    setup_folder,
    upload_folder_to_s3,
)
from finetune.utils.logging_util import get_logger

logger = get_logger(__name__)


class ConfigPreparationError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f"Faster Transformer Configration Preparation Error: {self.message}"


class FasterTransformerConfigWrittingError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f"Faster Transformer Configration Writting Error: {self.message}"


def upload_trained_model_to_s3(
    configs: Dict, training_job_name: str, source_aws_manager: Any,
    deploy_aws_manager: Any, s3_folder_existance: bool
) -> None:
    """
    Download the trained model from the training bucket and upload to the deployment bucket.

    Args:
        configs (Dict): Configuration dictionary.
        training_job_name (str): Name of the training job.
        source_aws_manager (AWSManager): AWS manager for the training region (to download model).
        deploy_aws_manager (AWSManager): AWS manager for the deployment region (to upload model).
        s3_folder_existance (bool): Flag to check if the folder already exists on S3.

    Raises:
        Exception: If an error occurs during file operations or AWS interactions.
    """
    destination_bucket_name = configs["faster_transformer"]["s3_faster_transformer_bucket"]
    source_object_key = os.path.join(
        configs["data"]["s3_root_folder"], "results", training_job_name, "output/model.tar.gz"
    )

    if not s3_folder_existance:
        # Use Python's tempfile for secure temporary directory
        temp_base = Path(tempfile.gettempdir()) / "fine_tune" / training_job_name
        local_folder_path = str(temp_base)
        destination_folder_path = f"{training_job_name}"
        file_name = "model.tar.gz"
        file_path = str(temp_base / file_name)

        # Ensure clean folder setup
        setup_folder(local_folder_path)

        # Download from training bucket (source region)
        logger.info(f"Downloading model from S3: {source_object_key} to {file_path}")
        _ = source_aws_manager.connect_to_s3()
        source_aws_manager.download_file_from_s3(source_object_key, file_path)
        logger.info("Model downloaded successfully.")

        # Extract the .tar.gz file
        extract_tarfile(file_path, local_folder_path)

        # Upload to deployment bucket (deployment region)
        upload_folder_to_s3(
            local_folder_path, destination_bucket_name, destination_folder_path, deploy_aws_manager
        )

        # Clean up local folder
        shutil.rmtree(local_folder_path)
        logger.info(
            f"Files from {source_object_key} uploaded to "
            f"{destination_bucket_name}/{training_job_name}."
        )

    else:
        logger.info(
            f"The model has already been uploaded to "
            f"{destination_bucket_name}/{training_job_name}."
        )


def prepare_fastertransformer_config(
    configs: Dict, aws_manager: Any, training_job_name: str
) -> None:
    """
    Prepare the FasterTransformer configuration file and upload it to S3.

    Args:
        configs (Dict): Configuration dictionary.
        aws_manager (AWSManager): AWS manager instance to handle S3 operations.
        training_job_name (str): Name of the training job.
    """
    dirname = os.path.dirname(__file__)
    config_dir = os.path.join(dirname, "../../fastertransformer-t5")

    properties_file = os.path.join(config_dir, "serving.properties")
    tar_file = os.path.join(config_dir, "fastertransformer-t5.tar.gz")

    s3_faster_transformer_folder = (
        f"{configs['faster_transformer']['s3_faster_transformer_bucket']}/" f"{training_job_name}"
    )

    try:
        # Delete existing properties and tar files if they exist
        cleanup_existing_files([properties_file, tar_file])

        # Create serving.properties file
        properties = {
            "engine": "FasterTransformer",
            "option.tensor_parallel_degree": configs["faster_transformer"][
                "tensor_parallel_degree"
            ],
            "option.model_id": f"s3://{s3_faster_transformer_folder}",
            "option.dtype": "fp32",
        }

        # in remote env, there is no write access, so move whole fastertransformer-t5 folder to temp folder
        if not os.environ.get("HOSTNAME", "").startswith("dev-dsk"):
            temp_base = Path(tempfile.gettempdir()) / "fine_tune" / training_job_name
            tmp_fastertransformer_t5_folder_path = str(temp_base / "fastertransformer-t5")
            copy_folder(config_dir, tmp_fastertransformer_t5_folder_path)
            properties_file = str(Path(tmp_fastertransformer_t5_folder_path) / "serving.properties")
            tar_file = str(
                Path(tmp_fastertransformer_t5_folder_path) / "fastertransformer-t5.tar.gz"
            )

        write_properties_file(properties, properties_file)

        # Create .tar.gz of FasterTransformer configuration
        with tarfile.open(tar_file, "w:gz") as tar:
            final_tar_path = (
                config_dir
                if os.environ.get("HOSTNAME", "").startswith("dev-dsk")
                else tmp_fastertransformer_t5_folder_path
            )
            tar.add(final_tar_path, arcname=os.path.basename(final_tar_path))
        logger.info(f"{tar_file} created successfully.")

        # Upload tar file to S3
        aws_manager.upload_file_to_s3(
            tar_file,
            configs["faster_transformer"]["s3_faster_transformer_bucket"],
            "fastertransformer-t5.tar.gz",
        )
        logger.info("FasterTransformer configuration uploaded to S3 successfully.")

    except Exception as e:
        logger.error(f"Failed to prepare FasterTransformer configuration: {e}")
        raise ConfigPreparationError(str(e))


def write_properties_file(properties: Dict[str, str], file_path: str) -> None:
    """
    Write properties to a serving.properties file.

    Args:
        properties (Dict[str, str]): Dictionary of properties to write.
        file_path (str): Path to the properties file.
    """
    try:
        with open(file_path, "w") as file:
            for key, value in properties.items():
                file.write(f"{key}={value}\n")
        logger.info(f"{file_path} created successfully.")
    except Exception as e:
        logger.error(f"Failed to write properties file: {e}")
        raise FasterTransformerConfigWrittingError(str(e))


def copy_folder(src_path: str, dst_path: str, overwrite: bool = False) -> bool:
    """
    Copy a folder from source to destination.

    Args:
        src_path (str): Source folder path
        dst_path (str): Destination folder path
        overwrite (bool): If True, overwrite existing destination folder

    Returns:
        bool: True if successful, False otherwise

    Raises:
        FileNotFoundError: If source folder doesn't exist
    """
    try:
        # Convert to Path objects for better path handling
        src = Path(src_path)
        dst = Path(dst_path)

        # Check if source exists
        if not src.exists():
            raise FileNotFoundError(f"Source folder does not exist: {src_path}")

        # Check if destination already exists
        if dst.exists():
            if overwrite:
                shutil.rmtree(dst)
            else:
                logger.info(f"Destination folder already exists: {dst_path}")
                return False

        # Create parent directories if they don't exist
        dst.parent.mkdir(parents=True, exist_ok=True)

        # Copy the folder
        shutil.copytree(src, dst)

        print(f"Successfully copied folder from {src_path} to {dst_path}")
        return True

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return False
    except PermissionError:
        print(f"Error: Permission denied. Check folder permissions.")
        return False
    except shutil.Error as e:
        print(f"Error during copy operation: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False
