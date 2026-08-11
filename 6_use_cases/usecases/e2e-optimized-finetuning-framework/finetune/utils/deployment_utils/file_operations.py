import os
import shlex
import shutil
import subprocess
import tarfile
from typing import Any, List

from finetune.utils.logging_util import get_logger

logger = get_logger(__name__)


def safe_extract_tar(tar: tarfile.TarFile, path: str) -> None:
    """
    Safely extract tar file preventing path traversal attacks.

    Args:
        tar: tarfile.TarFile object
        path: Extraction destination path

    Raises:
        ValueError: If path traversal attempt detected
    """
    for member in tar.getmembers():
        member_path = os.path.join(path, member.name)
        abs_member_path = os.path.abspath(member_path)
        abs_path = os.path.abspath(path)

        if not abs_member_path.startswith(abs_path + os.sep):
            raise ValueError(f"Path traversal attempt detected: {member.name}")

    tar.extractall(path=path)  # nosec B202 - validated above for path traversal


def setup_folder(folder_path: str) -> None:
    """
    Ensure the folder is set up by cleaning any existing content and creating a new folder.

    Args:
        folder_path (str): The directory path to set up.
    """
    try:
        if os.path.exists(folder_path):
            logger.debug(f"Folder {folder_path} exists. Deleting it for a clean setup.")
            shutil.rmtree(folder_path)
        os.makedirs(folder_path)
        logger.debug(f"New folder {folder_path} created.")
    except OSError as e:
        logger.error(f"Failed to set up folder {folder_path}: {e}")
        raise e


def extract_tarfile(file_path: str, destination_path: str) -> None:
    """
    Extract a .tar.gz file to the specified folder and remove the tar file after extraction.

    Args:
        file_path (str): Path to the .tar.gz file to be extracted.
        destination_path (str): Path where the contents should be extracted.

    Raises:
        Exception: If extraction fails.
    """
    try:
        logger.info(f"Extracting {file_path} to {destination_path}...")
        with tarfile.open(file_path, "r:gz") as tar:
            safe_extract_tar(tar, destination_path)
        os.remove(file_path)
        logger.info(f"Successfully extracted {file_path} and removed the tar file.")
    except (tarfile.TarError, OSError, ValueError) as e:
        logger.error(f"Error extracting {file_path} to {destination_path}: {e}")
        raise e


def cleanup_existing_files(file_paths: List[str]) -> None:
    """
    Remove specified files if they exist in local dev desktop.

    Args:
        file_paths (list of str): List of file paths to check and delete.

    Raises:
        OSError: If file removal fails.
    """
    try:
        if os.environ.get("HOSTNAME", "").startswith("dev-dsk"):
            for file_path in file_paths:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.debug(f"Deleted existing file: {file_path}")
    except OSError as e:
        logger.error(f"Failed to delete files: {e}")
        raise e


def upload_folder_to_s3(
    local_folder_path: str,
    destination_bucket_name: str,
    destination_folder_path: str,
    aws_manager: Any,
) -> None:
    """
    Upload a folder to an S3 bucket.

    Args:
        local_folder_path (str): Path of the local folder to upload.
        destination_bucket_name (str): S3 bucket name.
        destination_folder_path (str): Path of the s3 folder to be uploaded.
        aws_manager (Any):

    Raises:
        Exception: If the upload command fails.
    """
    try:
        profile_flag = f"--profile {aws_manager.profile_name}" if aws_manager.profile_name else ""
        region_flag = f"--region {aws_manager.region}" if aws_manager.region else ""
        command = (
            f"{aws_manager.write_cmd_credentials()} aws s3 cp --recursive {profile_flag} {region_flag} {local_folder_path} "
            f"s3://{destination_bucket_name}/{destination_folder_path}"
        )
        subprocess.run(shlex.split(command), shell=False, check=True)
        logger.info(f"Folder {local_folder_path} uploaded to S3 successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to upload {local_folder_path} to S3: {e}")
        raise e
