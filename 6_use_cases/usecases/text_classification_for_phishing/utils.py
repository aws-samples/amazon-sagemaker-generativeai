"""
Utility functions for phishing detection model training and deployment.

This module contains helper functions for:
- S3 operations (upload/download)
- Model artifact extraction and compression
- File system operations
"""

import os
import tarfile
import shutil
from pathlib import Path
import boto3
from typing import Tuple


def download_and_extract_model(
    s3_client,
    bucket_name: str,
    model_tar_key: str,
    local_tar_path: str = "model.tar.gz",
    extract_dir: str = "uncompressed_model"
) -> str:
    """
    Download model.tar.gz from S3 and extract it locally.

    Args:
        s3_client: Boto3 S3 client
        bucket_name: S3 bucket name
        model_tar_key: S3 key for model.tar.gz
        local_tar_path: Local path to save downloaded tar file
        extract_dir: Directory to extract model files

    Returns:
        str: Path to extracted model directory
    """
    print(f"\nDownloading model from: s3://{bucket_name}/{model_tar_key}")

    # Download tar file
    s3_client.download_file(bucket_name, model_tar_key, local_tar_path)
    print(f"✅ Downloaded to {local_tar_path}")

    # Create extraction directory
    os.makedirs(extract_dir, exist_ok=True)

    # Extract tar file
    print(f"\nExtracting {local_tar_path} to {extract_dir}/")
    with tarfile.open(local_tar_path, 'r:gz') as tar:
        tar.extractall(path=extract_dir, filter='data')
    print(f"✅ Extracted successfully")

    return extract_dir


def upload_directory_to_s3(
    s3_client,
    local_dir: str,
    bucket_name: str,
    s3_prefix: str,
    verbose: bool = True
) -> int:
    """
    Upload a local directory to S3 recursively.

    Args:
        s3_client: Boto3 S3 client
        local_dir: Local directory path to upload
        bucket_name: S3 bucket name
        s3_prefix: S3 prefix (folder path) for uploaded files
        verbose: Whether to print progress for each file

    Returns:
        int: Number of files uploaded
    """
    if verbose:
        print(f"\nUploading {local_dir}/ to: s3://{bucket_name}/{s3_prefix}/")

    uploaded_files = 0
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_file_path, local_dir)
            s3_key = f"{s3_prefix}/{relative_path}"

            if verbose:
                print(f"  Uploading {relative_path}...", end='')

            s3_client.upload_file(local_file_path, bucket_name, s3_key)

            if verbose:
                print(" ✅")

            uploaded_files += 1

    if verbose:
        print(f"\n✅ Upload complete! {uploaded_files} files uploaded")
        print(f"\nUploaded files available at:")
        print(f"s3://{bucket_name}/{s3_prefix}/")

    return uploaded_files


def cleanup_local_files(*paths):
    """
    Remove local files and directories.

    Args:
        *paths: Variable number of file or directory paths to remove
    """
    print("\nCleaning up local files...")
    for path in paths:
        if os.path.isfile(path):
            os.remove(path)
            print(f"  Removed file: {path}")
        elif os.path.isdir(path):
            shutil.rmtree(path)
            print(f"  Removed directory: {path}")
    print("✅ Cleanup complete")


def find_latest_training_job(
    s3_client,
    bucket_name: str,
    base_job_name: str
) -> Tuple[str, str]:
    """
    Find the most recent training job folder in S3.

    Args:
        s3_client: Boto3 S3 client
        bucket_name: S3 bucket name
        base_job_name: Base name of training jobs

    Returns:
        Tuple[str, str]: (training_job_name, training_job_prefix)
    """
    print(f"Searching for latest training job with prefix: {base_job_name}")

    response = s3_client.list_objects_v2(
        Bucket=bucket_name,
        Prefix=base_job_name + '/',
        Delimiter='/'
    )

    training_job_prefixes = []
    if 'CommonPrefixes' in response:
        for prefix in response['CommonPrefixes']:
            prefix_name = prefix['Prefix'].rstrip('/')
            training_job_prefixes.append(prefix_name)

    if not training_job_prefixes:
        raise Exception(f"No training job folders found with prefix: {base_job_name}")

    # Get the most recent (sorted alphabetically, which works for timestamp-based names)
    latest_training_job_prefix = sorted(training_job_prefixes)[-1]
    training_job_name = latest_training_job_prefix.split('/')[-1]

    print(f"Found latest training job: {training_job_name}")

    return training_job_name, latest_training_job_prefix


def get_mlflow_app_arn(region: str) -> str:
    """
    Get the ARN of the most recent MLflow app in the region.

    Args:
        region: AWS region name

    Returns:
        str: MLflow app ARN

    Raises:
        Exception: If no MLflow app is found
    """
    sm_client = boto3.client('sagemaker', region_name=region)
    response = sm_client.list_mlflow_apps(
        MaxResults=1,
        SortBy='CreationTime',
        SortOrder='Descending'
    )

    if response['Summaries']:
        mlflow_app_arn = response['Summaries'][0]['Arn']
        print(f"✅ Found MLflow app: {mlflow_app_arn}")
        return mlflow_app_arn
    else:
        raise Exception(
            "No MLflow app found. Please create an MLflow app in SageMaker Console → MLflow"
        )
