from typing import Any, Dict

from botocore.exceptions import ClientError

from finetune.utils.deployment_utils.config_prep import (
    prepare_fastertransformer_config,
    upload_trained_model_to_s3,
)
from finetune.utils.logging_util import get_logger

logger = get_logger(__name__)

# Deployment bucket hard requirement: must be in us-west-2
_DEPLOYMENT_REGION = "us-west-2"


def _ensure_deployment_bucket_exists(configs: Dict, multi_region_aws_manager: Any) -> None:
    """
    Ensure the FasterTransformer deployment S3 bucket exists in us-west-2.
    Creates it if it doesn't exist.
    """
    bucket = configs["faster_transformer"]["s3_faster_transformer_bucket"]
    aws_manager = multi_region_aws_manager.get_aws_manager(_DEPLOYMENT_REGION)
    aws_manager.connect_to_s3()
    s3_client = aws_manager.s3_client

    try:
        s3_client.head_bucket(Bucket=bucket)
        logger.info(f"Deployment bucket '{bucket}' already exists.")
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code in ("404", "NoSuchBucket"):
            logger.info(f"Deployment bucket '{bucket}' not found. Creating in {_DEPLOYMENT_REGION}...")
            try:
                s3_client.create_bucket(
                    Bucket=bucket,
                    CreateBucketConfiguration={"LocationConstraint": _DEPLOYMENT_REGION},
                )
                logger.info(f"Deployment bucket '{bucket}' created in {_DEPLOYMENT_REGION}.")
            except ClientError as create_err:
                logger.error(f"Failed to create deployment bucket '{bucket}': {create_err}")
                raise
        else:
            logger.error(f"Error checking deployment bucket '{bucket}': {e}")
            raise


def model_deployment(
    configs: Dict[str, Dict[str, Any]],
    training_job_name: str,
    endpoint_name: str,
    multi_region_aws_manager: Any,
    s3_folder_existance: bool = False,
):
    """
    Deploy the trained model to an S3 bucket, create configuration files for FasterTransformer, and deploy the model.

    Args:
        configs (dict): Configuration dictionary containing AWS and model settings.
        training_job_name (str): Name of the training job.
        endpoint_name (str): Name of the endpoint to be deployed.
        aws_manager (AWSManager): Instance to manage AWS services.
        s3_folder_existance (bool): Flag to check if the folder already exists on S3.

    Raises:
        Exception: If any error occurs during file operations or AWS interactions.
    """
    try:
        # Ensure deployment bucket exists in us-west-2 before any S3 operations
        _ensure_deployment_bucket_exists(configs, multi_region_aws_manager)

        if not s3_folder_existance:
            logger.info("Preparing for the model deplpyment using Faster Transformer.")
            # Download model from training bucket (us-east-1), upload to deployment bucket (us-west-2)
            source_aws_manager = multi_region_aws_manager.get_aws_manager("us-east-1")
            deploy_aws_manager = multi_region_aws_manager.get_aws_manager(_DEPLOYMENT_REGION)
            upload_trained_model_to_s3(
                configs,
                training_job_name,
                source_aws_manager,
                deploy_aws_manager,
                s3_folder_existance,
            )

            # Prepare the FasterTransformer configuration file
            prepare_fastertransformer_config(
                configs, deploy_aws_manager, training_job_name
            )

        # Deploy the Fast Transformer Endpoint
        if configs["faster_transformer"]["deployment"]:
            from finetune.utils.deployment_utils.deployment_prep import (
                deployment_fastertransformer_main,
            )

            logger.info(
                "Deploying FasterTransformer endpoint using deployment_fasttransformer_config.yaml"
            )
            deployment_fastertransformer_main(configs, training_job_name, endpoint_name, multi_region_aws_manager)

        logger.info(f"Endpoint {endpoint_name} has been successfully deployed.")

    except Exception as e:
        logger.error(f"Error during deployment: {e}", exc_info=True)
        raise e
