import os
import time
from typing import Dict, Optional

import boto3
import sagemaker
from sagemaker import deserializers, serializers
from sagemaker.model import Model
from finetune.services.awsmanager_service import MultiRegionAWSManager

from finetune.utils.logging_util import get_logger

logger = get_logger(__name__)


class EndpointDeploymentError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f"Endpoint Deployment Error: {self.message}"


class PredictionError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f"Predition Error: {self.message}"


def create_model(configs: Dict) -> Model:
    """
    Create a SageMaker model for FasterTransformer.

    Args:
        configs (Dict): Configuration dictionary.

    Returns:
        Model: A SageMaker model object.
    """
    model_data = os.path.join(
        "s3://",
        configs["faster_transformer"]["s3_faster_transformer_bucket"],
        configs["faster_transformer"]["faster_transformer_file"],
    )

    model = Model(
        image_uri=configs["faster_transformer"]["image_uri"],
        model_data=model_data,
        role=configs["faster_transformer"]["role"],
        name=configs["faster_transformer"]["model_name"],
        sagemaker_session=configs["faster_transformer"]["sagemaker_session"],
    )

    logger.info(f"Model created with name: {configs['faster_transformer']['model_name']}")
    return model


def deploy_model(configs: Dict) -> sagemaker.Predictor:
    """
    Deploy the SageMaker model to an endpoint.

    Args:
        configs (Dict): Configuration dictionary.

    Returns:
        sagemaker.Predictor: A SageMaker predictor for the deployed model.
    """
    try:
        configs["faster_transformer"]["model"].deploy(
            initial_instance_count=configs["faster_transformer"]["initial_instance_count"],
            instance_type=configs["faster_transformer"]["deployment_instance"],
            endpoint_name=configs["faster_transformer"]["endpoint_name"],
            model_data_download_timeout=1800,  # 30 minutes for model download
            container_startup_health_check_timeout=3600,  # 1 hour for FasterTransformer model loading
        )

        predictor = sagemaker.Predictor(
            endpoint_name=configs["faster_transformer"]["endpoint_name"],
            sagemaker_session=configs["faster_transformer"]["sagemaker_session"],
            serializer=serializers.JSONSerializer(),
            deserializer=deserializers.JSONDeserializer(),
        )

        logger.info(f"Model deployed to endpoint: {configs['faster_transformer']['endpoint_name']}")
    except Exception as e:
        logger.exception(e)
        raise EndpointDeploymentError(str(e))
    return predictor


def prediction(configs: Dict) -> None:
    """
    Make a prediction using the deployed model.

    Args:
        configs (Dict): Configuration dictionary.
    """
    start_time = time.time()
    try:
        response = configs["faster_transformer"]["predictor"].predict(
            {
                "text": configs["faster_transformer"]["input_text"],
                "parameters": {
                    "temperature": configs["faster_transformer"]["temperature"],
                    "max_seq_len": configs["faster_transformer"]["max_seq_len"],
                },
            }
        )
        logger.info(f"Generated text: {response}")
    except Exception as e:
        logger.exception(e)
        raise PredictionError(str(e))

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Execution time: {elapsed_time:.2f} seconds")


def deployment_fastertransformer_main(
    configs: Dict, model_name: Optional[str] = None, endpoint_name: Optional[str] = None, multi_region_aws_manager=None
) -> None:
    """
    Main function to handle model deployment and prediction for FasterTransformer.

    Args:
        configs (Dict): Configuration dictionary.
        model_name (Optional[str]): Optional model name for deployment.
        endpoint_name (Optional[str]): Optional endpoint name for deployment.
    """

    def generate_error_message(name: str) -> str:
        return f"Please provide the {name} name either to the function or update the deployment_fasttransformer_config.yaml"

    # Handle model_name
    if not configs["faster_transformer"]["model_name"]:
        configs["faster_transformer"]["model_name"] = model_name
        if not configs["faster_transformer"]["model_name"]:
            error_message = generate_error_message("model")
            logger.error(error_message)
            raise Exception(error_message)

    # Handle endpoint_name
    if not configs["faster_transformer"]["endpoint_name"]:
        configs["faster_transformer"]["endpoint_name"] = endpoint_name
        if not configs["faster_transformer"]["endpoint_name"]:
            error_message = generate_error_message("endpoint")
            logger.error(error_message)
            raise Exception(error_message)

    # Create SageMaker session and role
    # Use the passed multi_region_aws_manager if available, otherwise create new one
    if multi_region_aws_manager:
        aws_manager = multi_region_aws_manager.get_aws_manager("us-west-2")
    else:
        aws_manager = MultiRegionAWSManager(configs, is_sagemaker=True).get_aws_manager("us-west-2")
    
    sagemaker_session = aws_manager.create_sagemaker_session()
    configs["faster_transformer"]["sagemaker_session"] = sagemaker_session
    
    # Use role from aws_manager - it will have already validated role exists
    configs["faster_transformer"]["role"] = aws_manager.role
    
    if not configs["faster_transformer"]["role"]:
        raise ValueError(
            "SageMaker execution role is required for deployment. "
            "Provide via --role argument, 'role' in aws_setup.yaml, or SAGEMAKER_ROLE environment variable."
        )
    
    logger.info(
        f"SageMaker session: {configs['faster_transformer']['sagemaker_session']}, "
        f"Role: {configs['faster_transformer']['role']}"
    )

    # Create model
    configs["faster_transformer"]["model"] = create_model(configs)

    # SageMaker client for retry logic
    sm_client = boto3.Session(region_name=configs["faster_transformer"]["aws_region"]).client(
        "sagemaker"
    )

    # Retry logic for model deployment
    retry = 0
    retry_remaining_count = 20
    while retry_remaining_count > 0:
        try:
            configs["faster_transformer"]["predictor"] = deploy_model(configs)
            retry_remaining_count = 0
        except Exception as e:
            error_message = str(e)
            if "in-progress" in error_message:
                logger.warning(f"{endpoint_name} is still in progress. Retrying after 2 minutes...")
                time.sleep(120)
            else:
                logger.error(f"Error during deployment: {e}")
                # Cleanup endpoint and configuration in case of failure
                try:
                    # Create fresh client to avoid expired credentials
                    sm_client_fresh = boto3.Session(
                        region_name=configs["faster_transformer"]["aws_region"]
                    ).client("sagemaker")
                    
                    sm_client_fresh.delete_endpoint_config(EndpointConfigName=endpoint_name)
                    logger.info(f"Successfully deleted endpoint config: {endpoint_name}")
                    sm_client_fresh.delete_endpoint(EndpointName=endpoint_name)
                    logger.info(f"Successfully deleted endpoint: {endpoint_name}")
                except Exception as cleanup_error:
                    logger.error(f"Failed to cleanup resources: {cleanup_error}")
                
                retry += 1
                retry_remaining_count -= 1
                logger.info(
                    f"Retrying deployment... attempt {retry}, with remaining {retry_remaining_count} attempts."
                )

    # Make a prediction after deployment
    if "predictor" in configs.get("faster_transformer", {}):
        prediction(configs)
    else:
        logger.warning("Predictor not available, skipping prediction test")
