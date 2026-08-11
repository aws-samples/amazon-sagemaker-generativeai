import os
from typing import Any, Dict, Optional

from sagemaker.huggingface import HuggingFace

from finetune.constants import PEFT_METHODS, TOKENIZERS_PARALLELISM
from finetune.services import AWSManager
from finetune.utils import data_generate_main, import_yaml_files, load_env_config
from finetune.utils.logging_util import get_logger

logger = get_logger(__name__)


class TrainingError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f"Model Fine Tuning Error: {self.message}"


class AWSServiceConnectionError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f"AWS Service Connection Error: {self.message}"


class DataUpdateError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f"Data Update Error: {self.message}"


def set_environment_vars() -> None:
    """
    Set necessary environment variables for tokenizers and others.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = TOKENIZERS_PARALLELISM
    logger.info("Set environment variables for tokenizers parallelism.")


def load_configs() -> Dict[str, Any]:
    """
    Load YAML configuration files from the specified path.

    Args:
        config_file_path (str): The path to the configuration files.

    Returns:
        Dict[str, Any]: Parsed configuration data from YAML files.
    """
    configs: Dict[str, Any] = import_yaml_files()
    logger.debug(f"Configuration loaded from config: {configs}")
    return configs


def connect_to_aws_services(configs: Dict[str, Any], profile: str = None, role: str = None) -> AWSManager:
    """
    Initialize and connect to AWS services like SageMaker and S3.

    Args:
        configs (Dict[str, Any]): Parsed configuration data.
        profile (str, optional): AWS profile name. Uses default if not specified.
        role (str, optional): SageMaker execution role ARN. Auto-detected or from config/env if not provided.

    Returns:
        AWSManager: AWS Manager instance connected to services.

    Raises:
        Exception: If unable to connect to AWS services.
    """
    try:
        aws_manager = AWSManager(configs, is_sagemaker=True, profile_name=profile, role_arn=role)
        aws_manager.create_sagemaker_session()
        aws_manager.connect_to_s3()
        logger.info("Successfully connected to AWS services.")
        return aws_manager
    except Exception as e:
        error_message = f"Error connecting to AWS services: {e}"
        logger.error(error_message, exc_info=True)
        raise AWSServiceConnectionError(str(e))


def update_dataset(
    configs: Dict[str, Any], aws_manager: AWSManager, train_dataset_quality_check: bool
) -> None:
    """
    Update the training dataset based on configuration, or use an existing dataset.

    Args:
        configs (Dict[str, Any]): Parsed configuration data.
        aws_manager (AWSManager): Instance of AWSManager to interact with AWS services.
        train_dataset_quality_check (bool): the flag to indicate whether training dataset quality check is required.

    Raises:
        Exception: If dataset update fails.
    """
    if configs["data"].get("update_input_data", False):
        try:
            data_generate_main(configs, aws_manager, train_dataset_quality_check)
            logger.info("Successfully updated the training dataset.")
        except Exception as e:
            error_message = f"Error generating new dataset: {e}"
            logger.error(error_message, exc_info=True)
            raise DataUpdateError(str(e))
    else:
        logger.info("Using existing training dataset as specified in the configuration.")


def start_training_job(configs: Dict[str, Any], aws_manager: Any) -> None:
    """
    Start the HuggingFace model fine-tuning job using SageMaker.

    Args:
        configs (Dict[str, Any]): Parsed configuration data.

    Raises:
        Exception: If training job fails.
    """
    load_env_config(configs)

    # Define the training job name
    training_job_name: Optional[str] = configs["training"].get("train_job_name")
    if not training_job_name:
        training_type = (
            configs["training"].get("training_type", "fft")
            if configs["training"].get("training_type") in PEFT_METHODS
            else "fft"
        ).replace("_", "-")
        training_job_name = f"{configs['training']['model_id'].split('/')[-1]}-{training_type}"
    logger.info(f"Training job name is: {training_job_name}")

    # Create HuggingFace Estimator
    huggingface_estimator = HuggingFace(
        source_dir=os.path.join(os.path.dirname(__file__), ".."),
        entry_point=os.path.join("finetune", configs["training"]["train_script"]),
        instance_type=configs["training"]["training_instance"],
        instance_count=1,
        base_job_name=training_job_name,
        role=aws_manager.role,
        sagemaker_session=aws_manager.sagemaker_session,
        volume_size=200,
        transformers_version=configs["training"]["package_versions"]["transformers_version"],
        pytorch_version=configs["training"]["package_versions"]["pytorch_version"],
        py_version=configs["training"]["package_versions"]["py_version"],
        output_path=f"s3://{configs['aws_setup']['default_bucket']}/{configs['data']['s3_root_folder']}/results/",
        disable_profiler=True,
        debugger_hook_config=False,
    )

    # Define data path for training
    data_path: Dict[str, str] = {
        "train_data": os.path.join(
            "s3://",
            configs["aws_setup"]["default_bucket"],
            configs["data"]["s3_root_folder"],
            configs["data"]["s3_input_data_folder"],
        )
    }

    try:
        logger.info("Submitting training job (wait=False for robust polling)")
        huggingface_estimator.fit(data_path, wait=False)
        training_job_name = huggingface_estimator._current_job_name
        logger.info(f"Training job submitted: {training_job_name}")
        
        # Manual polling for completion using the same session
        import time
        sagemaker_client = aws_manager.sagemaker_session.sagemaker_client
        
        logger.info("Polling for training job completion...")
        while True:
            response = sagemaker_client.describe_training_job(TrainingJobName=training_job_name)
            status = response["TrainingJobStatus"]
            
            if status in ["Completed", "Failed", "Stopped"]:
                logger.info(f"Training job {status.lower()}: {training_job_name}")
                
                if status == "Completed":
                    logger.info(f"Model artifacts: {response.get('ModelArtifacts', {}).get('S3ModelArtifacts', 'N/A')}")
                    break
                elif status == "Failed":
                    failure_reason = response.get('FailureReason', 'Unknown')
                    logger.error(f"Training job failed: {failure_reason}")
                    raise TrainingError(f"Training job failed: {failure_reason}")
                else:  # Stopped
                    logger.error("Training job was stopped")
                    raise TrainingError("Training job was stopped")
            else:
                secondary_status = response.get("SecondaryStatus", "Unknown")
                logger.info(f"Status: {status} | Secondary Status: {secondary_status}")
                time.sleep(60)  # Poll every 60 seconds
                
    except TrainingError:
        raise
    except Exception as e:
        error_message = f"Error during training job: {e}"
        logger.error(error_message, exc_info=True)
        raise TrainingError(str(e))

    logger.info(f"Returning training job name: {training_job_name}")
    return training_job_name


def training_main(train_dataset_quality_check: bool, profile: str = None, role: str = None) -> str:
    """
    Main function to start the fine-tuning process.

    Args:
        train_dataset_quality_check (bool): the flag to indicate whether training dataset quality check is required.
        profile (str, optional): AWS profile name. Uses default if not specified.
        role (str, optional): SageMaker execution role ARN. Auto-detected in SageMaker or from config/env.

    Returns:
        str: the training job name or empty string if only checking the quality of training dataset.
    """
    logger.info("Starting the fine-tuning process.")

    # Set environment variables
    set_environment_vars()

    # Load configuration files
    configs: Dict[str, Any] = load_configs()

    # Connect to AWS services
    aws_manager = connect_to_aws_services(configs, profile, role)

    # Initialize metadata tracking
    from finetune.utils.metadata_manager import MetadataManager
    metadata = MetadataManager(
        bucket=configs["aws_setup"]["default_bucket"],
        profile=profile
    )
    logger.info("Metadata tracking initialized")

    # Update the dataset if required
    update_dataset(configs, aws_manager, train_dataset_quality_check)

    # Start the model training job
    if not train_dataset_quality_check:
        logger.info("Calling start_training_job()")
        training_job_name = start_training_job(configs, aws_manager)
        logger.info(f"start_training_job() returned: {training_job_name}")
        
        # Save training metadata (non-critical, don't fail if this crashes)
        try:
            logger.info("About to save training metadata")
            metadata.record_training(
                job_name=training_job_name,
                model_id=configs["training"]["model_id"],
                training_type=configs["training"]["training_type"]
            )
            logger.info(f"Training metadata saved for job: {training_job_name}")
        except Exception as e:
            logger.warning(f"Failed to save training metadata: {e}", exc_info=True)
        
        logger.info(f"About to return training_job_name: {training_job_name}")
        return training_job_name

    return ""
