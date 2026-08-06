from finetune.services import MultiRegionAWSManager
from finetune.utils import import_yaml_files, model_deployment
from finetune.utils.logging_util import get_logger

logger = get_logger(__name__)


def deployment_main(model_name, endpoint_name, s3_folder_existance, profile=None, role=None):

    configs = import_yaml_files()
    # configs is a dict where keys are YAML file names and values are the parsed content of each YAML file
    
    # Add profile to configs if provided
    if profile:
        if "aws_setup" not in configs:
            configs["aws_setup"] = {}
        configs["aws_setup"]["profile_name"] = profile
    
    # Add role to configs if provided
    if role:
        if "aws_setup" not in configs:
            configs["aws_setup"] = {}
        configs["aws_setup"]["role"] = role
    
    logger.info(configs)

    try:
        multi_region_aws_manager = MultiRegionAWSManager(configs, is_sagemaker=True)

    except Exception as e:
        error_message = f"Exception caught when connecting to AWS services: {e}"
        logger.error(error_message)
        raise Exception(error_message)

    logger.info("successfully connect to aws account")

    # Initialize metadata tracking
    from finetune.utils.metadata_manager import MetadataManager
    metadata = MetadataManager(
        bucket=configs["aws_setup"]["default_bucket"],
        profile=profile
    )
    logger.info("Metadata tracking initialized for deployment")

    # starting the model deployment
    model_deployment(
        configs,
        training_job_name=model_name,
        endpoint_name=endpoint_name,
        multi_region_aws_manager=multi_region_aws_manager,
        s3_folder_existance=s3_folder_existance,
    )
    
    # Save deployment metadata
    metadata.record_deployment(
        model_name=model_name,
        endpoint_name=endpoint_name
    )
    logger.info(f"Deployment metadata saved for endpoint: {endpoint_name}")
