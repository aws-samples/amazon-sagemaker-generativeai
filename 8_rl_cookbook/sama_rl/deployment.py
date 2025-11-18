"""
Model deployment utilities for SAMA RL
"""
import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker import get_execution_role
import logging

logger = logging.getLogger(__name__)


class ModelDeployer:
    """Deploy trained GRPO models to SageMaker endpoints"""
    
    def __init__(self, model_data_s3_path: str, base_model_name: str = "Qwen/Qwen2-0.5B-Instruct"):
        """
        Args:
            model_data_s3_path: S3 path to trained model artifacts
            base_model_name: Base model name for tokenizer/config
        """
        self.model_data_s3_path = model_data_s3_path
        self.base_model_name = base_model_name
        self.endpoint_name = None
        self.model = None
    
    def deploy_endpoint(self, instance_type: str = "ml.m5.xlarge", instance_count: int = 1) -> str:
        """
        Deploy model to SageMaker endpoint using HuggingFace LLM container
        
        Args:
            instance_type: Instance type for endpoint
            instance_count: Number of instances
            
        Returns:
            Endpoint name
        """
        try:
            role = get_execution_role()
        except:
            import boto3
            try:
                iam = boto3.client('iam')
                role = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']
            except:
                role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
        
        import time
        import json
        from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri
        
        self.endpoint_name = f"sama-grpo-endpoint-{int(time.time())}"
        
        # Hub Model configuration for LLM container
        hub_config = {
            'HF_MODEL_ID': self.base_model_name,
            'SM_NUM_GPUS': json.dumps(1)
        }
        
        # Create HuggingFace Model with LLM image
        self.model = HuggingFaceModel(
            model_data=self.model_data_s3_path,  # Use trained model artifacts
            image_uri=get_huggingface_llm_image_uri("huggingface", version="3.2.3"),
            env=hub_config,
            role=role
        )
        
        # Deploy with longer startup timeout for LLM models
        predictor = self.model.deploy(
            initial_instance_count=instance_count,
            instance_type=instance_type,
            endpoint_name=self.endpoint_name,
            container_startup_health_check_timeout=300
        )
        
        logger.info(f"Deployed endpoint: {self.endpoint_name}")
        return self.endpoint_name
    
    def get_endpoint_name(self) -> str:
        """Get deployed endpoint name"""
        return self.endpoint_name


def deploy_grpo_model(model_s3_path: str, instance_type: str = "ml.m5.xlarge") -> str:
    """
    Quick deployment of GRPO model using S3 artifacts directly
    
    Args:
        model_s3_path: S3 path to trained model artifacts
        instance_type: Instance type for deployment
        
    Returns:
        Endpoint name
    """
    deployer = ModelDeployer(model_s3_path)
    endpoint_name = deployer.deploy_endpoint(instance_type=instance_type)
    
    logger.info(f"GRPO model deployed to endpoint: {endpoint_name}")
    return endpoint_name
