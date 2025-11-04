#!/usr/bin/env python3
"""
SAMA Fine-tuning MCP Server for Q CLI
Provides tools for fine-tuning models using SageMaker JumpStart.
Supports dataset preparation, template creation, and model training.
"""

import logging
import json
import os
import time
from typing import Any, Dict, List, Optional
from datetime import datetime

try:
    from fastmcp import FastMCP
    import boto3
    from botocore.exceptions import ClientError
    from sagemaker.jumpstart.estimator import JumpStartEstimator
    from sagemaker.s3 import S3Uploader
    from sagemaker.session import Session
    import sagemaker
    from datasets import load_dataset
except ImportError as e:
    logging.error(f"Required dependencies not installed: {e}")
    raise ImportError(
        "Missing dependencies. Install with: "
        "pip install sama-finetuning-mcp-server"
    )

# Initialize MCP server
mcp = FastMCP("SAMA-Finetuning-Server")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@mcp.tool()
def start_finetuning_job(
    model_id: str,
    training_data_s3_uri: str,
    role_arn: Optional[str] = None,
    model_version: Optional[str] = None,
    job_name: Optional[str] = None,
    accept_eula: bool = True,
    instruction_tuned: bool = True,
    epochs: int = 5,
    max_input_length: int = 1024,
    learning_rate: Optional[float] = None,
    batch_size: Optional[int] = None,
    instance_type: str = "ml.g5.xlarge",
    instance_count: int = 1,
    disable_output_compression: bool = True,
    hyperparameters: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Start a SageMaker fine-tuning job using JumpStart.
    
    This tool creates and starts a fine-tuning job for a JumpStart model
    using the prepared training data.
    
    Args:
        model_id: JumpStart model ID (e.g., "meta-textgeneration-llama-3-2-3b")
        training_data_s3_uri: S3 URI containing training data
        role_arn: IAM role ARN with SageMaker permissions (auto-detected if not provided)
        model_version: Model version (uses latest if not specified)
        job_name: Custom job name (auto-generated if not provided)
        accept_eula: Accept model EULA (default: True)
        instruction_tuned: Use instruction tuning (default: True)
        epochs: Number of training epochs (default: 5)
        max_input_length: Maximum input sequence length (default: 1024)
        learning_rate: Learning rate for training (optional)
        batch_size: Training batch size (optional)
        instance_type: Training instance type (default: "ml.g5.xlarge")
        instance_count: Number of training instances (default: 1)
        disable_output_compression: Disable output compression (default: True)
        hyperparameters: Additional hyperparameters (optional)
        
    Returns:
        Dictionary with training job information and status
    """
    try:
        logger.info(f"Starting fine-tuning job for model: {model_id}")
        
        # Auto-detect role if not provided
        if not role_arn:
            try:
                session = sagemaker.Session()
                role_arn = sagemaker.get_execution_role()
                logger.info(f"Auto-detected SageMaker execution role: {role_arn}")
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Could not auto-detect SageMaker role. Please provide role_arn. Error: {str(e)}"
                }
        
        if not training_data_s3_uri.startswith("s3://"):
            return {
                "status": "error",
                "message": "training_data_s3_uri must be a valid S3 URI"
            }
        
        # Generate job name if not provided - SAMA naming convention
        if not job_name:
            import time
            timestamp = int(time.time())
            # Extract model name from model_id (e.g., "meta-textgeneration-llama-3-2-3b" -> "llama32")
            if "llama-3-2" in model_id.lower():
                model_short = "llama32"
            elif "llama" in model_id.lower():
                model_short = "llama"
            else:
                model_short = model_id.split("-")[-1] if "-" in model_id else model_id[:8]
            
            job_name = f"sama-finetune-{model_short}-{timestamp}"
        
        # Create JumpStart estimator
        estimator_kwargs = {
            "model_id": model_id,
            "role": role_arn,
            "instance_type": instance_type,
            "instance_count": instance_count,
            "disable_output_compression": disable_output_compression
        }
        
        if model_version:
            estimator_kwargs["model_version"] = model_version
        
        # Set environment variables for EULA
        environment = {}
        if accept_eula:
            environment["accept_eula"] = "true"
        else:
            environment["accept_eula"] = "false"
        
        estimator_kwargs["environment"] = environment
        
        logger.info("Creating JumpStart estimator")
        estimator = JumpStartEstimator(**estimator_kwargs)
        
        # Set hyperparameters
        hyperparams = {
            "instruction_tuned": str(instruction_tuned),
            "epoch": str(epochs),
            "max_input_length": str(max_input_length)
        }
        
        if learning_rate is not None:
            hyperparams["learning_rate"] = str(learning_rate)
        
        if batch_size is not None:
            hyperparams["per_device_train_batch_size"] = str(batch_size)
        
        # Add custom hyperparameters
        if hyperparameters:
            hyperparams.update(hyperparameters)
        
        logger.info(f"Setting hyperparameters: {hyperparams}")
        estimator.set_hyperparameters(**hyperparams)
        
        # Start training
        training_inputs = {"training": training_data_s3_uri}
        logger.info(f"Starting training job: {job_name}")
        logger.info(f"Training data location: {training_data_s3_uri}")
        
        estimator.fit(training_inputs, job_name=job_name, wait=False)
        
        return {
            "status": "success",
            "message": f"Fine-tuning job '{job_name}' started successfully",
            "job_info": {
                "job_name": job_name,
                "model_id": model_id,
                "model_version": model_version or "latest",
                "training_data_s3_uri": training_data_s3_uri,
                "instance_type": instance_type,
                "instance_count": instance_count,
                "start_time": datetime.now().isoformat()
            },
            "hyperparameters": hyperparams,
            "estimator_info": {
                "role": role_arn
            }
        }
        
    except Exception as e:
        error_msg = f"Failed to start fine-tuning job: {str(e)}"
        logger.error(error_msg)
        return {
            "status": "error",
            "message": error_msg
        }


@mcp.tool()
def monitor_training_job(
    job_name: str,
    region: Optional[str] = None
) -> Dict[str, Any]:
    """
    Monitor the status of a SageMaker training job.
    
    This tool checks the current status and progress of a running
    or completed training job.
    
    Args:
        job_name: Name of the SageMaker training job to monitor
        region: AWS region where the job is running (auto-detected if not provided)
        
    Returns:
        Dictionary with training job status and details
    """
    try:
        logger.info(f"Monitoring training job: {job_name}")
        
        # Auto-detect region if not provided
        if not region:
            session = boto3.Session()
            region = session.region_name or "us-east-1"
        
        # Create SageMaker client
        sagemaker_client = boto3.client('sagemaker', region_name=region)
        
        # Get training job details
        response = sagemaker_client.describe_training_job(TrainingJobName=job_name)
        
        training_job_status = response['TrainingJobStatus']
        creation_time = response['CreationTime']
        
        result = {
            "status": "success",
            "job_name": job_name,
            "training_job_status": training_job_status,
            "creation_time": creation_time.isoformat(),
            "region": region
        }
        
        # Add timing information
        if 'TrainingStartTime' in response:
            result["training_start_time"] = response['TrainingStartTime'].isoformat()
        
        if 'TrainingEndTime' in response:
            result["training_end_time"] = response['TrainingEndTime'].isoformat()
            
        # Add failure reason if job failed
        if training_job_status == 'Failed' and 'FailureReason' in response:
            result["failure_reason"] = response['FailureReason']
        
        # Add model artifacts location if completed
        if training_job_status == 'Completed' and 'ModelArtifacts' in response:
            result["model_artifacts_s3_uri"] = response['ModelArtifacts']['S3ModelArtifacts']
        
        # Add hyperparameters
        if 'HyperParameters' in response:
            result["hyperparameters"] = response['HyperParameters']
        
        # Add resource configuration
        if 'ResourceConfig' in response:
            result["resource_config"] = response['ResourceConfig']
        
        return result
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'ValidationException':
            return {
                "status": "error",
                "message": f"Training job '{job_name}' not found in region '{region}'"
            }
        else:
            return {
                "status": "error", 
                "message": f"AWS error while monitoring training job: {e}"
            }
    except Exception as e:
        error_msg = f"Failed to monitor training job: {str(e)}"
        logger.error(error_msg)
        return {
            "status": "error",
            "message": error_msg
        }


@mcp.tool()
def create_model_from_training_job(
    training_job_name: str,
    model_id: str,
    role_arn: Optional[str] = None,
    model_name: Optional[str] = None,
    region: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a SageMaker model from a completed training job.
    
    This tool creates a SageMaker model object from the artifacts
    produced by a fine-tuning job, which can then be deployed.
    
    Args:
        training_job_name: Name of the completed training job
        model_id: Original JumpStart model ID used for training
        role_arn: IAM role ARN with SageMaker permissions (auto-detected if not provided)
        model_name: Custom model name (auto-generated if not provided)
        region: AWS region (auto-detected if not provided)
        
    Returns:
        Dictionary with model creation results
    """
    try:
        logger.info(f"Creating model from training job: {training_job_name}")
        
        # Auto-detect region if not provided
        if not region:
            session = boto3.Session()
            region = session.region_name or "us-east-1"
        
        # Auto-detect role if not provided
        if not role_arn:
            try:
                role_arn = sagemaker.get_execution_role()
                logger.info(f"Auto-detected SageMaker execution role: {role_arn}")
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Could not auto-detect SageMaker role. Please provide role_arn. Error: {str(e)}"
                }
        
        # Create SageMaker session
        boto_session = boto3.Session(region_name=region)
        sagemaker_session = Session(
            boto_session=boto_session,
            sagemaker_client=boto_session.client('sagemaker')
        )
        
        # Generate model name if not provided
        if not model_name:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            model_name = f"{training_job_name}-model-{timestamp}"
        
        # Create JumpStart estimator to attach to existing training job
        estimator = JumpStartEstimator.attach(
            training_job_name=training_job_name,
            sagemaker_session=sagemaker_session
        )
        
        # Create model
        logger.info(f"Creating SageMaker model: {model_name}")
        model = estimator.create_model(
            name=model_name,
            role=role_arn
        )
        
        return {
            "status": "success",
            "message": f"Model '{model_name}' created successfully from training job '{training_job_name}'",
            "model_info": {
                "model_name": model_name,
                "training_job_name": training_job_name,
                "model_id": model_id,
                "role_arn": role_arn,
                "region": region,
                "creation_time": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        error_msg = f"Failed to create model from training job: {str(e)}"
        logger.error(error_msg)
        return {
            "status": "error",
            "message": error_msg
        }


@mcp.tool()
def get_finetuning_recommendations(
    model_id: str,
    dataset_size: Optional[int] = None,
    use_case: str = "general"
) -> Dict[str, Any]:
    """
    Get fine-tuning recommendations for a specific model and use case.
    
    This tool provides recommendations for hyperparameters, instance types,
    and training configurations based on the model and dataset characteristics.
    
    Args:
        model_id: JumpStart model ID (e.g., "meta-textgeneration-llama-3-2-3b")
        dataset_size: Number of training examples (optional)
        use_case: Use case type - "general", "instruction-following", "summarization", "qa"
        
    Returns:
        Dictionary with fine-tuning recommendations
    """
    try:
        logger.info(f"Getting fine-tuning recommendations for model: {model_id}")
        
        # Base recommendations
        recommendations = {
            "status": "success",
            "model_id": model_id,
            "use_case": use_case,
            "recommendations": {}
        }
        
        # Instance type recommendations based on model size
        if "3b" in model_id.lower():
            recommendations["recommendations"]["instance_type"] = "ml.g5.xlarge"
            recommendations["recommendations"]["instance_count"] = 1
        elif "7b" in model_id.lower():
            recommendations["recommendations"]["instance_type"] = "ml.g5.2xlarge"
            recommendations["recommendations"]["instance_count"] = 1
        elif "13b" in model_id.lower() or "11b" in model_id.lower():
            recommendations["recommendations"]["instance_type"] = "ml.g5.4xlarge"
            recommendations["recommendations"]["instance_count"] = 1
        else:
            recommendations["recommendations"]["instance_type"] = "ml.g5.xlarge"
            recommendations["recommendations"]["instance_count"] = 1
        
        # Hyperparameter recommendations based on use case
        if use_case == "instruction-following":
            recommendations["recommendations"]["epochs"] = 3
            recommendations["recommendations"]["learning_rate"] = 2e-5
            recommendations["recommendations"]["batch_size"] = 4
            recommendations["recommendations"]["max_input_length"] = 1024
        elif use_case == "summarization":
            recommendations["recommendations"]["epochs"] = 5
            recommendations["recommendations"]["learning_rate"] = 1e-5
            recommendations["recommendations"]["batch_size"] = 2
            recommendations["recommendations"]["max_input_length"] = 2048
        elif use_case == "qa":
            recommendations["recommendations"]["epochs"] = 4
            recommendations["recommendations"]["learning_rate"] = 1.5e-5
            recommendations["recommendations"]["batch_size"] = 4
            recommendations["recommendations"]["max_input_length"] = 1024
        else:  # general
            recommendations["recommendations"]["epochs"] = 5
            recommendations["recommendations"]["learning_rate"] = 2e-5
            recommendations["recommendations"]["batch_size"] = 4
            recommendations["recommendations"]["max_input_length"] = 1024
        
        # Adjust based on dataset size
        if dataset_size:
            if dataset_size < 1000:
                recommendations["recommendations"]["epochs"] = min(recommendations["recommendations"]["epochs"] + 2, 10)
                recommendations["recommendations"]["learning_rate"] *= 1.5
            elif dataset_size > 10000:
                recommendations["recommendations"]["epochs"] = max(recommendations["recommendations"]["epochs"] - 1, 2)
                recommendations["recommendations"]["learning_rate"] *= 0.8
        
        # Add cost and time estimates
        instance_type = recommendations["recommendations"]["instance_type"]
        epochs = recommendations["recommendations"]["epochs"]
        
        # Rough estimates (these would be more accurate with real pricing data)
        if "xlarge" in instance_type and "2x" not in instance_type:
            cost_per_hour = 1.0
        elif "2xlarge" in instance_type:
            cost_per_hour = 2.0
        elif "4xlarge" in instance_type:
            cost_per_hour = 4.0
        else:
            cost_per_hour = 1.0
        
        estimated_hours = epochs * 0.5  # Rough estimate
        estimated_cost = cost_per_hour * estimated_hours
        
        recommendations["recommendations"]["estimated_training_time_hours"] = estimated_hours
        recommendations["recommendations"]["estimated_cost_usd"] = round(estimated_cost, 2)
        
        recommendations["recommendations"]["additional_tips"] = [
            "Monitor training job progress regularly",
            "Consider using spot instances for cost savings",
            "Validate model performance on a held-out test set",
            "Save checkpoints during training for recovery"
        ]
        
        return recommendations
        
    except Exception as e:
        error_msg = f"Failed to get fine-tuning recommendations: {str(e)}"
        logger.error(error_msg)
        return {
            "status": "error",
            "message": error_msg
        }


def main():
    """Main entry point for the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
