#!/usr/bin/env python3
"""
SAMA Deployment MCP Server - Q CLI Compatible
Provides tools for SageMaker model deployment and serving for SAMA agents
Based on existing FastAPI deployment server implementation
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastMCP server for SAMA
mcp = FastMCP('sama-deployment-mcp-server')


def get_sagemaker_client():
    """Get SageMaker client with error handling."""
    try:
        return boto3.client('sagemaker')
    except NoCredentialsError:
        raise Exception("AWS credentials not configured")
    except Exception as e:
        raise Exception(f"Failed to create SageMaker client: {str(e)}")


@mcp.tool(description="Deploy a SageMaker JumpStart model to an endpoint")
async def deploy_jumpstart_model(
    model_id: str = Field(description="JumpStart model ID (e.g., meta-textgeneration-llama-3-2-3b)"),
    instance_type: str = Field(default="ml.g5.xlarge", description="Instance type for deployment"),
    model_version: str = Field(default="1.*", description="Model version"),
    initial_instance_count: int = Field(default=1, description="Number of instances"),
    endpoint_name: Optional[str] = Field(default=None, description="Custom endpoint name (optional)"),
    accept_eula: bool = Field(default=True, description="Accept model EULA for predictions"),
) -> str:
    """
    Deploy a JumpStart model to SageMaker endpoint and wait for completion.
    """
    try:
        logger.info(f"Deploying JumpStart model: {model_id} on {instance_type}")
        
        # Import SageMaker here to handle potential import issues
        try:
            from sagemaker.jumpstart.model import JumpStartModel
        except ImportError:
            error_result = {
                "success": False,
                "error": "SageMaker SDK not available. Please install sagemaker package."
            }
            return json.dumps(error_result, indent=2)
        
        # Generate endpoint name if not provided - SAMA naming convention
        if not endpoint_name:
            import time
            timestamp = int(time.time())
            # Extract model name from model_id (e.g., "meta-textgeneration-llama-3-2-3b" -> "llama32")
            if "llama-3-2" in model_id.lower():
                model_short = "llama32"
            elif "llama" in model_id.lower():
                model_short = "llama"
            else:
                model_short = model_id.split("-")[-1] if "-" in model_id else model_id[:8]
            
            endpoint_name = f"sama-endpoint-{model_short}-{timestamp}"
        
        # Create the model
        model = JumpStartModel(
            model_id=model_id,
            model_version=model_version
        )
        
        # Start deployment
        logger.info(f"Starting deployment to endpoint: {endpoint_name}")
        
        try:
            # Start the deployment (this initiates but doesn't wait)
            predictor = model.deploy(
                initial_instance_count=initial_instance_count,
                instance_type=instance_type,
                endpoint_name=endpoint_name,
                accept_eula=accept_eula,
                wait=False  # Don't wait in the deploy call, we'll handle it ourselves
            )
            
            logger.info(f"Deployment initiated for endpoint: {endpoint_name}")
            logger.info("Waiting for endpoint to be ready (this may take 10-20 minutes)...")
            
            # Now wait for the endpoint to be ready with proper polling
            sagemaker_client = get_sagemaker_client()
            
            # Poll endpoint status until ready or failed
            max_wait_time = 30 * 60  # 30 minutes maximum
            poll_interval = 30  # Check every 30 seconds
            start_time = time.time()
            
            while True:
                try:
                    response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
                    status = response['EndpointStatus']
                    
                    logger.info(f"Endpoint {endpoint_name} status: {status}")
                    
                    if status == 'InService':
                        # Success! Endpoint is ready
                        logger.info(f"Endpoint {endpoint_name} is now InService")
                        break
                    elif status == 'Failed':
                        # Deployment failed
                        failure_reason = response.get('FailureReason', 'Unknown failure reason')
                        error_result = {
                            "success": False,
                            "error": f"Endpoint deployment failed: {failure_reason}",
                            "endpoint_name": endpoint_name,
                            "status": status,
                            "troubleshooting": [
                                "Check if the instance type is available in your region",
                                "Verify you have sufficient service limits",
                                "Ensure the model ID is correct and supported",
                                "Check CloudWatch logs for detailed error information"
                            ]
                        }
                        return json.dumps(error_result, indent=2)
                    elif status in ['Creating', 'Updating']:
                        # Still in progress, continue waiting
                        elapsed_time = time.time() - start_time
                        if elapsed_time > max_wait_time:
                            error_result = {
                                "success": False,
                                "error": f"Deployment timeout after {max_wait_time/60:.1f} minutes",
                                "endpoint_name": endpoint_name,
                                "current_status": status,
                                "note": "Deployment may still be in progress. Check AWS console for updates."
                            }
                            return json.dumps(error_result, indent=2)
                        
                        # Wait before next poll
                        await asyncio.sleep(poll_interval)
                    else:
                        # Unexpected status
                        logger.warning(f"Unexpected endpoint status: {status}")
                        await asyncio.sleep(poll_interval)
                        
                except ClientError as e:
                    if e.response['Error']['Code'] == 'ValidationException':
                        # Endpoint might not exist yet, wait a bit more
                        await asyncio.sleep(poll_interval)
                    else:
                        raise
            
            # If we get here, deployment was successful
            result = {
                "success": True,
                "message": f"Model {model_id} successfully deployed and ready for inference",
                "endpoint_name": endpoint_name,
                "instance_type": instance_type,
                "model_id": model_id,
                "model_version": model_version,
                "deployment_time_minutes": round((time.time() - start_time) / 60, 1),
                "predictor_info": {
                    "endpoint_name": predictor.endpoint_name,
                    "content_type": predictor.content_type,
                    "accept_type": predictor.accept
                },
                "usage_note": "Endpoint is ready for inference calls",
                "timestamp": datetime.now().isoformat()
            }
            
            return json.dumps(result, indent=2, default=str)
            
        except Exception as deploy_error:
            logger.error(f"Deployment failed: {deploy_error}")
            error_result = {
                "success": False,
                "error": f"Deployment failed: {str(deploy_error)}",
                "endpoint_name": endpoint_name,
                "troubleshooting": [
                    "Check if the instance type is available in your region",
                    "Verify you have sufficient service limits",
                    "Ensure the model ID is correct and supported"
                ]
            }
            return json.dumps(error_result, indent=2)
        
    except Exception as e:
        logger.error(f"Error in deploy_jumpstart_model: {e}")
        error_result = {
            "success": False,
            "error": f"Failed to deploy model: {str(e)}"
        }
        return json.dumps(error_result, indent=2)


@mcp.tool(description="Create an inference payload for model prediction")
async def create_inference_payload(
    input_text: str = Field(description="Input text for generation"),
    max_new_tokens: int = Field(default=64, description="Maximum number of tokens to generate"),
    top_p: float = Field(default=0.9, description="Top-p sampling parameter"),
    temperature: float = Field(default=0.6, description="Temperature for sampling"),
    return_full_text: bool = Field(default=False, description="Whether to return full text including input"),
    seed: Optional[int] = Field(default=None, description="Random seed for generation"),
    repetition_penalty: Optional[float] = Field(default=None, description="Penalty for repetitive text"),
    do_sample: Optional[bool] = Field(default=None, description="Enable sampling"),
    top_k: Optional[int] = Field(default=None, description="Top-k sampling parameter"),
) -> str:
    """
    Create inference payload for model prediction.
    """
    try:
        logger.info("Creating inference payload")
        
        # Create payload structure
        payload = {
            "inputs": input_text,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "top_p": top_p,
                "temperature": temperature,
                "return_full_text": return_full_text
            }
        }
        
        # Add optional parameters if specified
        optional_params = {
            "seed": seed,
            "repetition_penalty": repetition_penalty,
            "do_sample": do_sample,
            "top_k": top_k
        }
        
        for param, value in optional_params.items():
            if value is not None:
                payload["parameters"][param] = value
        
        result = {
            "success": True,
            "message": "Inference payload created successfully",
            "payload": payload,
            "usage_examples": {
                "pretrained": "response = predictor.predict(payload, custom_attributes='accept_eula=true')",
                "finetuned": "response = predictor.predict(payload)"
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return json.dumps(result, indent=2, default=str)
        
    except Exception as e:
        logger.error(f"Error creating inference payload: {e}")
        error_result = {
            "success": False,
            "error": f"Failed to create payload: {str(e)}"
        }
        return json.dumps(error_result, indent=2)


@mcp.tool(description="Create an instruction tuning payload for the model")
async def create_instruction_payload(
    instruction: str = Field(description="Instruction for the model"),
    context: str = Field(default="", description="Additional context/input"),
    max_new_tokens: int = Field(default=100, description="Maximum tokens to generate"),
) -> Dict[str, Any]:
    """
    Create instruction tuning payload for the model.
    """
    try:
        logger.info("Creating instruction tuning payload")
        
        # Use template format from original code
        template = {
            "prompt": "Below is an instruction that describes a task, paired with an input that provides further context. "
                     "Write a response that appropriately completes the request.\n\n"
                     "### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n",
            "completion": " {response}"
        }
        
        # Format input using template
        input_output_demarkation_key = "\n\n### Response:\n"
        formatted_input = template["prompt"].format(
            instruction=instruction, 
            context=context
        ) + input_output_demarkation_key
        
        payload = {
            "inputs": formatted_input,
            "parameters": {
                "max_new_tokens": max_new_tokens
            }
        }
        
        return {
            "status": "success",
            "message": "Instruction tuning payload created successfully",
            "payload": payload,
            "template_used": template,
            "formatted_components": {
                "instruction": instruction,
                "context": context,
                "formatted_input": formatted_input
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error creating instruction payload: {e}")
        return {
            "status": "error",
            "message": f"Failed to create instruction payload: {str(e)}"
        }


@mcp.tool(description="List SageMaker endpoints")
async def list_sagemaker_endpoints(
    status_filter: Optional[str] = Field(default=None, description="Filter by status (InService, Creating, Updating, etc.)"),
    max_results: int = Field(default=10, description="Maximum number of endpoints to return"),
) -> str:
    """
    List SageMaker endpoints with their status and information.
    """
    try:
        sagemaker_client = get_sagemaker_client()
        
        # List endpoints
        list_params = {"MaxResults": max_results}
        if status_filter is not None and isinstance(status_filter, str) and status_filter.strip():
            list_params["StatusEquals"] = status_filter.strip()
            
        response = sagemaker_client.list_endpoints(**list_params)
        
        endpoints = []
        for endpoint in response['Endpoints']:
            endpoints.append({
                "endpoint_name": endpoint['EndpointName'],
                "endpoint_arn": endpoint['EndpointArn'],
                "endpoint_status": endpoint['EndpointStatus'],
                "creation_time": endpoint['CreationTime'].isoformat(),
                "last_modified_time": endpoint['LastModifiedTime'].isoformat(),
            })
        
        result = {
            "success": True,
            "message": f"Found {len(endpoints)} endpoints",
            "endpoints": endpoints,
            "timestamp": datetime.now().isoformat()
        }
        
        return json.dumps(result, indent=2, default=str)
        
    except Exception as e:
        logger.error(f"Error listing endpoints: {e}")
        error_result = {
            "success": False,
            "error": f"Failed to list endpoints: {str(e)}"
        }
        return json.dumps(error_result, indent=2)


@mcp.tool(description="Get detailed information about a SageMaker endpoint")
async def describe_sagemaker_endpoint(
    endpoint_name: str = Field(description="Name of the endpoint to describe"),
) -> Dict[str, Any]:
    """
    Get detailed information about a specific SageMaker endpoint.
    """
    try:
        sagemaker_client = get_sagemaker_client()
        
        response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        
        return {
            "endpoint_name": response['EndpointName'],
            "endpoint_arn": response['EndpointArn'],
            "endpoint_config_name": response['EndpointConfigName'],
            "endpoint_status": response['EndpointStatus'],
            "failure_reason": response.get('FailureReason'),
            "creation_time": response['CreationTime'].isoformat(),
            "last_modified_time": response['LastModifiedTime'].isoformat(),
            "production_variants": response.get('ProductionVariants', []),
            "timestamp": datetime.now().isoformat()
        }
        
    except ClientError as e:
        if e.response['Error']['Code'] == 'ValidationException':
            return {"error": f"Endpoint '{endpoint_name}' not found"}
        return {"error": f"AWS error: {str(e)}"}
    except Exception as e:
        logger.error(f"Error describing endpoint: {e}")
        return {"error": f"Failed to describe endpoint: {str(e)}"}


@mcp.tool(description="Delete a SageMaker endpoint")
async def delete_sagemaker_endpoint(
    endpoint_name: str = Field(description="Name of the endpoint to delete"),
    delete_endpoint_config: bool = Field(default=True, description="Also delete the endpoint configuration"),
) -> Dict[str, Any]:
    """
    Delete a SageMaker endpoint and optionally its configuration.
    """
    try:
        sagemaker_client = get_sagemaker_client()
        
        # Get endpoint details first
        try:
            endpoint_info = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            endpoint_config_name = endpoint_info['EndpointConfigName']
        except ClientError as e:
            if e.response['Error']['Code'] == 'ValidationException':
                return {"error": f"Endpoint '{endpoint_name}' not found"}
            raise
        
        # Delete the endpoint
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        
        result = {
            "status": "success",
            "message": f"Endpoint '{endpoint_name}' deletion initiated",
            "endpoint_name": endpoint_name,
            "timestamp": datetime.now().isoformat()
        }
        
        # Delete endpoint configuration if requested
        if delete_endpoint_config:
            try:
                sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
                result["endpoint_config_deleted"] = endpoint_config_name
                result["message"] += f" and endpoint config '{endpoint_config_name}' deleted"
            except Exception as config_error:
                result["endpoint_config_warning"] = f"Failed to delete endpoint config: {str(config_error)}"
        
        return result
        
    except Exception as e:
        logger.error(f"Error deleting endpoint: {e}")
        return {
            "status": "error",
            "message": f"Failed to delete endpoint: {str(e)}"
        }


@mcp.tool(description="Get deployment recommendations for a model")
async def get_deployment_recommendations(
    model_id: str = Field(description="JumpStart model ID"),
    use_case: str = Field(default="general", description="Use case: general, high-throughput, low-latency"),
) -> Dict[str, Any]:
    """
    Get deployment recommendations for a specific model and use case.
    """
    try:
        # Define instance type recommendations based on model and use case
        recommendations = {
            "meta-textgeneration-llama-3-2-3b": {
                "general": {
                    "instance_type": "ml.g5.xlarge",
                    "min_instances": 1,
                    "max_instances": 2,
                    "reasoning": "Good balance of cost and performance for general text generation"
                },
                "high-throughput": {
                    "instance_type": "ml.g5.2xlarge",
                    "min_instances": 2,
                    "max_instances": 5,
                    "reasoning": "Higher compute capacity for handling multiple concurrent requests"
                },
                "low-latency": {
                    "instance_type": "ml.g5.xlarge",
                    "min_instances": 1,
                    "max_instances": 1,
                    "reasoning": "Single instance for consistent low latency"
                }
            },
            "meta-textgeneration-llama-3-2-1b": {
                "general": {
                    "instance_type": "ml.g5.large",
                    "min_instances": 1,
                    "max_instances": 2,
                    "reasoning": "Smaller model can run efficiently on smaller instances"
                },
                "high-throughput": {
                    "instance_type": "ml.g5.xlarge",
                    "min_instances": 2,
                    "max_instances": 4,
                    "reasoning": "Scale up for high throughput needs"
                },
                "low-latency": {
                    "instance_type": "ml.g5.large",
                    "min_instances": 1,
                    "max_instances": 1,
                    "reasoning": "Single smaller instance for low latency"
                }
            }
        }
        
        # Get recommendation or provide default
        if model_id in recommendations and use_case in recommendations[model_id]:
            recommendation = recommendations[model_id][use_case]
        else:
            # Default recommendation
            recommendation = {
                "instance_type": "ml.g5.xlarge",
                "min_instances": 1,
                "max_instances": 2,
                "reasoning": "Default recommendation for unknown model/use case combination"
            }
        
        return {
            "status": "success",
            "model_id": model_id,
            "use_case": use_case,
            "recommendation": recommendation,
            "additional_notes": [
                "Consider your budget and expected traffic patterns",
                "Start with minimum instances and scale up as needed",
                "Monitor CloudWatch metrics for optimization opportunities"
            ],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting deployment recommendations: {e}")
        return {
            "status": "error",
            "message": f"Failed to get recommendations: {str(e)}"
        }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='SAMA Deployment MCP Server - SageMaker Model Deployment Tools for SAMA Agents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='sama-deployment-mcp-server 0.1.0'
    )
    
    parser.add_argument(
        '--allow-write',
        action='store_true',
        default=True,
        help='Allow write operations (model deployment, endpoint deletion)'
    )
    
    return parser.parse_args()


def validate_aws_credentials() -> bool:
    """Validate AWS credentials are available."""
    try:
        import boto3
        sts = boto3.client('sts')
        sts.get_caller_identity()
        return True
    except Exception as e:
        logger.error(f"AWS credentials validation failed: {e}")
        return False


def main():
    """Main entry point for the MCP server."""
    args = parse_args()
    
    # Set environment variables
    import os
    os.environ['ALLOW_WRITE'] = str(args.allow_write).lower()
    
    # Set default AWS region if not set
    if not os.getenv('AWS_REGION'):
        os.environ['AWS_REGION'] = 'us-east-2'
    
    # Validate AWS credentials
    if not validate_aws_credentials():
        logger.error("Failed to validate AWS credentials. Please check your AWS configuration.")
        import sys
        sys.exit(1)
    
    logger.info("Starting SAMA Deployment MCP Server...")
    logger.info(f"AWS Region: {os.getenv('AWS_REGION')}")
    logger.info(f"Allow Write: {args.allow_write}")
    logger.info("SAMA Agent Context: Model deployment and management")
    
    # Run the FastMCP server
    mcp.run()


if __name__ == '__main__':
    main()
