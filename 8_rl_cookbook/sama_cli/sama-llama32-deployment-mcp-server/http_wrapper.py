#!/usr/bin/env python3
"""
HTTP Wrapper for SAMA MCP Server
Provides REST API endpoints for MCP tools
"""

import asyncio
import json
import logging
from typing import Any, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Import SAMA server tools
from sama_deployment_mcp_server.server import (
    deploy_jumpstart_model,
    create_inference_payload,
    create_instruction_payload,
    list_sagemaker_endpoints,
    describe_sagemaker_endpoint,
    delete_sagemaker_endpoint,
    get_deployment_recommendations,
    validate_aws_credentials
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="SAMA Deployment MCP Server",
    description="HTTP wrapper for SAMA MCP deployment tools",
    version="0.1.0"
)

# Request models
class DeployModelRequest(BaseModel):
    model_id: str
    instance_type: str = "ml.g5.xlarge"
    model_version: str = "1.*"
    initial_instance_count: int = 1
    endpoint_name: str = None
    accept_eula: bool = True

class InferencePayloadRequest(BaseModel):
    input_text: str
    max_new_tokens: int = 64
    temperature: float = 0.6
    top_p: float = 0.9
    return_full_text: bool = False
    do_sample: bool = None
    seed: int = None
    repetition_penalty: float = None
    top_k: int = None

class InstructionPayloadRequest(BaseModel):
    instruction: str
    context: str = ""
    max_new_tokens: int = 100

class RecommendationsRequest(BaseModel):
    model_id: str
    use_case: str = "general"

@app.on_event("startup")
async def startup_event():
    """Validate AWS credentials on startup."""
    if not validate_aws_credentials():
        logger.error("AWS credentials validation failed")
        raise Exception("AWS credentials not configured properly")
    logger.info("SAMA HTTP MCP Server started successfully")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "sama-deployment-mcp-server"}

@app.post("/deploy")
async def deploy_model(request: DeployModelRequest):
    """Deploy a JumpStart model."""
    try:
        result = await deploy_jumpstart_model(
            model_id=request.model_id,
            instance_type=request.instance_type,
            model_version=request.model_version,
            initial_instance_count=request.initial_instance_count,
            endpoint_name=request.endpoint_name,
            accept_eula=request.accept_eula
        )
        return {"result": json.loads(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/inference-payload")
async def create_inference_payload_endpoint(request: InferencePayloadRequest):
    """Create inference payload."""
    try:
        result = await create_inference_payload(
            input_text=request.input_text,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            return_full_text=request.return_full_text,
            do_sample=request.do_sample,
            seed=request.seed,
            repetition_penalty=request.repetition_penalty,
            top_k=request.top_k
        )
        return {"payload": json.loads(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/instruction-payload")
async def create_instruction_payload_endpoint(request: InstructionPayloadRequest):
    """Create instruction payload."""
    try:
        result = await create_instruction_payload(
            instruction=request.instruction,
            context=request.context,
            max_new_tokens=request.max_new_tokens
        )
        return {"payload": json.loads(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/endpoints")
async def list_endpoints(status_filter: str = None, max_results: int = 10):
    """List SageMaker endpoints."""
    try:
        result = await list_sagemaker_endpoints(
            status_filter=status_filter,
            max_results=max_results
        )
        return {"endpoints": json.loads(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/endpoints/{endpoint_name}")
async def describe_endpoint(endpoint_name: str):
    """Describe a specific endpoint."""
    try:
        result = await describe_sagemaker_endpoint(endpoint_name=endpoint_name)
        return {"endpoint": json.loads(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/endpoints/{endpoint_name}")
async def delete_endpoint(endpoint_name: str, delete_endpoint_config: bool = True):
    """Delete an endpoint."""
    try:
        result = await delete_sagemaker_endpoint(
            endpoint_name=endpoint_name,
            delete_endpoint_config=delete_endpoint_config
        )
        return {"result": json.loads(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommendations")
async def get_recommendations(request: RecommendationsRequest):
    """Get deployment recommendations."""
    try:
        result = await get_deployment_recommendations(
            model_id=request.model_id,
            use_case=request.use_case
        )
        return {"recommendations": json.loads(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to listen on")
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port)
