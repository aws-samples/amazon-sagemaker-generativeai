#!/usr/bin/env python3
"""
SAMA RL Model Deployment MCP Server - Synchronous JSON-RPC Version
Deploy trained GRPO models to SageMaker endpoints
"""

import json
import sys
import os
from pathlib import Path

# Add parent directory to path for sama_rl imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "sama_rl"))

def deploy_model(training_job_name: str, endpoint_name: str = None, instance_type: str = None):
    """Deploy a trained GRPO model to SageMaker endpoint"""
    try:
        # Import SAMA RL
        from sama_rl import GRPO
        
        # Load existing training job
        trainer = GRPO(training_job_name=training_job_name)
        
        # Deploy with specified parameters
        deployed_endpoint = trainer.deploy(
            endpoint_name=endpoint_name,
            instance_type=instance_type
        )
        
        return {
            "status": "success",
            "endpoint_name": deployed_endpoint,
            "training_job": training_job_name,
            "instance_type": instance_type or "auto-selected"
        }
        
    except ImportError:
        return {"error": "SAMA RL not installed. Please run: pip install -e . from the project root"}
    except Exception as e:
        return {"error": f"Deployment failed: {str(e)}"}

def get_deployment_recommendations(model_size: str = "1.5b"):
    """Get deployment recommendations for model size"""
    recommendations = {
        "0.5b": {"instance": "ml.g5.xlarge", "cost": "$1.00/hour"},
        "1.5b": {"instance": "ml.g5.2xlarge", "cost": "$1.50/hour"},
        "7b": {"instance": "ml.g5.4xlarge", "cost": "$2.50/hour"},
        "13b": {"instance": "ml.g5.12xlarge", "cost": "$7.50/hour"}
    }
    
    return {
        "model_size": model_size,
        "recommendation": recommendations.get(model_size, recommendations["1.5b"]),
        "all_options": recommendations
    }

def main():
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
                
            request = json.loads(line.strip())
            method = request.get("method")
            request_id = request.get("id")
            
            if method == "initialize":
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "grpo-rl-deployment", "version": "1.0.0"}
                    }
                }
            elif method == "tools/list":
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "tools": [
                            {
                                "name": "deploy_model",
                                "description": "Deploy trained GRPO model to SageMaker endpoint",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "training_job_name": {"type": "string", "description": "GRPO training job name"},
                                        "endpoint_name": {"type": "string", "description": "Endpoint name (optional)"},
                                        "instance_type": {"type": "string", "description": "Instance type (optional)"}
                                    },
                                    "required": ["training_job_name"]
                                }
                            },
                            {
                                "name": "get_recommendations",
                                "description": "Get deployment recommendations for model size",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "model_size": {"type": "string", "description": "Model size (0.5b, 1.5b, 7b, 13b)", "default": "1.5b"}
                                    }
                                }
                            }
                        ]
                    }
                }
            elif method == "tools/call":
                tool_name = request.get("params", {}).get("name")
                args = request.get("params", {}).get("arguments", {})
                
                if tool_name == "deploy_model":
                    result = deploy_model(**args)
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {"content": [{"type": "text", "text": json.dumps(result)}]}
                    }
                elif tool_name == "get_recommendations":
                    result = get_deployment_recommendations(**args)
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {"content": [{"type": "text", "text": json.dumps(result)}]}
                    }
                else:
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}
                    }
            else:
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": f"Unknown method: {method}"}
                }
            
            print(json.dumps(response))
            sys.stdout.flush()
            
        except (EOFError, KeyboardInterrupt):
            break
        except Exception as e:
            error_response = {
                "jsonrpc": "2.0",
                "id": request_id if 'request_id' in locals() else None,
                "error": {"code": -32603, "message": str(e)}
            }
            print(json.dumps(error_response))
            sys.stdout.flush()

if __name__ == "__main__":
    main()
