#!/usr/bin/env python3
"""
SAMA RL Model Deployment MCP Server
Helps users deploy trained GRPO models to SageMaker endpoints
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path for sama_rl imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "sama_rl"))

try:
    import mcp.server.stdio
    import mcp.types as types
    from mcp.server import NotificationOptions, Server
except ImportError:
    print("MCP not installed. Install with: pip install mcp")
    sys.exit(1)

# Check SAMA RL availability at startup
sama_rl = None  # Lazy load when needed

# Initialize server
server = Server("sama-rl-model-deployment")

# Instance type recommendations
INSTANCE_RECOMMENDATIONS = {
    "0.5b": {
        "cpu": "ml.m5.large",
        "gpu": "ml.g5.xlarge", 
        "cost_cpu": "$0.10/hour",
        "cost_gpu": "$1.00/hour"
    },
    "1.5b": {
        "cpu": "ml.m5.xlarge",
        "gpu": "ml.g5.2xlarge",
        "cost_cpu": "$0.20/hour", 
        "cost_gpu": "$1.50/hour"
    },
    "7b": {
        "cpu": "ml.m5.2xlarge",
        "gpu": "ml.g5.4xlarge",
        "cost_cpu": "$0.40/hour",
        "cost_gpu": "$2.50/hour"
    },
    "13b": {
        "cpu": "ml.m5.4xlarge",
        "gpu": "ml.g5.12xlarge",
        "cost_cpu": "$0.80/hour",
        "cost_gpu": "$7.00/hour"
    }
}

@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """List available tools for model deployment"""
    return [
        types.Tool(
            name="recommend_deployment_instance",
            description="Recommend deployment instance type based on model size and requirements",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_size": {
                        "type": "string",
                        "enum": ["0.5b", "1.5b", "7b", "13b", "custom"],
                        "description": "Model size (number of parameters)"
                    },
                    "expected_traffic": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "Expected inference traffic"
                    },
                    "latency_requirements": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "Latency requirements (low = fast response needed)"
                    },
                    "budget": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "Deployment budget"
                    }
                },
                "required": ["model_size"]
            }
        ),
        types.Tool(
            name="deploy_model",
            description="Deploy a trained GRPO model to SageMaker endpoint",
            inputSchema={
                "type": "object",
                "properties": {
                    "training_job_name": {
                        "type": "string",
                        "description": "Name of the completed training job"
                    },
                    "instance_type": {
                        "type": "string",
                        "description": "SageMaker instance type for deployment"
                    },
                    "instance_count": {
                        "type": "integer",
                        "default": 1,
                        "description": "Number of instances for the endpoint"
                    }
                },
                "required": ["training_job_name"]
            }
        ),
        types.Tool(
            name="generate_deployment_code",
            description="Generate code to deploy a trained model",
            inputSchema={
                "type": "object",
                "properties": {
                    "training_job_name": {
                        "type": "string",
                        "description": "Name of the training job"
                    },
                    "instance_type": {
                        "type": "string",
                        "description": "Instance type for deployment"
                    },
                    "auto_scaling": {
                        "type": "boolean",
                        "default": False,
                        "description": "Enable auto-scaling"
                    }
                },
                "required": ["training_job_name"]
            }
        ),
        types.Tool(
            name="list_deployment_options",
            description="List all available deployment options and configurations",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="estimate_deployment_cost",
            description="Estimate deployment costs for different configurations",
            inputSchema={
                "type": "object",
                "properties": {
                    "instance_type": {
                        "type": "string",
                        "description": "SageMaker instance type"
                    },
                    "instance_count": {
                        "type": "integer",
                        "default": 1,
                        "description": "Number of instances"
                    },
                    "hours_per_day": {
                        "type": "integer",
                        "default": 24,
                        "description": "Expected hours of usage per day"
                    }
                },
                "required": ["instance_type"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle tool calls"""
    
    if name == "recommend_deployment_instance":
        model_size = arguments.get("model_size")
        traffic = arguments.get("expected_traffic", "medium")
        latency = arguments.get("latency_requirements", "medium")
        budget = arguments.get("budget", "medium")
        
        if model_size not in INSTANCE_RECOMMENDATIONS:
            return [types.TextContent(
                type="text",
                text=f"Unknown model size: {model_size}. Available sizes: {list(INSTANCE_RECOMMENDATIONS.keys())}"
            )]
        
        config = INSTANCE_RECOMMENDATIONS[model_size]
        
        # Recommendation logic
        if budget == "low" or latency == "high":
            recommended = config["cpu"]
            cost = config["cost_cpu"]
            note = "CPU instance recommended for cost optimization"
        else:
            recommended = config["gpu"]
            cost = config["cost_gpu"]
            note = "GPU instance recommended for better performance"
        
        # Adjust for traffic
        instance_count = 1
        if traffic == "high":
            instance_count = 3
        elif traffic == "medium":
            instance_count = 2
        
        recommendation = f"""
**Deployment Recommendation for {model_size} model:**

**Recommended Instance**: {recommended}
**Instance Count**: {instance_count}
**Estimated Cost**: {cost} per instance
**Total Cost**: {cost.replace('/hour', '')} × {instance_count} = ${float(cost.split('$')[1].split('/')[0]) * instance_count:.2f}/hour

**Configuration Details**:
- Traffic Level: {traffic}
- Latency Requirements: {latency} 
- Budget: {budget}
- Note: {note}

**Alternative Options**:
- CPU Option: {config['cpu']} ({config['cost_cpu']})
- GPU Option: {config['gpu']} ({config['cost_gpu']})
"""
        
        return [types.TextContent(type="text", text=recommendation)]
    
    elif name == "deploy_model":
        training_job_name = arguments.get("training_job_name")
        instance_type = arguments.get("instance_type")
        instance_count = arguments.get("instance_count", 1)
        
        if not training_job_name:
            return [types.TextContent(
                type="text",
                text="Training job name is required for deployment"
            )]
        
        # Generate deployment code
        code = f"""
from sama_rl import GRPO

# Load trained model from training job
trainer = GRPO(training_job_name="{training_job_name}")

# Deploy to SageMaker endpoint
endpoint_name = trainer.deploy(
    instance_type="{instance_type or 'auto'}",
    instance_count={instance_count}
)

print(f"Model deployed to endpoint: {{endpoint_name}}")
"""
        
        result = f"""
**Deploying Model**: {training_job_name}

**Configuration**:
- Instance Type: {instance_type or 'Auto-selected'}
- Instance Count: {instance_count}

**Deployment Code**:
```python{code}```

**Next Steps**:
1. Run the code above to deploy your model
2. Wait for deployment to complete (5-10 minutes)
3. Use the endpoint name for inference
"""
        
        return [types.TextContent(type="text", text=result)]
    
    elif name == "generate_deployment_code":
        training_job_name = arguments.get("training_job_name")
        instance_type = arguments.get("instance_type", "auto")
        auto_scaling = arguments.get("auto_scaling", False)
        
        base_code = f"""
from sama_rl import GRPO, create_inference_model

# Load trained model
trainer = GRPO(training_job_name="{training_job_name}")

# Deploy model
endpoint_name = trainer.deploy(
    instance_type="{instance_type}",
    instance_count=1
)

print(f"Model deployed to: {{endpoint_name}}")

# Create inference model
model = create_inference_model(endpoint_name)

# Test inference
test_prompt = "What is artificial intelligence?"
completion = model.generate(
    prompt=test_prompt,
    max_new_tokens=200,
    temperature=0.7
)

print(f"Test completion: {{completion}}")
"""
        
        if auto_scaling:
            scaling_code = """
# Optional: Configure auto-scaling
import boto3

client = boto3.client('application-autoscaling')

# Register scalable target
client.register_scalable_target(
    ServiceNamespace='sagemaker',
    ResourceId=f'endpoint/{endpoint_name}/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    MinCapacity=1,
    MaxCapacity=5
)

# Create scaling policy
client.put_scaling_policy(
    PolicyName=f'{endpoint_name}-scaling-policy',
    ServiceNamespace='sagemaker',
    ResourceId=f'endpoint/{endpoint_name}/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    PolicyType='TargetTrackingScaling',
    TargetTrackingScalingPolicyConfiguration={
        'TargetValue': 70.0,
        'PredefinedMetricSpecification': {
            'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
        }
    }
)
"""
            base_code += scaling_code
        
        return [types.TextContent(
            type="text",
            text=f"Complete deployment code:\n\n```python{base_code}```"
        )]
    
    elif name == "list_deployment_options":
        options = """
**Available Deployment Options:**

**Instance Types by Model Size:**

**Small Models (0.5B-1B parameters):**
- ml.m5.large (CPU) - $0.10/hour - Low cost, basic performance
- ml.g5.xlarge (GPU) - $1.00/hour - Better performance, faster inference

**Medium Models (1B-3B parameters):**
- ml.m5.xlarge (CPU) - $0.20/hour - Cost-effective for low traffic
- ml.g5.2xlarge (GPU) - $1.50/hour - Recommended for production

**Large Models (7B+ parameters):**
- ml.g5.4xlarge (GPU) - $2.50/hour - Required for large models
- ml.g5.12xlarge (GPU) - $7.00/hour - High performance, multiple GPUs

**Deployment Features:**
- Auto-scaling: Automatically adjust instance count based on traffic
- Multi-instance: Deploy across multiple instances for high availability
- Custom containers: Use HuggingFace LLM containers for optimized inference
- Cost optimization: Auto-select appropriate instance types

**Best Practices:**
- Start with single instance for testing
- Use GPU instances for production workloads
- Enable auto-scaling for variable traffic
- Monitor costs and performance metrics
"""
        
        return [types.TextContent(type="text", text=options)]
    
    elif name == "estimate_deployment_cost":
        instance_type = arguments.get("instance_type")
        instance_count = arguments.get("instance_count", 1)
        hours_per_day = arguments.get("hours_per_day", 24)
        
        # Cost mapping (approximate AWS pricing)
        costs = {
            "ml.m5.large": 0.10,
            "ml.m5.xlarge": 0.20,
            "ml.m5.2xlarge": 0.40,
            "ml.m5.4xlarge": 0.80,
            "ml.g5.xlarge": 1.00,
            "ml.g5.2xlarge": 1.50,
            "ml.g5.4xlarge": 2.50,
            "ml.g5.12xlarge": 7.00
        }
        
        hourly_cost = costs.get(instance_type, 0)
        if hourly_cost == 0:
            return [types.TextContent(
                type="text",
                text=f"Unknown instance type: {instance_type}. Available types: {list(costs.keys())}"
            )]
        
        daily_cost = hourly_cost * instance_count * hours_per_day
        monthly_cost = daily_cost * 30
        
        estimate = f"""
**Cost Estimate for {instance_type}:**

**Configuration:**
- Instance Type: {instance_type}
- Instance Count: {instance_count}
- Usage: {hours_per_day} hours/day

**Costs:**
- Per Instance: ${hourly_cost:.2f}/hour
- Total Hourly: ${hourly_cost * instance_count:.2f}/hour
- Daily Cost: ${daily_cost:.2f}/day
- Monthly Cost: ${monthly_cost:.2f}/month

**Cost Optimization Tips:**
- Use CPU instances for development/testing
- Enable auto-scaling to reduce idle costs
- Consider spot instances for non-critical workloads
- Monitor usage patterns and adjust accordingly
"""
        
        return [types.TextContent(type="text", text=estimate)]
    
    else:
        return [types.TextContent(
            type="text",
            text=f"Unknown tool: {name}"
        )]

async def main():
    """Main entry point"""
    # Check if SAMA RL is available
    try:
        import sama_rl
    except ImportError:
        print("ERROR: SAMA RL not installed. Please run: pip install -e . from the project root directory")
        sys.exit(1)
    
    # Run server
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)

if __name__ == "__main__":
    asyncio.run(main())
