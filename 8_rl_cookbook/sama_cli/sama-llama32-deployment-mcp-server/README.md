# SAMA Deployment MCP Server

A Model Context Protocol (MCP) server for SageMaker model deployment and management, specifically designed for SAMA (SageMaker AI Model Agents) context. This server can be used with Q CLI or as a standalone service for distributed agent architectures.

## Overview

This MCP server provides deployment capabilities for SAMA agents, enabling natural language interactions for SageMaker model deployment, management, and inference operations. It supports both stdio-based MCP communication (for Q CLI integration) and standalone network deployment for distributed systems.

## Features

This MCP server provides the following tools for SAMA agent operations:

### Core Deployment Tools

#### `deploy_jumpstart_model`
Deploy SageMaker JumpStart models to endpoints:
- Support for various JumpStart model IDs (LLaMA, Mistral, etc.)
- Configurable instance types and counts
- Automatic endpoint naming or custom names
- EULA acceptance handling
- Real-time deployment status monitoring

#### `create_inference_payload`
Create inference payloads for model prediction:
- Text generation parameters (temperature, top_p, max_new_tokens)
- Optional parameters (seed, repetition_penalty, do_sample, top_k)
- Support for both pretrained and fine-tuned models
- JSON-formatted output ready for SageMaker inference

#### `create_instruction_payload`
Create instruction tuning payloads:
- Structured instruction format
- Context and instruction separation
- Template-based formatting for fine-tuned models

### Endpoint Management Tools

#### `list_sagemaker_endpoints`
List SageMaker endpoints:
- Filter by status (InService, Creating, Updating, etc.)
- Configurable result limits
- Endpoint metadata and configuration details

#### `describe_sagemaker_endpoint`
Get detailed endpoint information:
- Endpoint configuration details
- Status and health information
- Production variant details
- Instance information and scaling configuration

#### `delete_sagemaker_endpoint`
Delete SageMaker endpoints:
- Endpoint deletion with confirmation
- Optional endpoint configuration cleanup
- Safety checks and error handling

### Optimization Tools

#### `get_deployment_recommendations`
Get deployment recommendations:
- Instance type suggestions based on model size
- Use case optimization (general, high-throughput, low-latency)
- Cost and performance trade-off analysis
- Scaling recommendations

## Installation

### Basic Installation
```bash
cd /path/to/sama-deployment-mcp-server
pip install -e .
```

### With Additional Dependencies (for standalone mode)
```bash
pip install -e .
pip install uvicorn fastapi  # For HTTP wrapper mode
```

## Usage Modes

### 1. Q CLI Integration (Recommended)

#### Configuration
The server integrates with Q CLI through MCP configuration files. Q CLI will automatically detect and use the appropriate configuration:

**Global Configuration** (`~/.aws/amazonq/mcp.json`):
```json
{
  "mcpServers": {
    "sama-deployment-mcp-server": {
      "command": "sama-deployment-mcp-server",
      "disabled": false,
      "autoApprove": []
    }
  }
}
```

**Project-Specific Configuration** (`.amazonq/mcp.json` in project directory):
```json
{
  "mcpServers": {
    "sama-deployment-mcp-server": {
      "command": "sama-deployment-mcp-server",
      "args": ["--allow-write"],
      "env": {
        "AWS_REGION": "us-east-2",
        "SAMA_CONTEXT": "true",
        "FASTMCP_LOG_LEVEL": "INFO"
      },
      "timeout": 120000,
      "disabled": false,
      "autoApprove": ["*"],
      "description": "SAMA deployment server for SageMaker model operations",
      "context": "sama-agents"
    }
  }
}
```

#### Starting with Q CLI
```bash
# Q CLI automatically launches MCP servers based on configuration
q chat

# In Q CLI, you can now use natural language commands:
# "Deploy meta-textgeneration-llama-3-2-3b on ml.g5.xlarge"
# "List my SageMaker endpoints"
# "Get deployment recommendations for LLaMA model"
```

#### Q CLI Usage Examples
```bash
# Start Q CLI session
q chat

# Natural language commands you can use:
# - "Deploy meta-textgeneration-llama-3-2-3b model"
# - "Create inference payload for text generation with temperature 0.8"
# - "List all my SageMaker endpoints that are InService"
# - "Show details for my LLaMA endpoint"
# - "Delete endpoint jumpstart-llama-20241201"
# - "Get deployment recommendations for high-throughput use case"
```

### 2. Standalone Network Mode

For distributed agent architectures or when you need multiple agents to access the same MCP server.

#### Option A: HTTP REST API Wrapper

Launch as HTTP service:
```bash
# Start HTTP wrapper server
python http_wrapper.py --host 0.0.0.0 --port 8001

# Server will be available at http://your-ip:8001
# API documentation at http://your-ip:8001/docs
```

**API Endpoints:**
- `POST /deploy` - Deploy a model
- `GET /endpoints` - List endpoints
- `GET /endpoints/{name}` - Describe endpoint
- `DELETE /endpoints/{name}` - Delete endpoint
- `POST /inference-payload` - Create inference payload
- `POST /instruction-payload` - Create instruction payload
- `POST /recommendations` - Get deployment recommendations

**Example API Usage:**
```bash
# Deploy a model
curl -X POST "http://localhost:8001/deploy" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "meta-textgeneration-llama-3-2-3b",
    "instance_type": "ml.g5.xlarge",
    "initial_instance_count": 1
  }'

# List endpoints
curl "http://localhost:8001/endpoints"

# Get recommendations
curl -X POST "http://localhost:8001/recommendations" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "meta-textgeneration-llama-3-2-3b",
    "use_case": "high-throughput"
  }'
```

#### Option B: Docker Container

Build and run as container:
```bash
# Build container
docker build -t sama-deployment-server .

# Run container
docker run -p 8001:8001 \
  -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  -e AWS_REGION=us-east-2 \
  sama-deployment-server
```

#### Option C: Direct MCP Network Server

```bash
# Launch standalone MCP server (network mode)
python standalone_server.py --host 0.0.0.0 --port 8001
```

### 3. Programmatic Integration

For custom agent implementations:

```python
from sama_deployment_mcp_server.server import (
    deploy_jumpstart_model,
    list_sagemaker_endpoints,
    get_deployment_recommendations
)

# Deploy a model
result = await deploy_jumpstart_model(
    model_id="meta-textgeneration-llama-3-2-3b",
    instance_type="ml.g5.xlarge"
)

# List endpoints
endpoints = await list_sagemaker_endpoints(status_filter="InService")

# Get recommendations
recommendations = await get_deployment_recommendations(
    model_id="meta-textgeneration-llama-3-2-3b",
    use_case="high-throughput"
)
```

## Prerequisites

### AWS Configuration
- AWS credentials configured (`aws configure` or environment variables)
- SageMaker permissions for model deployment
- Sufficient service limits for chosen instance types

### Required Permissions
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "sagemaker:CreateModel",
        "sagemaker:CreateEndpointConfig",
        "sagemaker:CreateEndpoint",
        "sagemaker:DescribeEndpoint",
        "sagemaker:DescribeEndpointConfig",
        "sagemaker:DescribeModel",
        "sagemaker:ListEndpoints",
        "sagemaker:DeleteEndpoint",
        "sagemaker:DeleteEndpointConfig",
        "sagemaker:DeleteModel",
        "iam:PassRole"
      ],
      "Resource": "*"
    }
  ]
}
```

## Configuration Options

### Environment Variables
- `AWS_REGION` - AWS region for deployments (default: us-east-2)
- `ALLOW_WRITE` - Enable write operations (default: true)
- `SAMA_CONTEXT` - Enable SAMA-specific optimizations (default: true)
- `FASTMCP_LOG_LEVEL` - Logging level (INFO, DEBUG, WARNING, ERROR)

### Command Line Arguments
```bash
sama-deployment-mcp-server --help

Options:
  --allow-write     Allow write operations (model deployment, endpoint deletion)
  --version         Show version information
```

## Monitoring and Logging

### Server Status
Check if MCP servers are running:
```bash
ps aux | grep sama-deployment-mcp-server
```

### Logs
- Q CLI mode: Logs appear in Q CLI output
- Standalone mode: Logs to stdout/stderr
- HTTP mode: Access logs and application logs

### Health Checks
For HTTP wrapper mode:
```bash
curl http://localhost:8001/health
```

## Supported Models

The server supports all SageMaker JumpStart models, with optimized recommendations for:

### Text Generation Models
- `meta-textgeneration-llama-3-2-3b`
- `meta-textgeneration-llama-3-2-1b`
- `meta-textgeneration-llama-3-1-8b`
- `mistral-7b-instruct`
- `anthropic-claude-3-sonnet`

### Recommended Instance Types
- **Small models (1-3B parameters)**: ml.g5.xlarge, ml.g5.2xlarge
- **Medium models (7-13B parameters)**: ml.g5.4xlarge, ml.g5.12xlarge
- **Large models (70B+ parameters)**: ml.g5.48xlarge, ml.p4d.24xlarge

## Troubleshooting

### Common Issues

#### AWS Credentials
```bash
# Verify AWS credentials
aws sts get-caller-identity

# Configure if needed
aws configure
```

#### Service Limits
- Check SageMaker service quotas in AWS Console
- Request limit increases for required instance types

#### Endpoint Deployment Failures
- Check CloudWatch logs for detailed error messages
- Verify IAM role permissions
- Ensure sufficient capacity in target region

### Debug Mode
Enable debug logging:
```bash
export FASTMCP_LOG_LEVEL=DEBUG
sama-deployment-mcp-server
```

## Development

### Project Structure
```
sama-deployment-mcp-server/
├── sama_deployment_mcp_server/
│   ├── __init__.py
│   └── server.py              # Main MCP server implementation
├── standalone_server.py       # Network mode launcher
├── http_wrapper.py           # HTTP REST API wrapper
├── Dockerfile               # Container configuration
├── pyproject.toml          # Package configuration
└── README.md              # This file
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make changes and add tests
4. Submit a pull request

### Testing
```bash
# Run basic functionality test
python test_deployment.py

# Test with polling
python test_deployment_with_polling.py

# Check endpoints
python check_endpoints.py
```

## Error Handling

The server includes comprehensive error handling for:
- AWS credential issues
- Service limit constraints
- Invalid model IDs or configurations
- Network and deployment failures
- Timeout scenarios
- Resource cleanup on failures

## Security Considerations

- AWS credentials are never logged or exposed
- All operations respect IAM permissions
- Network mode should be secured with appropriate firewall rules
- Consider using VPC endpoints for enhanced security
- Regular rotation of AWS credentials recommended

## Performance Optimization

- Deployment operations are asynchronous with proper polling
- Connection pooling for AWS API calls
- Configurable timeouts for long-running operations
- Resource cleanup to prevent orphaned resources
- Efficient endpoint status monitoring

## License

This project is part of the SAMA (SageMaker AI Model Agents) framework and follows the same licensing terms.
