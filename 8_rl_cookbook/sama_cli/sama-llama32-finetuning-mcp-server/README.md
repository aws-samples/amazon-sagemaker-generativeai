# SAMA Fine-tuning MCP Server

A Q CLI compatible MCP server for SageMaker model fine-tuning operations, specifically designed for SAMA (SageMaker AI Model Agents) context.

## Overview

This MCP server provides comprehensive fine-tuning capabilities for SAMA agents, enabling natural language interactions for SageMaker JumpStart model fine-tuning, monitoring, and model creation operations.

## Features

This MCP server provides the following tools for SAMA agent fine-tuning operations:

### `start_finetuning_job`
Start SageMaker fine-tuning jobs:
- Support for JumpStart model IDs
- Configurable hyperparameters (epochs, learning rate, batch size)
- Instance type and count configuration
- Automatic role detection
- EULA acceptance handling
- Custom job naming

### `monitor_training_job`
Monitor training job progress:
- Real-time status checking
- Training metrics and timing
- Failure reason reporting
- Resource configuration details
- Model artifacts location

### `create_model_from_training_job`
Create deployable models from training jobs:
- Automatic model creation from completed jobs
- Custom model naming
- Role and region auto-detection
- Integration with deployment pipeline

### `get_finetuning_recommendations`
Get fine-tuning recommendations:
- Instance type suggestions based on model size
- Hyperparameter recommendations by use case
- Cost and time estimates
- Dataset size considerations
- Best practices and tips

## Installation

```bash
cd /path/to/sama-finetuning-mcp-server
pip install -e .
```

## Configuration

Add to your MCP client configuration:

```json
{
  "mcpServers": {
    "sama-finetuning-mcp-server": {
      "command": "sama-finetuning-mcp-server",
      "disabled": false,
      "autoApprove": []
    }
  }
}
```

## Prerequisites

- AWS credentials configured
- SageMaker permissions for training jobs
- Sufficient service limits for chosen instance types
- Training data uploaded to S3

## Usage Examples

### Fine-tuning Operations
- "Start fine-tuning meta-textgeneration-llama-3-2-3b on ml.g5.xlarge with Dolly dataset"
- "Fine-tune LLaMA 3.2 3B model for 5 epochs with learning rate 2e-5"
- "Get fine-tuning recommendations for meta-textgeneration-llama-3-2-3b"

### Monitoring Operations
- "Monitor training job jumpstart-llama-finetune-20241201"
- "Check status of my fine-tuning job"
- "Show training job details and progress"

### Model Creation
- "Create model from completed training job jumpstart-llama-finetune-20241201"
- "Generate deployable model from fine-tuning artifacts"

## Supported Models

The server supports all SageMaker JumpStart text generation models, with optimized recommendations for:
- meta-textgeneration-llama-3-2-3b
- meta-textgeneration-llama-3-2-1b
- meta-textgeneration-llama-3-2-11b
- And other JumpStart language models

## Use Cases

### Instruction Following
- Optimized for chat and instruction-following tasks
- Recommended: 3 epochs, 2e-5 learning rate, batch size 4

### Summarization
- Optimized for text summarization tasks
- Recommended: 5 epochs, 1e-5 learning rate, longer input sequences

### Question Answering
- Optimized for QA tasks
- Recommended: 4 epochs, 1.5e-5 learning rate, balanced configuration

### General Purpose
- Balanced configuration for various tasks
- Recommended: 5 epochs, 2e-5 learning rate, standard settings

## Instance Type Recommendations

- **3B models**: ml.g5.xlarge (cost-effective)
- **7B models**: ml.g5.2xlarge (balanced performance)
- **11B+ models**: ml.g5.4xlarge (high performance)

## Error Handling

The server includes comprehensive error handling for:
- AWS credential and permission issues
- Training job failures and monitoring
- Resource limit constraints
- Invalid configurations
- Network and service failures

## Integration with Other SAMA Servers

This server is designed to work seamlessly with:
- **SAMA Data Prep MCP Server**: For dataset preparation
- **SAMA Deployment MCP Server**: For model deployment after fine-tuning

## Workflow Integration

Complete fine-tuning workflow:
1. **Data Prep**: Use data prep server to prepare training data
2. **Fine-tuning**: Use this server to start and monitor training
3. **Deployment**: Use deployment server to deploy fine-tuned model

Example integrated workflow:
```
"Fine-tune meta-textgeneration-llama-3-2-3b on Dolly dataset using ml.g5.xlarge, then deploy to endpoint"
```

This will automatically:
1. Prepare Dolly dataset (data prep server)
2. Start fine-tuning job (this server)
3. Monitor training progress (this server)
4. Create model from artifacts (this server)
5. Deploy to endpoint (deployment server)
