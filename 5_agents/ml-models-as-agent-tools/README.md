# Using ML Models as Agent Tools with Amazon SageMaker AI

This repository demonstrates how to integrate machine learning models deployed on Amazon SageMaker as tools for AI agents using the Model Context Protocol (MCP).

## Overview

Learn how to:
1. Train and deploy ML models on Amazon SageMaker
2. Create tool interfaces using MCP
3. Integrate ML capabilities into AI agent workflows
4. Use both direct MCP servers and Amazon Bedrock AgentCore

## Architecture

The solution enables AI agents to leverage specialized ML models for enhanced decision-making through standardized tool interfaces.

## Approaches

### 1. Direct MCP Implementation (`1-direct/`)
- Direct MCP server implementation
- Custom tool creation using FastMCP
- Local and remote testing capabilities

**Key Files:**
- `demand_forecasting.ipynb` - Model training and deployment
- `server.py` - MCP server implementation
- `strands-agents-sagemaker-as-tool.ipynb` - Agent integration example

### 2. Amazon Bedrock AgentCore (`2-agentcore/`)
- Managed MCP hosting with AgentCore Gateway and Runtime
- Smithy-based API integration
- Enterprise-grade authentication and scaling

**Key Files:**
- `1-demand_forecasting.ipynb` - Model training and deployment
- `2-agentcore-gateway.ipynb` - Gateway setup using Smithy models
- `3-agentcore-runtime.ipynb` - Custom MCP server hosting
- `9-cleanup.ipynb` - Resource cleanup

## Prerequisites

- AWS account with SageMaker access
- Python 3.8+
- Basic understanding of ML concepts
- Familiarity with Jupyter notebooks

## Quick Start

1. **Choose your approach:**
   - For simple prototyping: Use `1-direct/`
   - For production deployments: Use `2-agentcore/`

2. **Train and deploy the model:**
   ```bash
   # Navigate to your chosen approach
   cd 1-direct/  # or 2-agentcore/
   
   # Run the demand forecasting notebook
   jupyter notebook demand_forecasting.ipynb
   ```

3. **Set up the MCP integration:**
   - **Direct:** Run `server.py` and test with the Strands Agents notebook
   - **AgentCore:** Follow the gateway and runtime setup notebooks

4. **Test with AI agents:**
   Use the provided example notebooks to see agents making predictions via the ML model

## Key Components

- **XGBoost Model:** Demand forecasting with time series features
- **MCP Server:** Tool interface for agent communication
- **SageMaker Endpoint:** Real-time model inference
- **Agent Integration:** AI agents using ML predictions

## Learning Objectives

- Understand ML model deployment on SageMaker
- Learn MCP tool creation and integration
- Gain experience with agent-ML workflows
- Explore both direct and managed hosting approaches

## Cleanup

Remember to clean up AWS resources to avoid charges.

## License

MIT-0 License. See LICENSE file for details.
