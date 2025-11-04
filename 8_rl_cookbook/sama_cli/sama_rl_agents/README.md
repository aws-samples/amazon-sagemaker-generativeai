# SAMA RL MCP Agents

Model Context Protocol (MCP) servers for SAMA RL integration with Amazon Q CLI. These agents help users fine-tune language models using GRPO with natural language interactions.

## Overview

Three specialized MCP servers provide end-to-end GRPO fine-tuning capabilities:

1. **Model Builder** - Create training configurations and reward functions
2. **Model Deployment** - Deploy trained models to SageMaker endpoints  
3. **Model Evaluation** - Evaluate model performance with various metrics

## Installation

### Prerequisites
- Python 3.8+
- MCP library: `pip install mcp`
- SAMA RL (automatically installed by agents)

### Setup with Q CLI

1. Copy the `sama_rl_agents` folder to your project directory
2. Add MCP server configuration to your Q CLI config:

```json
{
  "mcpServers": {
    "sama-rl-model-builder": {
      "command": "python",
      "args": ["model_builder.py"],
      "cwd": "path/to/sama_rl_agents"
    },
    "sama-rl-model-deployment": {
      "command": "python", 
      "args": ["model_deployment.py"],
      "cwd": "path/to/sama_rl_agents"
    },
    "sama-rl-model-evaluation": {
      "command": "python",
      "args": ["model_evaluation.py"], 
      "cwd": "path/to/sama_rl_agents"
    }
  }
}
```

## Usage Examples

### Model Builder Agent

**Natural Language Requests:**
- "I want to train a model to generate 400-token summaries"
- "Create a sentiment-aware reward function for customer service"
- "Recommend training configuration for a 1.5B parameter model"
- "Generate GRPO training code for length control"

**Available Tools:**
- `list_available_models` - Show available model recipes
- `create_reward_function` - Create length/sentiment/custom reward functions
- `recommend_training_config` - Get training recommendations
- `generate_training_code` - Generate complete training code

### Model Deployment Agent

**Natural Language Requests:**
- "Deploy my trained model to a cost-effective endpoint"
- "Recommend instance type for a 0.5B model with high traffic"
- "Estimate deployment costs for ml.g5.2xlarge"
- "Generate deployment code with auto-scaling"

**Available Tools:**
- `recommend_deployment_instance` - Instance type recommendations
- `deploy_model` - Deploy trained models
- `generate_deployment_code` - Generate deployment code
- `estimate_deployment_cost` - Cost estimation

### Model Evaluation Agent

**Natural Language Requests:**
- "Evaluate my model's length control performance"
- "Set up comprehensive quality assessment"
- "Compare my model against baseline"
- "Create evaluation report for stakeholders"

**Available Tools:**
- `list_evaluation_techniques` - Show available evaluation methods
- `recommend_evaluation_strategy` - Get evaluation recommendations
- `generate_evaluation_code` - Generate evaluation code
- `create_evaluation_report` - Create report templates

## Workflow Example

```bash
# 1. Build Model Configuration
q chat "I want to train Qwen2-0.5B to generate 300-token responses for customer support"

# 2. Deploy Trained Model  
q chat "Deploy training job sama-grpo-qwen205b-123456 to a cost-effective endpoint"

# 3. Evaluate Model Performance
q chat "Evaluate the deployed model for length control and sentiment analysis"
```

## Features

### Intelligent Recommendations
- Automatic instance type selection based on model size
- Cost optimization suggestions
- Training parameter recommendations
- Evaluation strategy recommendations

### Recipe-Based Approach
- Uses existing SAMA RL recipes from `sama_rl/recipes/`
- Extensible to new model architectures
- Consistent configuration management

### Natural Language Interface
- Conversational interaction through Q CLI
- Context-aware responses
- Step-by-step guidance

### Production Ready
- Cost estimation and optimization
- Auto-scaling configuration
- Comprehensive evaluation metrics
- Report generation

## Extending the Agents

### Adding New Models
Add new YAML recipes to `sama_rl/recipes/GRPO/`:
```yaml
# new-model-config.yaml
model:
  name: "new/model-name"
  # ... configuration
```

### Adding New Evaluation Techniques
Extend `EVALUATION_TECHNIQUES` in `model_evaluation.py`:
```python
EVALUATION_TECHNIQUES["new_technique"] = {
    "description": "New evaluation method",
    "metrics": ["metric1", "metric2"],
    "use_cases": ["use_case1", "use_case2"]
}
```

### Adding New Reward Functions
Extend reward function creation in `model_builder.py`:
```python
def create_custom_reward():
    # Implementation
    pass
```

## Architecture

```
Q CLI
  ├── Model Builder MCP Server
  │   ├── Recipe Management
  │   ├── Reward Function Creation
  │   └── Training Code Generation
  │
  ├── Model Deployment MCP Server  
  │   ├── Instance Recommendations
  │   ├── Cost Estimation
  │   └── Deployment Automation
  │
  └── Model Evaluation MCP Server
      ├── Evaluation Strategies
      ├── Metrics Calculation
      └── Report Generation
```

## Best Practices

1. **Start Small** - Begin with small models and datasets for testing
2. **Cost Monitoring** - Use cost estimation tools before deployment
3. **Evaluation First** - Set up evaluation before training
4. **Iterative Improvement** - Use evaluation results to improve models
5. **Documentation** - Generate reports for stakeholder communication

## Troubleshooting

### Common Issues
- **MCP Connection**: Ensure Python path and working directory are correct
- **SAMA RL Installation**: Agents auto-install SAMA RL from parent directory
- **AWS Credentials**: Ensure AWS credentials are configured for SageMaker access
- **Recipe Not Found**: Check that recipe files exist in `sama_rl/recipes/GRPO/`

### Debug Mode
Run agents directly for debugging:
```bash
cd sama_rl_agents
python model_builder.py
```

## Contributing

1. Fork the repository
2. Add new features to appropriate agent
3. Update tool definitions and documentation
4. Test with Q CLI integration
5. Submit pull request

## License

Same as SAMA RL project license.
