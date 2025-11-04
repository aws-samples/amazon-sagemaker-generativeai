#!/usr/bin/env python3
"""
SAMA RL Model Builder MCP Server - Conversational GRPO Training Assistant
"""

import json
import sys
import os
import yaml
from pathlib import Path

# Global conversation state
conversation_state = {}

def get_available_recipes():
    """Get available GRPO recipes with details - ONLY use existing sama_rl recipes"""
    recipes_dir = Path(__file__).parent.parent / "sama_rl" / "recipes" / "GRPO"
    recipes = {}
    
    if not recipes_dir.exists():
        return {}
    
    for yaml_file in recipes_dir.glob("*.yaml"):
        try:
            with open(yaml_file, 'r') as f:
                config = yaml.safe_load(f)
                model_name = config.get('model', {}).get('name', 'Unknown')
                dataset = config.get('data', {}).get('dataset_name', 'Unknown')
                instance_type = config.get('sagemaker', {}).get('instance_type', 'ml.p4d.24xlarge')
                max_steps = config.get('training', {}).get('max_steps', 800)
                
                recipes[yaml_file.stem] = {
                    'path': str(yaml_file),
                    'model': model_name,
                    'dataset': dataset,
                    'instance_type': instance_type,
                    'max_steps': max_steps
                }
        except:
            recipes[yaml_file.stem] = {'path': str(yaml_file), 'model': 'Unknown', 'dataset': 'Unknown'}
    
    return recipes

def get_instance_costs():
    """Get estimated hourly costs for different instance types"""
    return {
        'ml.g4dn.xlarge': {'cost': '$0.526/hr', 'description': 'Small models (0.5B-1B params)', 'gpu': '1x T4'},
        'ml.g4dn.2xlarge': {'cost': '$0.752/hr', 'description': 'Medium models (1B-3B params)', 'gpu': '1x T4'},
        'ml.g5.xlarge': {'cost': '$1.006/hr', 'description': 'Small-medium models', 'gpu': '1x A10G'},
        'ml.g5.2xlarge': {'cost': '$1.515/hr', 'description': 'Medium models (1B-7B params)', 'gpu': '1x A10G'},
        'ml.g5.4xlarge': {'cost': '$2.03/hr', 'description': 'Large models (7B+ params)', 'gpu': '1x A10G'},
        'ml.p4d.24xlarge': {'cost': '$32.77/hr', 'description': 'Very large models (13B+ params)', 'gpu': '8x A100'},
        'ml.p5.48xlarge': {'cost': '$98.32/hr', 'description': 'Massive models (70B+ params)', 'gpu': '8x H100'}
    }

def check_wandb_config():
    """Check if wandb is configured"""
    wandb_key = os.environ.get('WANDB_API_KEY')
    if wandb_key:
        return f"✓ Using configured wandb API key: {wandb_key[:8]}..."
    return None

def parse_initial_request(user_id, request):
    """Parse initial request and start conversation with follow-up questions - ONLY use existing recipes"""
    # Check available recipes first
    recipes = get_available_recipes()
    
    if not recipes:
        return """❌ **No GRPO recipes found**

I cannot find any GRPO recipes in sama_rl/recipes/GRPO/. 

Available recipes are required to use the sama_rl framework. Please ensure you have valid recipe files in:
- sama_rl/recipes/GRPO/

I can only work with existing sama_rl recipes and will not create new training scripts or configurations."""
    
    # Only use available recipes
    available_recipe_names = list(recipes.keys())
    default_recipe = available_recipe_names[0]  # Use first available recipe
    
    conversation_state[user_id] = {
        'step': 'instance_confirmation',
        'initial_request': request,
        'recipe': default_recipe,
        'available_recipes': recipes,
        'instance_type': None,
        'max_steps': None,
        'wandb_key': None,
        'reward_function': 'length',
        'hyperparams': {}
    }
    
    recipe_info = recipes[default_recipe]
    costs = get_instance_costs()
    
    # Recommend instance based on available recipe
    recommended = ['ml.g4dn.xlarge', 'ml.g4dn.2xlarge', 'ml.g5.2xlarge']
    
    cost_list = "\n".join([f"• **{inst}**: {costs[inst]['cost']} - {costs[inst]['description']} ({costs[inst]['gpu']})" 
                          + (" ⭐ **Recommended**" if inst in recommended else "")
                          for inst in costs.keys()])
    
    return f"""🚀 **GRPO Training Setup** (Using existing sama_rl recipes only)

**Available Recipe**: {default_recipe}
- Model: {recipe_info['model']}
- Dataset: {recipe_info['dataset']}
- Default Steps: {recipe_info['max_steps']}

**1. Instance Type Selection:**
{cost_list}

Which instance type would you prefer? I recommend **ml.g4dn.2xlarge** for cost-effectiveness."""

def handle_instance_confirmation(user_id, user_input):
    """Handle instance type confirmation and move to hyperparameters"""
    costs = get_instance_costs()
    instance_type = user_input.strip()
    
    if instance_type in costs:
        conversation_state[user_id]['instance_type'] = instance_type
        conversation_state[user_id]['step'] = 'hyperparameters'
        
        suggested_steps = conversation_state[user_id].get('suggested_steps', 100)
        
        return f"""✅ **Instance Selected**: {instance_type} ({costs[instance_type]['cost']})

**2. Hyperparameters Configuration:**

Based on your request for 100 steps, here are my recommendations:

**Training Steps**: {suggested_steps} (as requested)
- Quick test: 10-50 steps
- Short training: 100-200 steps  
- Full training: 500-1000+ steps

**Learning Rate Options:**
- Conservative: 1e-5 (safer, slower learning)
- **Recommended**: 5e-5 (recipe default)
- Aggressive: 1e-4 (faster, but riskier)

**Batch Size Options:**
- Small: 2 (memory efficient)
- **Recommended**: 4 (recipe default)
- Large: 8 (faster training, more memory)

Would you like to:
1. **Use recommended settings** (100 steps, 5e-5 LR, batch size 4)
2. **Customize hyperparameters** (I'll ask about each one)

What's your preference?"""
    
    else:
        available = ", ".join(costs.keys())
        return f"❌ Instance type '{instance_type}' not recognized. Available: {available}"

def handle_hyperparameters(user_id, user_input):
    """Handle hyperparameter selection"""
    user_input = user_input.strip().lower()
    
    if 'recommended' in user_input or '1' in user_input:
        conversation_state[user_id]['max_steps'] = conversation_state[user_id].get('suggested_steps', 100)
        conversation_state[user_id]['hyperparams'] = {
            'learning_rate': '5e-5',
            'batch_size': 4,
            'temperature': 0.7
        }
        conversation_state[user_id]['step'] = 'reward_confirmation'
        
        return f"""✅ **Hyperparameters Set**:
- Steps: {conversation_state[user_id]['max_steps']}
- Learning Rate: 5e-5 (recipe default)
- Batch Size: 4
- Temperature: 0.7

**3. Reward Function Confirmation:**

You mentioned keeping summaries to 400 tokens. I'll use the length reward function from reward_functions.py:

```python
length_400_reward = create_length_reward(400)
```

This will:
- Target exactly 400 tokens
- Penalize summaries that are too long or too short
- Use quadratic penalty for distance from target

Would you also like to add:
1. **Just length control** (as requested)
2. **Length + quality control** (also penalize repetition)
3. **Custom reward function**

What's your preference?"""
    
    elif 'customize' in user_input or '2' in user_input:
        conversation_state[user_id]['step'] = 'custom_hyperparams'
        return """**Custom Hyperparameter Setup:**

Let's configure each parameter:

**Training Steps**: How many steps? (you mentioned 100)
- 10-50: Quick test
- 100-200: Short training
- 500+: Full training

Please specify the number of steps:"""
    
    else:
        return "Please choose: 1 for recommended settings, or 2 to customize hyperparameters."

def handle_steps_selection(user_id, user_input):
    """Handle training steps selection"""
    user_input = user_input.strip().lower()
    
    if user_input == "default":
        recipes = get_available_recipes()
        recipe_name = conversation_state[user_id]['recipe']
        max_steps = recipes[recipe_name]['max_steps']
    else:
        try:
            max_steps = int(user_input)
        except:
            return "❌ Please enter a number or 'default'"
    
    conversation_state[user_id]['max_steps'] = max_steps
    conversation_state[user_id]['step'] = 'reward_function'
    
    return f"""✅ **Training Steps**: {max_steps}

**Reward Function:**
I can suggest reward functions based on your use case:
- **length**: Controls output length (good for summarization)
- **quality**: Focuses on output quality
- **custom**: You provide your own function
- **none**: Use default GRPO rewards

What type of reward function would you like?"""

def handle_reward_function(user_id, user_input):
    """Handle reward function selection"""
    user_input = user_input.strip().lower()
    
    reward_functions = {
        'length': '''def length_reward(completions, **kwargs):
    target_length = 400
    tokenizer = kwargs.get('tokenizer')
    rewards = []
    for completion in completions:
        if tokenizer:
            num_tokens = len(tokenizer.encode(completion, add_special_tokens=False))
        else:
            num_tokens = len(completion.split())
        reward = -(abs(num_tokens - target_length) ** 2) / 1000
        rewards.append(reward)
    return rewards''',
        
        'quality': '''def quality_reward(completions, **kwargs):
    # Simple quality heuristic - penalize repetition, reward coherence
    rewards = []
    for completion in completions:
        words = completion.split()
        unique_ratio = len(set(words)) / max(len(words), 1)
        reward = unique_ratio * 2 - 1  # Scale to [-1, 1]
        rewards.append(reward)
    return rewards''',
        
        'none': '[]  # No custom reward functions',
        'custom': '# You can add your custom reward function here'
    }
    
    if user_input in reward_functions:
        conversation_state[user_id]['reward_function'] = user_input
        conversation_state[user_id]['step'] = 'wandb_check'
        
        wandb_status = check_wandb_config()
        if wandb_status:
            return handle_final_summary(user_id, wandb_status)
        else:
            return f"""✅ **Reward Function**: {user_input}

**Weights & Biases Setup:**
I don't see a wandb API key configured. Would you like to:
- Provide your wandb API key for experiment tracking
- Skip wandb tracking (just say "skip")

If you have a wandb key, please provide it."""
    
    else:
        return "❌ Please choose: length, quality, custom, or none"

def handle_wandb_setup(user_id, user_input):
    """Handle wandb API key setup"""
    user_input = user_input.strip()
    
    if user_input.lower() == 'skip':
        conversation_state[user_id]['wandb_key'] = None
        return handle_final_summary(user_id, "Skipping wandb tracking")
    else:
        conversation_state[user_id]['wandb_key'] = user_input
        return handle_final_summary(user_id, f"✓ Using provided wandb API key: {user_input[:8]}...")

def handle_final_summary(user_id, wandb_status):
    """Generate final training summary and code - ONLY using sama_rl framework"""
    state = conversation_state[user_id]
    recipes = state['available_recipes']
    costs = get_instance_costs()
    
    recipe_info = recipes[state['recipe']]
    cost_info = costs[state['instance_type']]
    
    # Estimate cost
    estimated_time = max(state['max_steps'] / 100, 0.5)  # Rough estimate
    estimated_cost = float(cost_info['cost'].split('$')[1].split('/')[0]) * estimated_time
    
    # Generate training code using ONLY sama_rl framework
    reward_functions = {
        'length': '''def length_reward(completions, **kwargs):
    target_length = 400
    tokenizer = kwargs.get('tokenizer')
    rewards = []
    for completion in completions:
        if tokenizer:
            num_tokens = len(tokenizer.encode(completion, add_special_tokens=False))
        else:
            num_tokens = len(completion.split())
        reward = -(abs(num_tokens - target_length) ** 2) / 1000
        rewards.append(reward)
    return rewards

reward_funcs = [length_reward]''',
        'quality': '''def quality_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        words = completion.split()
        unique_ratio = len(set(words)) / max(len(words), 1)
        reward = unique_ratio * 2 - 1
        rewards.append(reward)
    return rewards

reward_funcs = [quality_reward]''',
        'none': 'reward_funcs = []',
        'custom': 'reward_funcs = []  # Add your custom functions here'
    }
    
    reward_code = reward_functions.get(state['reward_function'], 'reward_funcs = []')
    wandb_param = f'wandb_api_key="{state["wandb_key"]}",' if state['wandb_key'] else ''
    
    # ONLY use sama_rl framework - no new scripts
    training_code = f'''from sama_rl import GRPO

{reward_code}

trainer = GRPO(
    yaml_file="{recipe_info['path']}",
    reward_functions=reward_funcs,
    max_steps={state['max_steps']},
    instance_type="{state['instance_type']}",
    {wandb_param}
)

trainer.train()
print(f"Training completed. Job name: {{trainer.training_job_name}}")'''
    
    summary = f"""🎯 **Training Job Summary** (Using sama_rl framework only)

**Configuration:**
- Recipe: {state['recipe']} (existing sama_rl recipe)
- Model: {recipe_info['model']}
- Dataset: {recipe_info['dataset']}
- Instance: {state['instance_type']} ({cost_info['cost']})
- Steps: {state['max_steps']}
- Reward: {state['reward_function']}
- {wandb_status}

**Estimated Cost**: ~${estimated_cost:.2f} ({estimated_time:.1f} hours)

**Training Code:**
```python
{training_code}
```

Ready to start training! Run the code above to launch your GRPO training job.

⚠️ **Note**: This uses only the existing sama_rl framework and recipes. No new scripts or configurations will be created."""
    
    # Reset conversation
    del conversation_state[user_id]
    
    return summary

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
                        "serverInfo": {"name": "sama-rl-model-builder", "version": "1.0.0"}
                    }
                }
            elif method == "tools/list":
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "tools": [
                            {
                                "name": "setup_grpo_training",
                                "description": "Interactive GRPO training setup with follow-up questions",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "initial_request": {"type": "string", "description": "Your training requirements"}
                                    },
                                    "required": ["initial_request"]
                                }
                            },
                            {
                                "name": "continue_conversation",
                                "description": "Continue the training setup conversation",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "message": {"type": "string", "description": "Your response to continue the conversation"}
                                    },
                                    "required": ["message"]
                                }
                            }
                        ]
                    }
                }
            elif method == "tools/call":
                tool_name = request.get("params", {}).get("name")
                args = request.get("params", {}).get("arguments", {})
                user_id = "default"  # Simple user ID for now
                
                if tool_name == "setup_grpo_training":
                    initial_request = args.get("initial_request", "")
                    result = parse_initial_request(user_id, initial_request)
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {"content": [{"type": "text", "text": result}]}
                    }
                elif tool_name == "continue_conversation":
                    message = args.get("message", "")
                    
                    if user_id not in conversation_state:
                        result = "No active conversation. Please start with 'start_training_setup'."
                    else:
                        step = conversation_state[user_id]['step']
                        
                        if step == 'instance_confirmation':
                            result = handle_instance_confirmation(user_id, message)
                        elif step == 'hyperparameters':
                            result = handle_hyperparameters(user_id, message)
                        elif step == 'reward_confirmation':
                            result = handle_reward_function(user_id, message)
                        elif step == 'wandb_check':
                            result = handle_wandb_setup(user_id, message)
                        elif step == 'custom_hyperparams':
                            result = handle_steps_selection(user_id, message)
                        else:
                            result = "Conversation completed. Start a new one with 'setup_grpo_training'."
                    
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {"content": [{"type": "text", "text": result}]}
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
