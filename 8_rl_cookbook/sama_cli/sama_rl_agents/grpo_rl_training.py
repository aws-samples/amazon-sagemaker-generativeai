#!/usr/bin/env python3
"""
GRPO RL Training MCP Server - ONLY for Reinforcement Learning tasks
Use this ONLY when you specifically want GRPO, PPO, or other RL training
"""

import json
import sys
import os
import yaml
from pathlib import Path

# Global conversation state
conversation_state = {}

def get_available_recipes():
    """Get available RL recipes - detects algorithm from request"""
    base_recipes_dir = Path(__file__).parent.parent / "sama_rl" / "recipes"
    all_recipes = {}
    
    if not base_recipes_dir.exists():
        return {}
    
    # Check for different RL algorithms
    for algorithm_dir in base_recipes_dir.iterdir():
        if algorithm_dir.is_dir() and algorithm_dir.name in ['GRPO', 'PPO', 'DPO', 'RLHF']:
            algorithm = algorithm_dir.name
            for yaml_file in algorithm_dir.glob("*.yaml"):
                try:
                    with open(yaml_file, 'r') as f:
                        config = yaml.safe_load(f)
                        model_name = config.get('model', {}).get('name', 'Unknown')
                        dataset = config.get('data', {}).get('dataset_name', 'Unknown')
                        
                        recipe_key = f"{algorithm.lower()}-{yaml_file.stem}"
                        all_recipes[recipe_key] = {
                            'path': str(yaml_file),
                            'algorithm': algorithm,
                            'model': model_name,
                            'dataset': dataset
                        }
                except:
                    recipe_key = f"{algorithm.lower()}-{yaml_file.stem}"
                    all_recipes[recipe_key] = {
                        'path': str(yaml_file), 
                        'algorithm': algorithm,
                        'model': 'Unknown', 
                        'dataset': 'Unknown'
                    }
    
    return all_recipes

def detect_rl_algorithm(request: str):
    """Detect RL algorithm from user request"""
    request_lower = request.lower()
    
    if 'grpo' in request_lower or 'group relative policy' in request_lower:
        return 'GRPO'
    elif 'ppo' in request_lower or 'proximal policy' in request_lower:
        return 'PPO'
    elif 'dpo' in request_lower or 'direct preference' in request_lower:
        return 'DPO'
    elif 'rlhf' in request_lower or 'reinforcement learning from human feedback' in request_lower:
        return 'RLHF'
    else:
        return 'GRPO'  # Default to GRPO

def setup_grpo_training(initial_request: str):
    """Setup RL training - detects algorithm and finds appropriate recipes"""
    recipes = get_available_recipes()
    
    if not recipes:
        return """❌ **No RL recipes found**

I cannot find any RL recipes in sama_rl/recipes/. 

Supported algorithms: GRPO, PPO, DPO, RLHF
Expected structure: sama_rl/recipes/GRPO/, sama_rl/recipes/PPO/, etc.

⚠️ **IMPORTANT**: This server is ONLY for Reinforcement Learning (RL) tasks.

For standard fine-tuning (Llama, GPT, etc.), use:
- standard_finetuning_data_prep
- standard_finetuning_training"""
    
    # Detect requested algorithm
    requested_algorithm = detect_rl_algorithm(initial_request)
    
    # Filter recipes by algorithm
    algorithm_recipes = {k: v for k, v in recipes.items() if v['algorithm'] == requested_algorithm}
    
    if not algorithm_recipes:
        available_algorithms = list(set(r['algorithm'] for r in recipes.values()))
        return f"""❌ **No {requested_algorithm} recipes found**

I don't have {requested_algorithm} recipes available.

**Available RL algorithms**: {', '.join(available_algorithms)}

**Available recipes**:
{chr(10).join([f"- {r['algorithm']}: {r['path']}" for r in recipes.values()])}

I cannot provide {requested_algorithm} capability right now."""
    
    # Show available recipes for the requested algorithm
    recipe_list = "\n".join([f"- {k}: {v['model']} on {v['dataset']}" for k, v in algorithm_recipes.items()])
    
    return f"""🤖 **{requested_algorithm} RL Training Setup**

**Available {requested_algorithm} recipes**:
{recipe_list}

**Algorithm detected**: {requested_algorithm}
**Recipe directory**: sama_rl/recipes/{requested_algorithm}/

⚠️ **Note**: This is for Reinforcement Learning. For standard fine-tuning, use standard_finetuning_* servers.

Which recipe would you like to use?"""

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
                        "serverInfo": {"name": "grpo-rl-training", "version": "1.0.0"}
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
                                "description": "Setup GRPO training - ONLY for Reinforcement Learning tasks",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "initial_request": {"type": "string", "description": "RL training requirements"}
                                    },
                                    "required": ["initial_request"]
                                }
                            }
                        ]
                    }
                }
            elif method == "tools/call":
                tool_name = request.get("params", {}).get("name")
                args = request.get("params", {}).get("arguments", {})
                
                if tool_name == "setup_grpo_training":
                    result = setup_grpo_training(args.get("initial_request", ""))
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
