"""
YAML configuration loader for SAMA RL
"""
import yaml
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration container"""
    model: Dict[str, Any]
    data: Dict[str, Any]
    training: Dict[str, Any]
    algorithm: Dict[str, Any]  # grpo, ppo, orpo, dpo specific configs
    reward: Dict[str, Any]
    wandb: Dict[str, Any]
    output: Dict[str, Any]
    sagemaker: Optional[Dict[str, Any]] = None


def load_config(yaml_file: str) -> Config:
    """
    Load configuration from YAML file
    
    Args:
        yaml_file: Path to YAML configuration file
        
    Returns:
        Config object with all settings
    """
    if not os.path.exists(yaml_file):
        raise FileNotFoundError(f"Configuration file not found: {yaml_file}")
    
    with open(yaml_file, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Extract algorithm-specific config (grpo, ppo, orpo, dpo)
    algorithm_config = {}
    for alg in ['grpo', 'ppo', 'orpo', 'dpo']:
        if alg in config_dict:
            algorithm_config = config_dict.pop(alg)
            break
    
    return Config(
        model=config_dict.get('model', {}),
        data=config_dict.get('data', {}),
        training=config_dict.get('training', {}),
        algorithm=algorithm_config,
        reward=config_dict.get('reward', {}),
        wandb=config_dict.get('wandb', {}),
        output=config_dict.get('output', {}),
        sagemaker=config_dict.get('sagemaker')
    )


def merge_config_with_overrides(config: Config, **overrides) -> Config:
    """
    Merge config with runtime overrides
    
    Args:
        config: Base configuration
        **overrides: Runtime parameter overrides
        
    Returns:
        Updated configuration
    """
    # Simple merge - can be made more sophisticated
    for key, value in overrides.items():
        if hasattr(config, key) and isinstance(getattr(config, key), dict):
            getattr(config, key).update(value if isinstance(value, dict) else {key: value})
    
    return config
