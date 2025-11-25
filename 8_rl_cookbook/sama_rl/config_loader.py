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
    """Load configuration from YAML file, handling both nested and flat structures"""
    with open(yaml_file, "r") as f:
        raw_config = yaml.safe_load(f)
    
    # Check if config is flat (has model_name_or_path) or nested (has model.name)
    if "model_name_or_path" in raw_config:
        # Flat config - convert to nested
        config_dict = {
            "model": {"name": raw_config.get("model_name_or_path", "")},
            "data": {"dataset_name": raw_config.get("train_dataset_id_or_path", "")},
            "training": {
                "learning_rate": raw_config.get("learning_rate", 5e-5),
                "batch_size": raw_config.get("per_device_train_batch_size", 1),
                "num_epochs": raw_config.get("num_train_epochs", 1)
            },
            "algorithm": {},
            "reward": {},
            "wandb": {},
            "output": {"dir": raw_config.get("output_dir", "/opt/ml/model")},
            "sagemaker": {}
        }
        
        # Create config object
        config = Config(**{key: config_dict.get(key, {}) for key in ["model", "data", "training", "algorithm", "reward", "wandb", "output", "sagemaker"]})
        
        # Store original flat config attributes for training script access
        for key, value in raw_config.items():
            setattr(config, key, value)
            
        return config
    else:
        # Nested config - use as is
        return Config(**{key: raw_config.get(key, {}) for key in ["model", "data", "training", "algorithm", "reward", "wandb", "output", "sagemaker"]})


def merge_config_with_overrides(config: Config, overrides: Dict[str, Any]) -> Config:
    """Merge configuration with override values"""
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config
