#!/usr/bin/env python3
import logging
import json
import os
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def install_dependencies():
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "typing_extensions>=4.8.0"])
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    ])

def patch_backend_detection():
    import transformers.utils.import_utils
    def fixed_requires_backends(obj, backends):
        return
    transformers.utils.requires_backends = fixed_requires_backends

def main():
    logger.info("Installing dependencies...")
    install_dependencies()
    
    logger.info("Patching backend detection...")
    patch_backend_detection()
    
    logger.info("Starting GRPO training...")
    
    # Import all required modules
    from trl import GRPOConfig, GRPOTrainer as TRLGRPOTrainer
    from datasets import load_dataset
    from transformers import AutoTokenizer
    
    # Default config as fallback
    default_config = {
        "model": {"name": "Qwen/Qwen2.5-0.5B-Instruct"},
        "data": {"dataset_name": "trl-lib/tldr", "train_split": "train[:100]", "test_split": "test[:20]"},
        "training": {"max_steps": 10, "learning_rate": 1e-5}
    }
    
    # Try to load config from SageMaker
    config_data = None
    config_paths = [
        "/opt/ml/input/config/hyperparameters.json",
        "/opt/ml/input/data/config/config.json"
    ]
    
    for config_path in config_paths:
        try:
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config_data = json.load(f)
                logger.info(f"Loaded config from {config_path}")
                break
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")
    
    if not config_data:
        logger.warning("No config file found, using default config")
        config = default_config
        reward_function_code = None
    else:
        # Try to load from yaml_file first
        yaml_file = config_data.get("yaml_file")
        if yaml_file and os.path.exists(yaml_file):
            try:
                import yaml
                with open(yaml_file, "r") as f:
                    config = yaml.safe_load(f)
                logger.info(f"Loaded recipe from {yaml_file}")
            except Exception as e:
                logger.warning(f"Could not load yaml file {yaml_file}: {e}, using fallback")
                config = config_data.get("config", default_config)
        else:
            config = config_data.get("config", default_config)
        
        reward_function_code = config_data.get("reward_function")
    
    # Ensure config has required keys with defaults
    if "model" not in config:
        config["model"] = default_config["model"]
    if "data" not in config:
        config["data"] = default_config["data"]
    if "training" not in config:
        config["training"] = default_config["training"]
    
    logger.info(f"Using config: {config}")
    
    train_dataset = load_dataset(config["data"]["dataset_name"], split=config["data"]["train_split"])
    eval_dataset = load_dataset(config["data"]["dataset_name"], split=config["data"]["test_split"])
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"], trust_remote_code=True)
    
    # Load reward function from config_data or use default
    if reward_function_code:
        try:
            exec_globals = {"tokenizer": tokenizer}
            exec(reward_function_code, exec_globals)
            reward_function = exec_globals["reward_function"]
            logger.info("Using custom reward function from config")
        except Exception as e:
            logger.warning(f"Could not load custom reward function: {e}, using default")
            reward_function = None
    
    if not reward_function_code or 'reward_function' not in locals():
        # Default reward function
        target_length = config.get("reward", {}).get("target_length", 100)
        def reward_function(completions, **kwargs):
            num_tokens = [len(tokenizer.encode(c, add_special_tokens=False)) for c in completions]
            rewards = [-(l - target_length) ** 2 / 1000 for l in num_tokens]
            return rewards
        logger.info("Using default length-based reward function")
    
    grpo_config = GRPOConfig(
        output_dir="/opt/ml/model",
        max_steps=int(config.get("training", {}).get("max_steps", 10)),
        learning_rate=float(config.get("training", {}).get("learning_rate", 1e-5)),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=int(config.get("training", {}).get("gradient_accumulation_steps", 8)),
        fp16=bool(config.get("training", {}).get("fp16", True)),
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        num_generations=4,
        generation_batch_size=4,
        report_to="none"
    )
    
    trainer = TRLGRPOTrainer(
        model=config["model"]["name"],
        reward_funcs=reward_function,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    trainer.train()
    trainer.save_model("/opt/ml/model")
    logger.info("GRPO training completed!")

if __name__ == "__main__":
    main()
