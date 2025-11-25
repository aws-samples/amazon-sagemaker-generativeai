#!/usr/bin/env python3
"""
SageMaker training script for SAMA RL
This script runs inside the SageMaker training container
"""
import os
import subprocess
import sys
import logging
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import GRPOTrainer as TRLGRPOTrainer, GRPOConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def install_dependencies():
    """Install flash-attention and other dependencies"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "typing_extensions>=4.8.0"])
    flash_attn_url = "https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    subprocess.check_call([sys.executable, "-m", "pip", "install", flash_attn_url])

def patch_backend_detection():
    """Patch transformers backend detection"""
    import torch
    import transformers.utils.import_utils as import_utils
    
    original_requires_backends = import_utils.requires_backends
    def fixed_requires_backends(obj, backends):
        filtered_backends = [b for b in backends if b != "torch"]
        if filtered_backends:
            return original_requires_backends(obj, filtered_backends)
        return None
    
    import_utils.requires_backends = fixed_requires_backends
    import transformers.utils
    transformers.utils.requires_backends = fixed_requires_backends

def main():
    logger.info("Installing dependencies...")
    install_dependencies()
    
    logger.info("Patching backend detection...")
    patch_backend_detection()
    
    # Get config from S3
    import boto3
    import json
    
    bucket = os.environ.get("CONFIG_S3_BUCKET")
    key = os.environ.get("CONFIG_S3_KEY")
    wandb_key = os.environ.get("WANDB_API_KEY")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    
    # Download config from S3
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket, Key=key)
    config_data = json.loads(response['Body'].read())
    
    config = config_data['config']
    reward_function_code = config_data['reward_function']
    
    # Setup wandb (optional)
    if wandb_key:
        try:
            import wandb
            wandb.login(key=wandb_key)
            wandb.init(
                project=config['wandb']['project'],
                name=config['wandb']['run_name']
            )
            logger.info("W&B logging enabled")
        except Exception as e:
            logger.warning(f"W&B setup failed: {e}. Continuing without W&B logging.")
    else:
        logger.info("W&B logging disabled (no API key provided)")
    
    # Load data and tokenizer
    logger.info("Loading dataset and tokenizer...")
    train_dataset = load_dataset(config['data']['dataset_name'], split=config['data']['train_split'])
    eval_dataset = load_dataset(config['data']['dataset_name'], split=config['data']['test_split'])
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'], trust_remote_code=True)
    
    # Setup reward function
    if reward_function_code:
        exec_globals = {
            'tokenizer': tokenizer, 
            'target_length': config['reward']['target_length'],
            'List': list,  # Add List type for compatibility
            'Dict': dict,  # Add Dict type for compatibility
            'Any': object  # Add Any type for compatibility
        }
        # Import typing if needed
        try:
            from typing import List, Dict, Any
            exec_globals.update({'List': List, 'Dict': Dict, 'Any': Any})
        except ImportError:
            pass
            
        exec(reward_function_code, exec_globals)
        reward_function = exec_globals['reward_function']
    else:
        target_length = config['reward']['target_length']
        def reward_function(completions, **kwargs):
            num_tokens = [len(tokenizer.encode(c, add_special_tokens=False)) for c in completions]
            rewards = [-(l - target_length) ** 2 / 1000 for l in num_tokens]
            return rewards
    
    # Create GRPO config with memory optimization
    num_generations = int(config.get('grpo', {}).get('num_generations', 2))
    # Auto-calculate eval batch size to be divisible by num_generations
    per_device_eval_batch_size = max(num_generations, 2)
    
    grpo_config = GRPOConfig(
        output_dir=model_dir,
        max_steps=int(config['training'].get('max_steps', 500)),
        learning_rate=float(config['training'].get('learning_rate', 1e-5)),
        per_device_train_batch_size=1,  # Reduce batch size for memory
        per_device_eval_batch_size=per_device_eval_batch_size,  # Auto-calculated
        gradient_accumulation_steps=int(config['training'].get('gradient_accumulation_steps', 8)),
        num_generations=num_generations,
        max_completion_length=int(config.get('grpo', {}).get('max_completion_length', 512)),
        fp16=bool(config['training'].get('fp16', True)),
        gradient_checkpointing=bool(config['training'].get('gradient_checkpointing', True)),
        dataloader_num_workers=0,  # Reduce workers to save memory
        logging_steps=10,
        eval_strategy='steps',
        eval_steps=50,
        do_eval=True,
        save_strategy='steps',
        save_steps=100,
    )
    
    # Create trainer
    trainer = TRLGRPOTrainer(
        model=config['model']['name'],
        reward_funcs=reward_function,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # Train
    trainer.train()
    trainer.save_model(model_dir)
    
    logger.info(f"Training completed! Model saved to {model_dir}")

if __name__ == "__main__":
    main()
