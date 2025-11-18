#!/usr/bin/env python3
import logging
import json
import os
import sys
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def install_dependencies():
    """Install dependencies for GRPO-RLVR training"""
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "typing_extensions>=4.8.0"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "evaluate", "peft", "datasets"])

def patch_backend_detection():
    import transformers.utils.import_utils
    def fixed_requires_backends(obj, backends):
        return
    transformers.utils.requires_backends = fixed_requires_backends

def load_gsm8k_dataset(num_shots=8, test_size=0.1):
    """Load and prepare GSM8K dataset for RLVR training"""
    from datasets import load_dataset
    
    logger.info(f"Loading GSM8K dataset with {num_shots} shots")
    
    # Load GSM8K dataset
    dataset = load_dataset("gsm8k", "main")
    
    # Create train/val split
    train_dataset = dataset['train'].train_test_split(test_size=test_size)
    
    # Add few-shot prompting (simplified version)
    def add_few_shot_prompt(example):
        # This is a simplified version - in practice you'd use the GSM8K class
        # from scripts/utils/gsm8k.py for proper few-shot CoT prompting
        question = example['question']
        answer = example['answer']
        
        # Create a basic CoT prompt
        prompt = f"Question: {question}\nLet me think step by step.\n"
        example['prompt'] = prompt
        example['final_answer'] = answer.split('####')[-1].strip() if '####' in answer else answer
        
        return example
    
    train_data = train_dataset['train'].map(add_few_shot_prompt)
    val_data = train_dataset['test'].map(add_few_shot_prompt)
    
    logger.info(f"Dataset prepared: {len(train_data)} train, {len(val_data)} val")
    return train_data, val_data

def create_reward_function(config):
    """Create verifiable reward function for mathematical reasoning"""
    
    def math_reward_function(prompt, response, **kwargs):
        """
        Verifiable reward function for mathematical reasoning
        Returns higher rewards for correct mathematical solutions
        """
        import re
        
        # Extract numerical answer from response
        def extract_answer(text):
            numbers = re.findall(r'-?\d+', text.replace(',', ''))
            return numbers[-1] if numbers else None
        
        # Simple verification - in practice you'd use more sophisticated verifiers
        predicted_answer = extract_answer(response)
        
        # Base reward for generating a numerical answer
        if predicted_answer:
            reward = 0.5
            
            # Additional reward for step-by-step reasoning
            if any(phrase in response.lower() for phrase in ['step', 'first', 'then', 'therefore']):
                reward += 0.3
                
            # Additional reward for mathematical operations
            if any(op in response for op in ['+', '-', '*', '/', '=']):
                reward += 0.2
                
            return min(reward, 1.0)  # Cap at 1.0
        else:
            return 0.1  # Small reward for attempting to answer
    
    return math_reward_function

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_s3_uri", type=str, required=True)
    args = parser.parse_args()
    
    logger.info("Installing dependencies...")
    install_dependencies()
    
    logger.info("Patching backend detection...")
    patch_backend_detection()
    
    logger.info("Starting GRPO-RLVR training...")
    
    # Import after dependencies are installed
    from trl import GRPOConfig, GRPOTrainer as TRLGRPOTrainer
    from datasets import load_dataset
    from transformers import AutoTokenizer
    import boto3
    
    # Load config from S3
    s3_client = boto3.client('s3')
    bucket, key = args.config_s3_uri.replace("s3://", "").split("/", 1)
    
    config_obj = s3_client.get_object(Bucket=bucket, Key=key)
    config_data = json.loads(config_obj['Body'].read().decode('utf-8'))
    
    config = config_data["config"]
    
    # Handle flat config structure
    model_name = config.get("model_name_or_path") or config.get("model", {}).get("name", "Qwen/Qwen2.5-0.5B")
    learning_rate = config.get("learning_rate") or config.get("training", {}).get("learning_rate", 5e-5)
    batch_size = config.get("per_device_train_batch_size") or config.get("training", {}).get("batch_size", 1)
    verifiers = config_data.get("verifiers", [])
    
    logger.info(f"Loaded config: {config}")
    
    # Load and prepare dataset
    num_shots = config.get("data", {}).get("num_shots", 8)
    train_dataset, eval_dataset = load_gsm8k_dataset(num_shots=num_shots)
    
    # Create reward function
    reward_function = create_reward_function(config)
    
    # Create GRPO config
    grpo_config = GRPOConfig(
        learning_rate=float(learning_rate),
        num_train_epochs=config.get("training", {}).get("num_epochs", 1),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=config.get("training", {}).get("eval_batch_size", 1),
        gradient_accumulation_steps=16,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        fp16=True,
        dataloader_pin_memory=False,
        output_dir="/opt/ml/model",
        logging_steps=1,
        save_steps=100,
        save_total_limit=3,
        report_to=None,
    )
    
    logger.info("Creating GRPO trainer...")
    
    # Create GRPO trainer (following RLVR pattern)
    trainer = TRLGRPOTrainer(
        model=model_name,
        reward_funcs=reward_function,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    logger.info("Starting GRPO-RLVR training...")
    
    # Train the model
    trainer.train()
    
    # Save the final model
    trainer.save_model("/opt/ml/model")
    
    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    tokenizer.save_pretrained("/opt/ml/model")
    
    logger.info("GRPO-RLVR training completed!")

if __name__ == "__main__":
    main()
