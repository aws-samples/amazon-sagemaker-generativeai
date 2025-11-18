#!/usr/bin/env python3

import os
import json
import logging
import time
from typing import Optional, List, Callable
import boto3
from sagemaker.pytorch import PyTorch
from sagemaker import get_execution_role

from .config_loader import load_config, merge_config_with_overrides

logger = logging.getLogger(__name__)

class PPO:
    """
    PPO (Proximal Policy Optimization) trainer for SageMaker
    """
    
    def __init__(
        self,
        yaml_file: str = None,
        reward_functions: Optional[List[Callable]] = None,
        verifiers: Optional[List] = None,  # Add verifiers parameter
        instance_type: Optional[str] = None,
        wandb_api_key: Optional[str] = None,
        training_job_name: Optional[str] = None,  # New parameter for existing job
        **overrides
    ):
        """
        Initialize PPO trainer
        
        Args:
            yaml_file: Path to YAML configuration file
            reward_functions: List of reward functions
            verifiers: List of verifiers to validate reward functions
            instance_type: SageMaker instance type override
            wandb_api_key: Weights & Biases API key
            training_job_name: Existing training job name to load (optional)
            **overrides: Any configuration overrides
        """
        if training_job_name:
            # Load existing training job
            self._load_existing_training_job(training_job_name)
            return
        
        if not yaml_file:
            raise ValueError("Either yaml_file or training_job_name must be provided")
            
        self.config = load_config(yaml_file)
        self.yaml_file = yaml_file

        print("Config file parameters {}".format(self.config))
        
        # Store reward functions
        self.reward_functions = reward_functions or []
        
        # Apply overrides
        if instance_type:
            if self.config.sagemaker:
                self.config.sagemaker['instance_type'] = instance_type
        else:
            # Auto-select instance type based on model size
            recommended_instance = self._get_recommended_instance_type()
            if self.config.sagemaker:
                self.config.sagemaker['instance_type'] = recommended_instance
            logger.info(f"Auto-selected instance type: {recommended_instance} for model {self.config.model['name']}")
            
        if wandb_api_key:
            self.config.wandb['api_key'] = wandb_api_key
        
        # Merge any additional overrides
        self.config = merge_config_with_overrides(self.config, overrides)
        
        # Apply all overrides to appropriate config sections
        for key, value in overrides.items():
            # Training parameters
            if key in ['max_steps', 'learning_rate', 'per_device_train_batch_size', 
                      'per_device_eval_batch_size', 'gradient_accumulation_steps', 
                      'warmup_steps', 'weight_decay', 'fp16', 'gradient_checkpointing',
                      'dataloader_num_workers', 'eval_strategy', 'eval_steps', 
                      'logging_steps', 'save_strategy', 'save_steps']:
                if hasattr(self.config, 'training') and isinstance(self.config.training, dict):
                    self.config.training[key] = value
                else:
                    # Handle case where training is an object with attributes
                    setattr(self.config.training, key, value)
            
            # Model parameters
            elif key in ['name', 'trust_remote_code']:
                if hasattr(self.config, 'model') and isinstance(self.config.model, dict):
                    self.config.model[key] = value
                else:
                    setattr(self.config.model, key, value)
            
            # Data parameters
            elif key in ['dataset_name', 'train_split', 'test_split']:
                if hasattr(self.config, 'data') and isinstance(self.config.data, dict):
                    self.config.data[key] = value
                else:
                    setattr(self.config.data, key, value)
        
        # Debug: Print all config parameters
        logger.info("=== PPO Configuration Debug ===")
        logger.info(f"YAML file: {yaml_file}")
        logger.info(f"Model config: {self.config.model}")
        logger.info(f"Data config: {self.config.data}")
        logger.info(f"Training config: {self.config.training}")
        logger.info(f"SageMaker config: {self.config.sagemaker}")
        logger.info(f"Reward functions: {len(self.reward_functions)} provided")
        logger.info(f"Overrides applied: {overrides}")
        logger.info("=== End Configuration Debug ===")
        
        # Validate reward functions if verifiers provided
        if verifiers and self.reward_functions:
            for verifier in verifiers:
                for reward_func in self.reward_functions:
                    verifier.verify(reward_func)
        
        # Setup SageMaker training job
        self._setup_sagemaker_training()
    
    def _get_recommended_instance_type(self):
        """Get recommended instance type based on model size"""
        model_name = self.config.model.get('name', '').lower()
        
        if '0.5b' in model_name or '500m' in model_name:
            return 'ml.g4dn.xlarge'
        elif '1.5b' in model_name or '1b' in model_name:
            return 'ml.g4dn.2xlarge'
        elif '7b' in model_name:
            return 'ml.g5.2xlarge'
        elif '13b' in model_name:
            return 'ml.g5.4xlarge'
        else:
            return 'ml.g4dn.xlarge'  # Default for small models
    
    def _load_existing_training_job(self, training_job_name):
        """Load configuration from existing training job"""
        self.training_job_name = training_job_name
        logger.info(f"Loading existing PPO training job: {training_job_name}")
        # Implementation would load job details from SageMaker
    
    def _setup_sagemaker_training(self):
        """Setup SageMaker training job"""
        # Get IAM role
        try:
            role = get_execution_role()
        except:
            role = self.config.sagemaker.get('role')
            if not role:
                raise ValueError("Could not determine IAM role for SageMaker")
        
        # Generate unique training job name
        model_name_clean = self.config.model['name'].replace('/', '').replace('.', '').replace('-', '').lower()
        import time
        self.training_job_name = f"sama-ppo-{model_name_clean}-{int(time.time())}"
        
        logger.info(f"Setting up PPO training job: {self.training_job_name}")
        
        # Create PyTorch estimator
        self.estimator = PyTorch(
            entry_point="sagemaker_train_ppo.py",
            source_dir=os.path.dirname(__file__),
            role=role,
            instance_type=self.config.sagemaker.get('instance_type', 'ml.g4dn.xlarge'),
            instance_count=1,
            framework_version='2.0.1',
            py_version='py310',
            hyperparameters={},
            environment={
                "WANDB_API_KEY": self.config.wandb.get("api_key", ""),
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "CUDA_LAUNCH_BLOCKING": "1"
            },
            max_run=24*60*60,  # 24 hours
            keep_alive_period_in_seconds=1800,  # 30 minutes
            volume_size=100
        )
        
        logger.info(f"PPO estimator created with instance type: {self.config.sagemaker.get('instance_type')}")
    
    def _prepare_reward_function_code(self):
        """Prepare reward function code for SageMaker"""
        if not self.reward_functions:
            return None
        
        # For now, return a simple reward function
        # In a full implementation, this would serialize the actual reward functions
        reward_code = '''
def reward_function(completions, **kwargs):
    """Simple length-based reward function"""
    import torch
    rewards = []
    for completion in completions:
        # Simple reward based on length
        reward = len(completion) / 100.0  # Normalize by length
        rewards.append(reward)
    return rewards
'''
        return reward_code
    
    def train(self):
        """Start PPO training on SageMaker"""
        if hasattr(self, 'training_job_name') and not hasattr(self, 'estimator'):
            logger.info(f"Training job {self.training_job_name} already exists")
            return self
        
        logger.info("Starting PPO training...")
        logger.info(f"Job name: {self.training_job_name}")
        logger.info(f"Model: {self.config.model['name']}")
        
        # Create config file for SageMaker
        s3 = boto3.client('s3')
        bucket = self.estimator.sagemaker_session.default_bucket()
        
        config_data = {
            'config': {**self.config.__dict__, 'algorithm': 'ppo'},
            'reward_function': self._prepare_reward_function_code(),
            'yaml_file': getattr(self, 'yaml_file', None)
        }
        
        # Upload config to S3
        config_key = f"sama-rl-configs/{self.training_job_name or 'ppo-config'}.json"
        s3.put_object(
            Bucket=bucket,
            Key=config_key,
            Body=json.dumps(config_data, indent=2, default=str)
        )
        
        # Set hyperparameters to point to config
        self.estimator.set_hyperparameters(
            config_s3_uri=f"s3://{bucket}/{config_key}"
        )
        
        # Start training job
        self.estimator.fit(job_name=self.training_job_name)
        
        logger.info("PPO training job completed!")
        return self
    
    def get_model_artifacts(self):
        """Get the S3 URI of trained model artifacts"""
        if hasattr(self, 'estimator') and self.estimator.latest_training_job:
            return self.estimator.latest_training_job.describe()['ModelArtifacts']['S3ModelArtifacts']
        return None
    
    def deploy(self, instance_type='ml.g4dn.xlarge', initial_instance_count=1):
        """Deploy the trained model to a SageMaker endpoint"""
        if not hasattr(self, 'estimator'):
            raise ValueError("No training job found. Run train() first.")
        
        logger.info(f"Deploying PPO model to {instance_type}")
        
        predictor = self.estimator.deploy(
            initial_instance_count=initial_instance_count,
            instance_type=instance_type
        )
        
        logger.info(f"Model deployed to endpoint: {predictor.endpoint_name}")
        return predictor
