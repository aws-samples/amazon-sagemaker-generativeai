#!/usr/bin/env python3

import os
import json
import logging
import time
from typing import Optional, List, Callable
import boto3
from sagemaker.pytorch import PyTorch
from sagemaker import get_execution_role
from .config_loader import load_config

logger = logging.getLogger(__name__)

class GRPO_RLVR:
    """
    GRPO with Reinforcement Learning from Verifiable Rewards (RLVR)
    
    This class implements GRPO training with verifiable rewards for mathematical reasoning,
    following the RLVR_finetuning.py pattern but integrated with sama_rl.
    """
    
    def __init__(
        self,
        yaml_file: str,
        verifiers: Optional[List] = None,
        instance_type: Optional[str] = None,
        wandb_api_key: Optional[str] = None,
        hf_token: Optional[str] = None,
        training_job_name: Optional[str] = None,
        **overrides
    ):
        """
        Initialize GRPO-RLVR trainer
        
        Args:
            yaml_file: Path to YAML configuration file
            verifiers: List of verifiers for reward validation
            instance_type: SageMaker instance type override
            wandb_api_key: Weights & Biases API key
            hf_token: Hugging Face token for model access
            training_job_name: Existing training job name to load (optional)
            **overrides: Any configuration overrides
        """
        self.config = load_config(yaml_file)
        
        # Handle flat YAML configs (like Qwen2.5-0.5B.yaml)
        if hasattr(self.config, "model_name_or_path"):
            # Convert flat config to nested structure
            if not hasattr(self.config, "model") or self.config.model is None:
                self.config.model = {"name": self.config.model_name_or_path}
            if not hasattr(self.config, "training") or not self.config.training:
                self.config.training = {
                    "learning_rate": getattr(self.config, "learning_rate", 5e-5),
                    "num_epochs": getattr(self.config, "num_train_epochs", 2),
                    "batch_size": getattr(self.config, "per_device_train_batch_size", 64)
                }
        self.verifiers = verifiers or []
        self.training_job_name = training_job_name
        self.estimator = None
        
        # Apply overrides
        for key, value in overrides.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            elif key in ['max_steps', 'learning_rate', 'batch_size']:
                if not hasattr(self.config, 'training'):
                    self.config.training = {}
                self.config.training[key] = value
        
        # Initialize sagemaker config if missing
        # Initialize wandb config if missing
        if not hasattr(self.config, "wandb") or self.config.wandb is None:
            self.config.wandb = {}
        if not hasattr(self.config, "sagemaker") or self.config.sagemaker is None:
            self.config.sagemaker = {}
        # Set instance type
        if instance_type:
            self.config.sagemaker['instance_type'] = instance_type
        elif not self.config.sagemaker.get('instance_type'):
            # Auto-select instance type for RLVR (needs more memory for verifiers)
            recommended_instance = self._get_recommended_instance()
            self.config.sagemaker['instance_type'] = recommended_instance
            logger.info(f"Auto-selected instance type: {recommended_instance} for RLVR training")
            
        if wandb_api_key:
            self.config.wandb['api_key'] = wandb_api_key
            
        if hf_token:
            self.config.model['hf_token'] = hf_token
        
        # Merge any additional overrides
        for key, value in overrides.items():
            if '.' in key:
                # Handle nested keys like 'training.max_steps'
                parts = key.split('.')
                current = self.config
                for part in parts[:-1]:
                    if not hasattr(current, part):
                        setattr(current, part, {})
                    current = getattr(current, part)
                current[parts[-1]] = value
    
    def _get_recommended_instance(self):
        """Get recommended instance type for RLVR training"""
        model_name = self.config.model.get('name', '')
        
        # RLVR needs more memory for verifiers and mathematical reasoning
        if '0.5b' in model_name.lower():
            return "ml.g5.2xlarge"  # 24GB GPU
        elif '1.5b' in model_name.lower() or '1b' in model_name.lower():
            return "ml.g5.4xlarge"  # 24GB GPU x2
        elif '7b' in model_name.lower():
            return "ml.g5.12xlarge"  # 24GB GPU x4
        elif '13b' in model_name.lower():
            return "ml.p4d.24xlarge"  # 40GB GPU x8
        else:
            return "ml.g5.2xlarge"  # Default
    
    def prepare_dataset(self, dataset_name: str = "gsm8k", num_shots: int = 8, test_size: float = 0.1):
        """
        Prepare GSM8K dataset for RLVR training
        
        Args:
            dataset_name: Dataset to use (default: gsm8k)
            num_shots: Number of few-shot examples
            test_size: Fraction for validation split
        """
        from datasets import load_dataset
        
        logger.info(f"Preparing {dataset_name} dataset with {num_shots} shots")
        
        # Load GSM8K dataset (following RLVR pattern)
        if dataset_name == "gsm8k":
            # This would use the GSM8K class from scripts/utils/gsm8k.py
            # For now, use standard dataset loading
            dataset = load_dataset("gsm8k", "main", split="train")
            dataset = dataset.train_test_split(test_size=test_size)
            
            self.train_dataset = dataset['train']
            self.val_dataset = dataset['test']
            
            logger.info(f"Dataset prepared: {len(self.train_dataset)} train, {len(self.val_dataset)} val")
            return dataset
        else:
            raise ValueError(f"Dataset {dataset_name} not supported for RLVR")
    
    def upload_dataset_to_s3(self, bucket_name: Optional[str] = None):
        """Upload prepared dataset to S3"""
        import boto3
        
        if not hasattr(self, 'train_dataset'):
            raise ValueError("Dataset not prepared. Call prepare_dataset() first.")
        
        session = boto3.Session()
        s3_client = session.client('s3')
        
        if not bucket_name:
            import sagemaker
            sagemaker_session = sagemaker.Session()
            bucket_name = sagemaker_session.default_bucket()
        
        # Create S3 paths
        prefix = "datasets/rlvr-training"
        train_s3_path = f"s3://{bucket_name}/{prefix}/train/dataset.json"
        val_s3_path = f"s3://{bucket_name}/{prefix}/val/dataset.json"
        
        # Save and upload datasets
        os.makedirs("./temp/data/train", exist_ok=True)
        os.makedirs("./temp/data/val", exist_ok=True)
        
        self.train_dataset.to_json("./temp/data/train/dataset.json", orient="records")
        self.val_dataset.to_json("./temp/data/val/dataset.json", orient="records")
        
        s3_client.upload_file("./temp/data/train/dataset.json", bucket_name, f"{prefix}/train/dataset.json")
        s3_client.upload_file("./temp/data/val/dataset.json", bucket_name, f"{prefix}/val/dataset.json")
        
        logger.info(f"Dataset uploaded to S3:")
        logger.info(f"Train: {train_s3_path}")
        logger.info(f"Val: {val_s3_path}")
        
        return train_s3_path, val_s3_path
    
    def train(self, wait: bool = True):
        """
        Start GRPO-RLVR training on SageMaker
        
        Args:
            wait: Whether to wait for training completion
        """
        if self.training_job_name:
            logger.info(f"Loading existing training job: {self.training_job_name}")
            return
        
        # Get IAM role
        try:
            role = get_execution_role()
        except:
            role = f"arn:aws:iam::{boto3.client('sts').get_caller_identity()['Account']}:role/SageMakerExecutionRole"
        
        # Create unique training job name
        timestamp = int(time.time())
        model_name_clean = getattr(self.config, "model_name_or_path", "qwen25-05b").replace('/', '').replace('.', '').lower()
        self.training_job_name = f"grpo-rlvr-{model_name_clean}-{timestamp}"
        
        # Upload config to S3
        config_s3_uri = self._upload_config_to_s3()
        
        # Create PyTorch estimator
        self.estimator = PyTorch(
            entry_point="sagemaker_train_grpo_rlvr.py",
            source_dir=os.path.dirname(__file__),
            role=role,
            instance_type=self.config.sagemaker['instance_type'],
            instance_count=self.config.sagemaker.get('instance_count', 1),
            framework_version="2.0.1",
            py_version="py310",
            hyperparameters={
                "config_s3_uri": config_s3_uri
            },
            environment={
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "CUDA_LAUNCH_BLOCKING": "1",
                "WANDB_API_KEY": self.config.wandb.get("api_key", ""),
                "HF_TOKEN": self.config.model.get("hf_token", ""),
                "MLFLOW_EXPERIMENT_NAME": "grpo-rlvr",
                "MLFLOW_TAGS": '{"source.job": "sm-training-jobs", "source.type": "grpo-rlvr", "source.framework": "pytorch"}'
            },
            max_run=self.config.sagemaker.get('max_run', 24*60*60),
            keep_alive_period_in_seconds=self.config.sagemaker.get('keep_alive_period', 1800),
            job_name=self.training_job_name
        )
        
        logger.info(f"Starting GRPO-RLVR training job: {self.training_job_name}")
        logger.info(f"Instance type: {self.config.sagemaker['instance_type']}")
        logger.info(f"Model: {getattr(self.config, "model_name_or_path", "qwen25-05b")}")
        
        # Start training
        self.estimator.fit(wait=wait)
        
        if wait:
            logger.info("GRPO-RLVR training completed!")
        else:
            logger.info("GRPO-RLVR training started in background")
    
    def _upload_config_to_s3(self):
        """Upload configuration to S3"""
        import boto3
        import sagemaker
        
        session = sagemaker.Session()
        bucket = session.default_bucket()
        
        config_key = f"sama-rl-configs/{self.training_job_name}.json"
        config_s3_uri = f"s3://{bucket}/{config_key}"
        
        # Convert config to dict and upload
        config_dict = {
            "config": self.config.__dict__,
            "verifiers": self.verifiers,
            "algorithm": "grpo_rlvr"
        }
        
        s3_client = boto3.client('s3')
        s3_client.put_object(
            Bucket=bucket,
            Key=config_key,
            Body=json.dumps(config_dict, indent=2),
            ContentType='application/json'
        )
        
        logger.info(f"Config uploaded to: {config_s3_uri}")
        return config_s3_uri
    
    def get_model_artifacts(self):
        """Get S3 URI of trained model artifacts"""
        if not self.estimator:
            raise ValueError("No training job found. Call train() first.")
        
        return self.estimator.model_data
    
    def deploy(self, instance_type: str = "ml.g4dn.xlarge", initial_instance_count: int = 1):
        """Deploy trained model to SageMaker endpoint"""
        if not self.estimator:
            raise ValueError("No training job found. Call train() first.")
        
        logger.info(f"Deploying GRPO-RLVR model to {instance_type}")
        
        predictor = self.estimator.deploy(
            initial_instance_count=initial_instance_count,
            instance_type=instance_type
        )
        
        logger.info(f"Model deployed to endpoint: {predictor.endpoint_name}")
        return predictor
