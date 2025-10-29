"""
GRPO - Group Relative Policy Optimization
"""
import os
import logging
import yaml
from typing import List, Callable, Optional, Dict, Any
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker import get_execution_role

from .config_loader import load_config, merge_config_with_overrides

logger = logging.getLogger(__name__)


class GRPO:
    """
    GRPO - Group Relative Policy Optimization
    
    Uses SageMaker training jobs under the hood
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
        Initialize GRPO trainer
        
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
        self.config = merge_config_with_overrides(self.config, **overrides)
        
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
            
            # GRPO parameters
            elif key in ['num_generations', 'max_completion_length', 'max_prompt_length',
                        'temperature', 'top_p', 'top_k', 'repetition_penalty', 'do_sample', 'beta']:
                if not hasattr(self.config, 'algorithm'):
                    self.config.algorithm = {}
                self.config.algorithm[key] = value
            
            # Model parameters
            elif key in ['model_name']:
                if key == 'model_name':
                    self.config.model['name'] = value
            
            # Data parameters
            elif key in ['dataset_name', 'train_split', 'test_split']:
                self.config.data[key] = value
            
            # Reward parameters
            elif key in ['target_length', 'reward_type']:
                self.config.reward[key] = value
            
            # W&B parameters
            elif key in ['wandb_project', 'wandb_run_name']:
                if key == 'wandb_project':
                    self.config.wandb['project'] = value
                elif key == 'wandb_run_name':
                    self.config.wandb['run_name'] = value
            
            # Output parameters
            elif key in ['output_dir']:
                self.config.output['dir'] = value
        
        # Debug print to verify override worked
        logger.info(f"Config after overrides - max_steps: {getattr(self.config.training, 'max_steps', 'NOT_FOUND')}")
        
        # SageMaker components
        self.estimator = None
        self.training_job_name = None
    
    def _load_existing_training_job(self, training_job_name: str):
        """Load an existing training job for deployment"""
        import boto3
        
        self.training_job_name = training_job_name
        
        # Get training job details
        sagemaker_client = boto3.client('sagemaker')
        try:
            response = sagemaker_client.describe_training_job(
                TrainingJobName=training_job_name
            )
            
            # Create a minimal config for deployment
            self.config = type('Config', (), {
                'model': {'name': 'Qwen/Qwen2-0.5B-Instruct'},  # Default, can be overridden
                'sagemaker': {
                    'instance_type': response.get('ResourceConfig', {}).get('InstanceType', 'ml.g4dn.2xlarge')
                }
            })()
            
            # Create estimator from existing job
            from sagemaker.pytorch import PyTorch
            self.estimator = PyTorch.attach(training_job_name)
            
            logger.info(f"Loaded existing training job: {training_job_name}")
            
        except Exception as e:
            raise ValueError(f"Could not load training job {training_job_name}: {e}")
        
        # Set components
        self.reward_functions = []
        self.verifiers = []
        
    def _get_recommended_instance_type(self) -> str:
        """Get recommended instance type based on model size"""
        model_name = self.config.model['name'].lower()
        
        # Small models (< 1B parameters)
        if any(size in model_name for size in ['0.5b', '125m', '350m', '774m']):
            return "ml.g4dn.xlarge"  # $1.20/hour
        
        # Medium models (1B - 3B parameters)  
        elif any(size in model_name for size in ['1b', '1.5b', '2b', '3b']):
            return "ml.g4dn.2xlarge"  # $2.40/hour
        
        # Large models (7B - 13B parameters)
        elif any(size in model_name for size in ['7b', '8b', '9b', '11b', '13b']):
            return "ml.g4dn.12xlarge"  # $9.60/hour
        
        # Very large models (20B+ parameters)
        elif any(size in model_name for size in ['20b', '30b', '32b', '70b']):
            return "ml.p4d.24xlarge"  # $32/hour
        
        # Default for unknown sizes
        else:
            return "ml.g4dn.2xlarge"  # Safe default
        
    def _prepare_reward_function_code(self) -> Optional[str]:
        """Convert reward function to simple code string"""
        if not self.reward_functions:
            return None
        
        # Always use simple reward function for now
        simple_reward_code = '''
def reward_function(completions, **kwargs):
    """Simple length-based reward function"""
    target_length = kwargs.get('target_length', 400)
    tokenizer = kwargs.get('tokenizer')
    rewards = []
    
    for completion in completions:
        if tokenizer:
            num_tokens = len(tokenizer.encode(completion, add_special_tokens=False))
        else:
            num_tokens = len(completion.split())
        
        distance = abs(num_tokens - target_length)
        reward = -(distance ** 2) / 1000
        rewards.append(reward)
    
    return rewards
'''
        return simple_reward_code
    
    def _extract_base_function_code(self, func):
        """Extract source code for a regular function"""
        if hasattr(func, '__name__'):
            import inspect
            import textwrap
            try:
                source = inspect.getsource(func)
                source = textwrap.dedent(source)
                # Rename to reward_function
                if 'def reward_function(' not in source:
                    lines = source.split('\n')
                    for i, line in enumerate(lines):
                        if line.strip().startswith('def ') and '(' in line:
                            func_def = line.split('def ')[1]
                            lines[i] = line.split('def ')[0] + 'def reward_function(' + func_def.split('(', 1)[1]
                            break
                    source = '\n'.join(lines)
                return source
            except Exception as e:
                logger.warning(f"Could not extract source: {e}")
        return None
    
    def _create_sagemaker_estimator(self):
        """Create SageMaker PyTorch estimator"""
        try:
            role = get_execution_role()
        except:
            # Fallback role - user should set this properly
            role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
        
        sagemaker_config = self.config.sagemaker or {}
        
        # Upload config to S3 instead of using environment variables
        import boto3
        import json
        import tempfile
        
        s3 = boto3.client('s3')
        sagemaker_session = sagemaker.Session()
        bucket = sagemaker_session.default_bucket()
        
        # Create config file
        config_data = {
            'config': self.config.__dict__,
            'reward_function': self._prepare_reward_function_code()
        }
        
        # Upload to S3
        config_key = f"sama-rl-configs/{self.training_job_name or 'config'}.json"
        s3.put_object(
            Bucket=bucket,
            Key=config_key,
            Body=json.dumps(config_data)
        )
        
        # Small environment variables only
        environment = {
            "CONFIG_S3_BUCKET": bucket,
            "CONFIG_S3_KEY": config_key,
            "WANDB_API_KEY": self.config.wandb.get('api_key', '')[:500]  # Truncate if needed
        }
        
        self.estimator = PyTorch(
            entry_point="sagemaker_train.py",
            source_dir=os.path.dirname(__file__),
            role=role,
            instance_type=sagemaker_config.get('instance_type', 'ml.g4dn.2xlarge'),
            instance_count=sagemaker_config.get('instance_count', 1),
            framework_version="2.0.1",
            py_version="py310",
            environment=environment,
            max_run=sagemaker_config.get('max_run', 3600),
            keep_alive_period_in_seconds=sagemaker_config.get('keep_alive_period', 1800),
        )
    
    def verify_reward_functions(self, test_completions: List[str] = None) -> bool:
        """
        Verify reward functions using RLVR verifiers before training
        
        Args:
            test_completions: Test completions for verification
            
        Returns:
            True if all verifications pass, False otherwise
        """
        if not self.verifiers or not self.reward_functions:
            logger.info("No verifiers or reward functions to verify")
            return True
        
        # Default test completions if none provided
        if not test_completions:
            test_completions = [
                "This is a short response.",
                "This is a much longer response with detailed information and helpful explanations.",
                "This response is great and excellent, providing wonderful insights.",
                "This is a bad and terrible response that is awful.",
                "The algorithm implementation uses optimization for better performance."
            ]
        
        logger.info("Verifying reward functions with RLVR...")
        
        from .verifiers import verify_reward_function
        
        all_passed = True
        for i, reward_func in enumerate(self.reward_functions):
            logger.info(f"Verifying reward function {i+1}/{len(self.reward_functions)}")
            
            result = verify_reward_function(
                reward_function=reward_func,
                test_completions=test_completions,
                verifiers=self.verifiers
            )
            
            if not result['overall_passed']:
                logger.error(f"Reward function verification failed: {result}")
                all_passed = False
            else:
                logger.info(f"Reward function verified successfully (accuracy: {result['overall_accuracy']:.2f})")
        
        return all_passed
    
    def train(self):
        """Start GRPO training using SageMaker"""
        logger.info("Starting GRPO training on SageMaker...")
        
        # Verify reward functions first
        if not self.verify_reward_functions():
            raise ValueError("Reward function verification failed. Training aborted.")
        
        # Generate unique job name with model name
        import time
        timestamp = int(time.time())
        model_name = self.config.model['name'].split('/')[-1].lower().replace('-', '').replace('.', '')
        self.training_job_name = f"sama-grpo-{model_name}-{timestamp}"
        
        # Create SageMaker estimator (needs job name for S3 key)
        self._create_sagemaker_estimator()
        
        logger.info(f"Launching SageMaker training job: {self.training_job_name}")
        logger.info(f"Instance type: {self.estimator.instance_type}")
        logger.info(f"Model: {self.config.model['name']}")
        
        # Start training job
        self.estimator.fit(job_name=self.training_job_name)
        
        logger.info("Training job completed!")
        return self
    
    def get_model_artifacts(self) -> str:
        """Get S3 path to trained model artifacts"""
        if not self.estimator:
            raise ValueError("No training job has been started")
        return self.estimator.model_data
    
    def get_training_job_status(self) -> str:
        """Get current training job status"""
        if not self.training_job_name:
            return "Not started"
        
        import boto3
        sagemaker_client = boto3.client('sagemaker')
        response = sagemaker_client.describe_training_job(
            TrainingJobName=self.training_job_name
        )
        return response['TrainingJobStatus']
    
    def get_logs(self):
        """Get training job logs"""
        if not self.estimator:
            raise ValueError("No training job has been started")
        
        sagemaker_session = sagemaker.Session()
        sagemaker_session.logs_for_job(self.training_job_name, wait=False)
    
    def get_sagemaker_config(self) -> Dict[str, Any]:
        """Get SageMaker configuration for deployment"""
        if not self.config.sagemaker:
            return {}
        
        return {
            'instance_type': self.config.sagemaker.get('instance_type', 'ml.g4dn.2xlarge'),
            'instance_count': self.config.sagemaker.get('instance_count', 1),
            'max_run': self.config.sagemaker.get('max_run', 3600),
        }
    
    def deploy(self, instance_type: str = None, instance_count: int = 1) -> str:
        """
        Deploy the trained model to a SageMaker endpoint
        
        Args:
            instance_type: Instance type for endpoint (auto-selected if None)
            instance_count: Number of instances
            
        Returns:
            Endpoint name
        """
        if not self.estimator:
            raise ValueError("No trained model found. Either run train() first or load existing training job.")
        
        # Auto-select GPU instance type for LLM deployment
        if not instance_type:
            model_name = getattr(self.config.model, 'name', 'unknown').lower()
            if any(size in model_name for size in ['0.5b', '125m', '350m']):
                instance_type = "ml.g5.xlarge"  # Small models - GPU needed for LLM container
            elif any(size in model_name for size in ['1b', '1.5b', '2b', '3b']):
                instance_type = "ml.g5.2xlarge"  # Medium models  
            elif any(size in model_name for size in ['7b', '8b', '13b']):
                instance_type = "ml.g5.12xlarge"  # Large models
            else:
                instance_type = "ml.g5.2xlarge"  # Default GPU instance
            
            logger.info(f"Auto-selected GPU instance type: {instance_type}")
        
        from .deployment import ModelDeployer
        
        # Get model artifacts from training job
        model_s3_path = self.get_model_artifacts()
        base_model_name = getattr(self.config.model, 'name', 'Qwen/Qwen2-0.5B-Instruct')
        
        # Deploy using ModelDeployer
        deployer = ModelDeployer(model_s3_path, base_model_name)
        endpoint_name = deployer.deploy_endpoint(instance_type, instance_count)
        
        logger.info(f"Model deployed to endpoint: {endpoint_name}")
        return endpoint_name
    
    def save_config(self, path: str):
        """Save current configuration to file"""
        with open(path, 'w') as f:
            yaml.dump(self.config.__dict__, f, default_flow_style=False)
