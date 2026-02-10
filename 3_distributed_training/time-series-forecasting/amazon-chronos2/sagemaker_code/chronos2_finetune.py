"""
Chronos-2 Fine-Tuning script for SageMaker with Accelerate support.

This script supports:
- Full fine-tuning and LoRA training
- Time series data with covariates
- MLflow logging
- Distributed training with Accelerate
"""

import logging
import os
import json
import mlflow
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any

import numpy as np
import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from chronos import BaseChronosPipeline, Chronos2Pipeline
from transformers import set_seed, HfArgumentParser, TrainerCallback, TrainerState, TrainerControl
from transformers.trainer_callback import TrainingArguments


# Configure logging
def setup_logging() -> logging.Logger:
    """Set up logging configuration for the training script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


logger = setup_logging()


@dataclass
class Chronos2ScriptArguments:
    """Arguments for Chronos-2 fine-tuning."""

    # Model arguments
    model_name_or_path: str = field(
        default="amazon/chronos-2",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    
    device_map: str = field(
        default="cuda",
        metadata={"help": "Device map for model loading (cuda, cpu, auto)"}
    )

    # Dataset arguments
    dataset_path: str = field(
        default="/opt/ml/input/data/training/",
        metadata={"help": "Path to training dataset directory"}
    )
    
    dataset_file: str = field(
        default="train.jsonl",
        metadata={"help": "Name of the dataset file (JSONL format)"}
    )
    
    validation_split: float = field(
        default=0.1,
        metadata={"help": "Fraction of data to use for validation (0 to disable)"}
    )

    # Training arguments
    prediction_length: int = field(
        default=24,
        metadata={"help": "Forecast horizon for fine-tuning"}
    )
    
    context_length: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum context length (None uses model default)"}
    )
    
    finetune_mode: str = field(
        default="full",
        metadata={"help": "Fine-tuning mode: 'full' or 'lora'"}
    )
    
    learning_rate: float = field(
        default=1e-5,
        metadata={"help": "Learning rate for optimizer"}
    )
    
    num_steps: int = field(
        default=1000,
        metadata={"help": "Number of training steps"}
    )
    
    batch_size: int = field(
        default=32,
        metadata={"help": "Batch size for training"}
    )
    
    min_past: Optional[int] = field(
        default=None,
        metadata={"help": "Minimum past context length (None sets to prediction_length)"}
    )
    
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for reproducibility"}
    )

    # LoRA arguments
    lora_r: int = field(
        default=8,
        metadata={"help": "LoRA attention dimension"}
    )
    
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha parameter"}
    )
    
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout probability"}
    )

    # Output arguments
    output_dir: str = field(
        default="/opt/ml/output/amazon/chronos-2/",
        metadata={"help": "Directory for saving model outputs"}
    )
    
    finetuned_ckpt_name: str = field(
        default="finetuned-ckpt",
        metadata={"help": "Name of the finetuned checkpoint directory"}
    )

    # MLflow arguments
    mlflow_tracking_uri: str = field(
        default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
        metadata={"help": "MLflow tracking server URI"}
    )
    
    mlflow_experiment_name: str = field(
        default=os.getenv("MLFLOW_EXPERIMENT_NAME", "chronos2-finetuning"),
        metadata={"help": "MLflow experiment name"}
    )
    
    run_name: str = field(
        default="chronos2-finetune",
        metadata={"help": "MLflow run name"}
    )
    
    logging_steps: int = field(
        default=100,
        metadata={"help": "Log metrics every N steps"}
    )


def load_training_data(args: Chronos2ScriptArguments) -> tuple:
    """
    Load and prepare training data from JSONL file.
    
    Expected dataset structure:
    {
        "target": [...],  # 1D array or 2D array for multivariate
        "past_covariates": {"cov1": [...], "cov2": [...]},  # optional
        "future_covariates": {"cov1": [...], "cov2": [...]}  # optional
    }
    
    Returns:
        Tuple of (train_inputs, validation_inputs)
    """
    dataset_file = os.path.join(args.dataset_path, args.dataset_file)
    
    if not os.path.exists(dataset_file):
        # Try to find any jsonl file in the directory
        jsonl_files = [f for f in os.listdir(args.dataset_path) if f.endswith('.jsonl')]
        if jsonl_files:
            dataset_file = os.path.join(args.dataset_path, jsonl_files[0])
            logger.info(f"Using dataset file: {dataset_file}")
        else:
            raise FileNotFoundError(f"No JSONL file found in {args.dataset_path}")
    
    logger.info(f"Loading dataset from {dataset_file}")
    dataset = load_dataset('json', data_files=dataset_file, split='train')
    
    logger.info(f"Dataset loaded: {len(dataset)} samples")
    logger.info(f"Dataset features: {dataset.features}")
    
    # Convert dataset to Chronos-2 input format
    inputs = []
    for sample in dataset:
        input_dict = {}
        
        # Handle target - convert to numpy array
        target = sample.get('target')
        if target is not None:
            input_dict['target'] = np.array(target, dtype=np.float32)
        else:
            logger.warning("Sample missing 'target' field, skipping")
            continue
        
        # Handle past covariates
        past_cov = sample.get('past_covariates')
        if past_cov and isinstance(past_cov, dict):
            input_dict['past_covariates'] = {
                k: np.array(v, dtype=np.float32) if v is not None else None
                for k, v in past_cov.items()
            }
        
        # Handle future covariates
        future_cov = sample.get('future_covariates')
        if future_cov and isinstance(future_cov, dict):
            # For training, future covariate values can be None
            # but keys must exist to indicate they'll be available at inference
            input_dict['future_covariates'] = {
                k: None for k in future_cov.keys()
            }
        
        inputs.append(input_dict)
    
    logger.info(f"Prepared {len(inputs)} training samples")
    
    # Split into train/validation if requested
    if args.validation_split > 0 and len(inputs) > 10:
        split_idx = int(len(inputs) * (1 - args.validation_split))
        train_inputs = inputs[:split_idx]
        val_inputs = inputs[split_idx:]
        logger.info(f"Split: {len(train_inputs)} train, {len(val_inputs)} validation samples")
        return train_inputs, val_inputs
    else:
        return inputs, None


def setup_mlflow(args: Chronos2ScriptArguments) -> None:
    """Configure MLflow tracking."""
    try:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(args.mlflow_experiment_name)
        logger.info(f"MLflow configured: {args.mlflow_tracking_uri}, experiment: {args.mlflow_experiment_name}")
    except Exception as e:
        logger.warning(f"Failed to configure MLflow: {e}. Continuing without MLflow.")


def get_lora_config(args: Chronos2ScriptArguments) -> LoraConfig:
    """Create LoRA configuration from arguments."""
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )


class MLflowLoggingCallback(TrainerCallback):
    """
    Callback to log training and validation metrics to MLflow.
    """
    
    def __init__(self, run_name: str = None):
        self.run_name = run_name
        self.run_started = False
        self.step_metrics = {}
    
    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Dict[str, float] = None,
        **kwargs
    ):
        """Log metrics at each logging step."""
        if logs is None:
            return
        
        # Filter out non-numeric values
        metrics_to_log = {}
        for key, value in logs.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                metrics_to_log[key] = value
        
        if metrics_to_log:
            try:
                mlflow.log_metrics(metrics_to_log, step=state.global_step)
                logger.info(f"Step {state.global_step}: {metrics_to_log}")
            except Exception as e:
                logger.warning(f"Failed to log metrics to MLflow: {e}")
    
    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Log training start."""
        logger.info("Training started - MLflow callback active")
    
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Log final metrics at training end."""
        logger.info("Training ended - logging final metrics")
        
        # Log best metric if available
        if state.best_metric is not None:
            try:
                mlflow.log_metric("best_metric", state.best_metric)
            except Exception as e:
                logger.warning(f"Failed to log best_metric: {e}")
    
    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Dict[str, float] = None,
        **kwargs
    ):
        """Log evaluation metrics."""
        if metrics is None:
            return
        
        eval_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                # Prefix with eval_ if not already
                metric_name = key if key.startswith("eval_") else f"eval_{key}"
                eval_metrics[metric_name] = value
        
        if eval_metrics:
            try:
                mlflow.log_metrics(eval_metrics, step=state.global_step)
                logger.info(f"Evaluation at step {state.global_step}: {eval_metrics}")
            except Exception as e:
                logger.warning(f"Failed to log eval metrics to MLflow: {e}")


class MetricsPrinterCallback(TrainerCallback):
    """
    Callback to print training metrics to console with formatting.
    """
    
    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Dict[str, float] = None,
        **kwargs
    ):
        """Print metrics at each logging step."""
        if logs is None:
            return
        
        # Format metrics for display
        metrics_str = " | ".join([
            f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in logs.items()
            if isinstance(v, (int, float))
        ])
        
        if metrics_str:
            print(f"[Step {state.global_step}] {metrics_str}")


def get_model_save_directory() -> str:
    """
    Get the directory path for saving the final model.
    
    Args:
        model_name: Name/path of the model
        
    Returns:
        Path to save directory
    """
    if "SM_MODEL_DIR" in os.environ:
        base_dir = os.environ["SM_MODEL_DIR"]
    else:
        base_dir = "/opt/ml/model"
    
    return base_dir

    
def train(args: Chronos2ScriptArguments) -> None:
    """Main training function."""
    logger.info("=" * 60)
    logger.info("Starting Chronos-2 Fine-Tuning")
    logger.info("=" * 60)
    
    # Log configuration
    logger.info(f"Model: {args.model_name_or_path}")
    logger.info(f"Fine-tune mode: {args.finetune_mode}")
    logger.info(f"Prediction length: {args.prediction_length}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Num steps: {args.num_steps}")
    logger.info(f"Batch size: {args.batch_size}")
    
    # Set seed
    set_seed(args.seed)
    logger.info(f"Random seed set to {args.seed}")
    
    # Setup MLflow
    setup_mlflow(args)
    
    # Load the Chronos-2 pipeline
    logger.info(f"Loading Chronos-2 model from {args.model_name_or_path}")
    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
        args.model_name_or_path,
        device_map=args.device_map
    )
    logger.info("Model loaded successfully")
    
    # Load training data
    train_inputs, val_inputs = load_training_data(args)
    
    # Prepare LoRA config if needed
    lora_config = None
    if args.finetune_mode == "lora":
        lora_config = get_lora_config(args)
        logger.info(f"LoRA config: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Start MLflow run
    start_time = datetime.now()
    
    with mlflow.start_run(run_name=args.run_name):
        # Log parameters
        mlflow.log_params({
            "model_name_or_path": args.model_name_or_path,
            "finetune_mode": args.finetune_mode,
            "prediction_length": args.prediction_length,
            "context_length": args.context_length,
            "learning_rate": args.learning_rate,
            "num_steps": args.num_steps,
            "batch_size": args.batch_size,
            "lora_r": args.lora_r if args.finetune_mode == "lora" else None,
            "lora_alpha": args.lora_alpha if args.finetune_mode == "lora" else None,
            "train_samples": len(train_inputs),
            "val_samples": len(val_inputs) if val_inputs else 0,
            "seed": args.seed,
        })
        
        # Create callbacks for logging
        callbacks = [
            MLflowLoggingCallback(run_name=args.run_name),
            MetricsPrinterCallback(),
        ]
        
        logger.info("Starting fine-tuning with MLflow logging callback...")
        
        # Fine-tune the model
        finetuned_pipeline = pipeline.fit(
            inputs=train_inputs,
            prediction_length=args.prediction_length,
            validation_inputs=val_inputs,
            finetune_mode=args.finetune_mode,
            lora_config=lora_config,
            context_length=args.context_length,
            learning_rate=args.learning_rate,
            num_steps=args.num_steps,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            min_past=args.min_past,
            finetuned_ckpt_name=args.finetuned_ckpt_name,
            callbacks=callbacks,
            logging_steps=args.logging_steps,
        )
        
        end_time = datetime.now()
        training_duration = end_time - start_time
        
        # Log training duration
        mlflow.log_metric("training_duration_seconds", training_duration.total_seconds())
        
        logger.info(f"Fine-tuning completed in {training_duration}")
        
        # Save the final model
        final_model_dir = get_model_save_directory()
        final_model_path = os.path.join(final_model_dir, args.model_name_or_path)
        finetuned_pipeline.model.save_pretrained(final_model_path)
        logger.info(f"Model saved to {final_model_path}")
        
        # Log model artifact path
        mlflow.log_param("model_output_path", final_model_path)
    
    logger.info("=" * 60)
    logger.info("Training completed successfully!")
    logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = HfArgumentParser(Chronos2ScriptArguments)
    
    # Parse arguments from command line or config file
    import sys
    if len(sys.argv) == 2 and sys.argv[1].endswith(('.yaml', '.yml', '.json')):
        # Load from config file
        args = parser.parse_yaml_file(sys.argv[1])[0]
    else:
        args = parser.parse_args_into_dataclasses()[0]
    
    train(args)


if __name__ == "__main__":
    main()
