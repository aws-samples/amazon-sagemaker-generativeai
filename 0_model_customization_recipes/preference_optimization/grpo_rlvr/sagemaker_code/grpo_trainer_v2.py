"""
GRPO (Group Relative Policy Optimization) Trainer for Tool Calling

This script supports:
- Dynamic tool function loading from configurable modules
- YAML-based configuration via TrlParser
- Distributed training with DeepSpeed and Accelerate
- Checkpoint resumption and model saving for deployment

Usage:
    python3 grpo_trainer.py --config recipes/Qwen/Qwen3-0.6B--grpo.yaml
"""

import importlib
import inspect
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from distutils.util import strtobool
from typing import Any, Callable, Dict, List, Optional

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOTrainer, GRPOConfig, TrlParser, ModelConfig


def setup_logging() -> logging.Logger:
    """Set up logging configuration for the training script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


logger = setup_logging()


def print_function_details(func: Callable, label: str = "Function") -> None:
    """
    Print detailed information about a function including name, signature, and docstring.

    Args:
        func: The function to inspect
        label: Label to display (e.g., "Tool Function", "Reward Function")
    """
    try:
        name = func.__name__
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or "No docstring available"
        
        # Truncate long docstrings for readability
        doc_lines = doc.split('\n')
        if len(doc_lines) > 5:
            doc = '\n'.join(doc_lines[:5]) + '\n    ...(truncated)'
        
        logger.info(f"  ├─ {label}: {name}")
        logger.info(f"  │    Signature: {name}{sig}")
        logger.info(f"  │    Docstring: {doc.split(chr(10))[0]}")  # First line only
    except Exception as e:
        logger.warning(f"  ├─ Could not inspect {label}: {e}")


def print_loaded_functions(
    tool_functions: List[Callable],
    reward_func: Callable,
    tools_source: str,
    reward_source: str
) -> None:
    """
    Print a summary of all loaded tool functions and reward function.

    Args:
        tool_functions: List of loaded tool functions
        reward_func: The loaded reward function
        tools_source: Source path/module of tool functions
        reward_source: Source path/name of reward function
    """
    logger.info("=" * 70)
    logger.info("LOADED FUNCTIONS SUMMARY")
    logger.info("=" * 70)
    
    # Print tool functions
    logger.info(f"┌─ TOOL FUNCTIONS ({len(tool_functions)} loaded)")
    logger.info(f"│  Source: {tools_source}")
    logger.info("│")
    for i, func in enumerate(tool_functions):
        is_last = (i == len(tool_functions) - 1)
        prefix = "└" if is_last else "├"
        try:
            name = func.__name__
            sig = inspect.signature(func)
            doc = inspect.getdoc(func)
            first_line = doc.split('\n')[0] if doc else "No description"
            logger.info(f"│  {prefix}─ [{i+1}] {name}{sig}")
            logger.info(f"│  {'   ' if is_last else '│  '}   └─ {first_line}")
        except Exception as e:
            logger.info(f"│  {prefix}─ [{i+1}] <unable to inspect: {e}>")
    
    logger.info("│")
    
    # Print reward function
    logger.info(f"└─ REWARD FUNCTION")
    logger.info(f"   Source: {reward_source}")
    try:
        name = reward_func.__name__
        sig = inspect.signature(reward_func)
        doc = inspect.getdoc(reward_func)
        first_line = doc.split('\n')[0] if doc else "No description"
        logger.info(f"   └─ {name}{sig}")
        logger.info(f"      └─ {first_line}")
    except Exception as e:
        logger.info(f"   └─ <unable to inspect: {e}>")
    
    logger.info("=" * 70)


@dataclass
class ScriptArguments:
    """Custom arguments for GRPO training scripts."""

    dataset_id_or_path: str = field(
        metadata={"help": "Path to dataset file (.jsonl) or HuggingFace dataset identifier."}
    )

    tool_functions_module: str = field(
        default="tools_funcs.financial_tools_complex",
        metadata={"help": "Python module path containing tool functions to load."}
    )

    tools_script: Optional[str] = field(
        default=None,
        metadata={"help": "Path to custom tool functions script (must export TOOL_FUNCTIONS list). Overrides tool_functions_module."}
    )

    reward_function_name: str = field(
        default="accuracy",
        metadata={"help": "Name of reward function to use: 'accuracy' or custom."}
    )

    reward_fn: Optional[str] = field(
        default=None,
        metadata={"help": "Path to custom reward function script (must export reward_func callable). Overrides reward_function_name."}
    )

    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to tokenizer. If None, uses model tokenizer."}
    )

    max_grpo_prompt_length: int = field(
        default=512,
        metadata={"help": "Maximum prompt length for generation."}
    )

    max_grpo_completion_length: int = field(
        default=512,
        metadata={"help": "Maximum completion length for generation."}
    )


def load_tool_functions_from_file(file_path: str) -> List[Callable]:
    """
    Load tool functions from a Python file.

    Args:
        file_path: Path to Python file containing TOOL_FUNCTIONS list

    Returns:
        List of callable tool functions

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If no TOOL_FUNCTIONS found in file
    """
    import importlib.util

    logger.info(f"Loading tool functions from file: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Tool functions file not found: {file_path}")

    spec = importlib.util.spec_from_file_location("custom_tools", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, 'TOOL_FUNCTIONS'):
        raise ValueError(
            f"File '{file_path}' does not have a TOOL_FUNCTIONS list. "
            "Please define TOOL_FUNCTIONS = [func1, func2, ...] in your file."
        )

    tool_functions = module.TOOL_FUNCTIONS
    if not isinstance(tool_functions, list) or not tool_functions:
        raise ValueError(f"TOOL_FUNCTIONS in {file_path} must be a non-empty list")

    logger.info(f"Loaded {len(tool_functions)} tool functions from {file_path}")
    return tool_functions


def load_tool_functions(module_path: str) -> List[Callable]:
    """
    Dynamically load tool functions from a module.

    Args:
        module_path: Dot-separated module path (e.g., "tools_funcs.financial_tools_complex")

    Returns:
        List of callable tool functions

    Raises:
        ImportError: If module cannot be imported
        ValueError: If no TOOL_FUNCTIONS found in module
    """
    logger.info(f"Loading tool functions from module: {module_path}")

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        logger.error(f"Failed to import module '{module_path}': {e}")
        raise ImportError(f"Cannot import module '{module_path}': {e}")

    # Look for explicit TOOL_FUNCTIONS list
    if hasattr(module, 'TOOL_FUNCTIONS'):
        tool_functions = module.TOOL_FUNCTIONS
        if not isinstance(tool_functions, list):
            raise ValueError(f"TOOL_FUNCTIONS in {module_path} must be a list")
        if not tool_functions:
            raise ValueError(f"TOOL_FUNCTIONS in {module_path} is empty")
        logger.info(f"Loaded {len(tool_functions)} tool functions from {module_path}")
        return tool_functions

    raise ValueError(
        f"Module '{module_path}' does not have a TOOL_FUNCTIONS list. "
        "Please define TOOL_FUNCTIONS = [func1, func2, ...] in your module."
    )


def accuracy_reward(completions: List[List[Dict]], answer: List[str], **kwargs) -> List[float]:
    """
    Simple reward function for tool calling accuracy.

    Reward values:
    - 0.0: No tool call (tool_response is None)
    - 0.1: Some tool call made (any tool response)
    - 1.0: Exact match with expected answer

    Args:
        completions: List of completion message lists
        answer: List of expected answers

    Returns:
        List of reward values
    """
    rewards = []

    for completion, ans in zip(completions, answer):
        ans_str = str(ans).lower().strip()
        reward = 0.0
        tool_response = None

        for message in completion:
            role = message.get("role", "")
            content = message.get("content", "") or ""

            if role == "tool":
                tool_response = content

        # Assign reward based on tool response
        if tool_response is None:
            reward = 0.0  # No tool call
        elif ans_str == tool_response.lower().strip():
            reward = 1.0  # Exact match
        else:
            reward = 0.1  # Some tool call made

        rewards.append(reward)

    logger.info(f"BATCH: {len(rewards)} completions, Mean: {sum(rewards)/len(rewards):.3f}")
    return rewards


REWARD_FUNCTIONS = {
    "accuracy": accuracy_reward,
}


def load_reward_function_from_file(file_path: str) -> Callable:
    """
    Load a reward function from a Python file.

    Args:
        file_path: Path to Python file containing reward_func callable

    Returns:
        Callable reward function

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If no reward_func found in file
    """
    import importlib.util

    logger.info(f"Loading reward function from file: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file not found: {file_path}")

    spec = importlib.util.spec_from_file_location("custom_reward", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, 'reward_func'):
        raise ValueError(
            f"File '{file_path}' does not have a reward_func callable. "
            "Please define reward_func(completions, answer, **kwargs) in your file."
        )

    reward_func = module.reward_func
    if not callable(reward_func):
        raise ValueError(f"reward_func in {file_path} must be callable")

    logger.info(f"Loaded reward function from {file_path}")
    return reward_func


def load_training_dataset(script_args: ScriptArguments) -> Dataset:
    """
    Load training dataset based on script arguments.

    Args:
        script_args: Script arguments containing dataset configuration

    Returns:
        Loaded dataset

    Raises:
        ValueError: If dataset loading fails
    """
    dataset_path = script_args.dataset_id_or_path

    try:
        if dataset_path.endswith('.jsonl'):
            logger.info(f"Loading JSONL dataset from {dataset_path}")
            dataset = load_dataset('json', data_files=dataset_path, split='train')
        else:
            logger.info(f"Loading HuggingFace dataset: {dataset_path}")
            dataset = load_dataset(dataset_path, split='train')

        logger.info(f"Loaded dataset: {len(dataset)} samples, features: {dataset.features}")
        return dataset

    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise


def setup_tokenizer(
    script_args: ScriptArguments,
    model_args: ModelConfig
) -> PreTrainedTokenizer:
    """
    Load and configure the tokenizer.

    Args:
        script_args: Script arguments containing tokenizer configuration
        model_args: Model arguments containing model configuration

    Returns:
        Configured tokenizer
    """
    tokenizer_name = script_args.tokenizer_name_or_path or model_args.model_name_or_path

    logger.info(f"Loading tokenizer from {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")

    # Configure padding side for generation
    tokenizer.padding_side = "left"
    logger.info("Set padding_side to 'left' for generation")

    return tokenizer


def create_model_kwargs(
    model_args: ModelConfig,
    training_args: GRPOConfig
) -> Dict[str, Any]:
    """
    Create model loading arguments based on configuration.

    Args:
        model_args: Model configuration
        training_args: Training configuration

    Returns:
        Dictionary of model loading arguments
    """
    # Determine torch dtype
    if model_args.dtype in ['auto', None]:
        torch_dtype = model_args.dtype
    else:
        torch_dtype = getattr(torch, model_args.dtype)

    model_kwargs = {
        'revision': model_args.model_revision,
        'trust_remote_code': model_args.trust_remote_code,
        'attn_implementation': model_args.attn_implementation,
        'dtype': torch_dtype,
    }

    # Set low_cpu_mem_usage based on DeepSpeed usage
    use_deepspeed = strtobool(os.environ.get("ACCELERATE_USE_DEEPSPEED", "false"))
    if not use_deepspeed:
        model_kwargs['low_cpu_mem_usage'] = True

    return model_kwargs


def load_model(
    model_args: ModelConfig,
    model_kwargs: Dict[str, Any]
) -> PreTrainedModel:
    """
    Load the pretrained model with appropriate configuration.

    Args:
        model_args: Model configuration
        model_kwargs: Model loading arguments

    Returns:
        Loaded model
    """
    model_name = model_args.model_name_or_path
    logger.info(f"Loading model from {model_name}")

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    logger.info(f"Model loaded successfully: {model.__class__.__name__}")

    return model


def get_checkpoint_path(training_args: GRPOConfig) -> Optional[str]:
    """
    Get the path to the last checkpoint if it exists.

    Args:
        training_args: Training configuration containing output directory

    Returns:
        Path to last checkpoint or None if no checkpoint exists
    """
    if os.path.isdir(training_args.output_dir):
        checkpoint = get_last_checkpoint(training_args.output_dir)
        if checkpoint:
            logger.info(f"Found checkpoint: {checkpoint}")
        return checkpoint
    return None


def get_model_save_directory(model_name: str) -> str:
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

    return os.path.join(base_dir, model_name)



def save_model(
    trainer: GRPOTrainer,
    training_args: GRPOConfig,
    model_args: ModelConfig
) -> str:
    """
    Save the full model safely under distributed training.

    Args:
        trainer: The GRPO trainer instance
        training_args: Training configuration
        model_args: Model configuration

    Returns:
        Path to saved model directory
    """
    acc = trainer.accelerator
    final_model_dir = get_model_save_directory(model_args.model_name_or_path)

    logger.info(f"Saving full model to {final_model_dir}")
    trainer.save_model(final_model_dir)

    # Wait for all processes
    if hasattr(training_args, 'distributed_state'):
        training_args.distributed_state.wait_for_everyone()

    if acc.is_main_process:
        # Save tokenizer/processor
        tokenizer_or_processor = getattr(trainer, "processing_class", None)
        if tokenizer_or_processor is not None:
            logger.info("Saving tokenizer to disk")
            tokenizer_or_processor.save_pretrained(final_model_dir)

    return final_model_dir


def train_function(
    model_args: ModelConfig,
    script_args: ScriptArguments,
    training_args: GRPOConfig
) -> None:
    """
    Main training function that orchestrates the entire GRPO process.

    Args:
        model_args: Model configuration from TRL parser
        script_args: Custom script arguments
        training_args: Training configuration from TRL parser
    """
    logger.info("=" * 50)
    logger.info("Starting GRPO Training for Tool Calling")
    logger.info("=" * 50)

    # Log all parameters
    logger.info(f"Model parameters: {model_args}")
    logger.info(f"Script parameters: {script_args}")
    logger.info(f"Training parameters: {training_args}")

    # Load tool functions - prefer file path over module path
    if script_args.tools_script:
        tool_functions = load_tool_functions_from_file(script_args.tools_script)
        tools_source = script_args.tools_script
    else:
        tool_functions = load_tool_functions(script_args.tool_functions_module)
        tools_source = script_args.tool_functions_module

    # Get reward function - prefer file path over built-in name
    if script_args.reward_fn:
        reward_func = load_reward_function_from_file(script_args.reward_fn)
        reward_source = script_args.reward_fn
        logger.info(f"Using custom reward function from: {script_args.reward_fn}")
    else:
        reward_func_name = script_args.reward_function_name
        if reward_func_name not in REWARD_FUNCTIONS:
            raise ValueError(
                f"Unknown reward function: {reward_func_name}. "
                f"Available: {list(REWARD_FUNCTIONS.keys())}"
            )
        reward_func = REWARD_FUNCTIONS[reward_func_name]
        reward_source = f"built-in:{reward_func_name}"
        logger.info(f"Using built-in reward function: {reward_func_name}")

    # Print loaded functions summary
    print_loaded_functions(tool_functions, reward_func, tools_source, reward_source)

    # Load dataset
    dataset = load_training_dataset(script_args)

    # Setup tokenizer
    tokenizer = setup_tokenizer(script_args, model_args)

    # Load model
    model_kwargs = create_model_kwargs(model_args, training_args)
    model = load_model(model_args, model_kwargs)
    
    grpo_training_args = GRPOConfig(
        output_dir=training_args.output_dir,
        learning_rate=training_args.learning_rate,
        lr_scheduler_type=training_args.lr_scheduler_type,
        warmup_ratio=training_args.warmup_ratio,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        num_generations=training_args.num_generations,
        num_train_epochs=training_args.num_train_epochs,
        logging_steps=training_args.logging_steps,
        save_steps=training_args.save_steps,
        save_strategy=training_args.save_strategy,
        save_total_limit=training_args.save_total_limit,
        # max_prompt_length=script_args.max_grpo_prompt_length,
        max_completion_length=script_args.max_grpo_completion_length,
        seed=training_args.seed,
        gradient_checkpointing=training_args.gradient_checkpointing,
        report_to=training_args.report_to,
    )

    # Wait for all processes in distributed training
    if hasattr(grpo_training_args, 'distributed_state'):
        grpo_training_args.distributed_state.wait_for_everyone()

    print(f"=" * 50)
    print(f"GRPO TRAINING CONFIGURATION ->")
    for arg, value in vars(grpo_training_args).items():
        print(f"  ├─ {arg}: {value}")
    print(f"=" * 50)
    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        args=grpo_training_args,
        processing_class=tokenizer,
        train_dataset=dataset,
        reward_funcs=reward_func,
        tools=tool_functions,
    )

    # Check for existing checkpoint
    last_checkpoint = get_checkpoint_path(training_args)
    if last_checkpoint:
        logger.info(f"Resuming training from checkpoint: {last_checkpoint}")

    # Start training
    start_time = datetime.now()
    logger.info(f"Starting training at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Training for {training_args.num_train_epochs} epochs")

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    # Log training metrics
    metrics = train_result.metrics
    metrics['train_samples'] = len(dataset)
    trainer.log_metrics('train', metrics)
    trainer.save_metrics('train', metrics)
    trainer.save_state()

    # Ensure all ranks finish training/metric logging before save
    trainer.accelerator.wait_for_everyone()

    # Save model
    save_model(trainer, training_args, model_args)

    # Wait for all processes before finishing
    if hasattr(training_args, 'distributed_state'):
        training_args.distributed_state.wait_for_everyone()

    end_time = datetime.now()
    training_duration = end_time - start_time
    logger.info(f"Training completed successfully in {training_duration}")
    logger.info("=" * 50)


def main() -> None:
    """
    Main entry point for the GRPO training script.

    Parses arguments using TRL parser and runs the training function.
    """
    try:
        # Parse arguments using TRL parser
        parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
        model_args, script_args, training_args = parser.parse_args_and_config()

        # Set seed for reproducibility
        set_seed(training_args.seed)
        logger.info(f"Set random seed to {training_args.seed}")

        # Run the main training loop
        train_function(model_args, script_args, training_args)

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == '__main__':
    main()
