from accelerate import Accelerator
import bitsandbytes as bnb
from dataclasses import dataclass, field
from datasets import Dataset, load_dataset
import datetime
from huggingface_hub import snapshot_download
import logging
import mlflow
from mlflow.models import infer_signature
import os
from peft import (
    AutoPeftModelForCausalLM,
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
)
from trl import DPOConfig, DPOTrainer, TrlParser
from transformers.trainer_utils import get_last_checkpoint
from transformers.integrations import WandbCallback
from typing import Any, Dict, List, Optional, Tuple
import wandb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
    """
    Arguments for the script execution.
    """

    checkpoint_dir: str = field(default=None, metadata={"help": "Checkpoint directory"})

    lora_r: Optional[int] = field(default=8, metadata={"help": "lora_r"})

    lora_alpha: Optional[int] = field(default=16, metadata={"help": "lora_dropout"})

    lora_dropout: Optional[float] = field(
        default=0.1, metadata={"help": "lora_dropout"}
    )

    merge_weights: Optional[bool] = field(
        default=False, metadata={"help": "Merge adapter with base model"}
    )

    mlflow_uri: Optional[str] = field(
        default=None, metadata={"help": "MLflow tracking ARN"}
    )

    mlflow_experiment_name: Optional[str] = field(
        default=None, metadata={"help": "MLflow experiment name"}
    )

    model_id: str = field(
        default=None, metadata={"help": "Model ID to use for SFT training"}
    )

    token: str = field(default=None, metadata={"help": "Hugging Face API token"})

    train_dataset_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the training dataset"}
    )

    val_dataset_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the test dataset"}
    )

    wandb_token: str = field(default="", metadata={"help": "Wandb API token"})

    wandb_project: str = field(
        default="project", metadata={"help": "Wandb project name"}
    )


class CustomWandbCallback(WandbCallback):
    """Custom Wandb callback that logs metrics for all GPUs."""

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if state.is_world_process_zero and logs:
            # Format logs to include GPU index
            logs = {f"gpu_{i}_{k}": v for i in range(8) for k, v in logs.items()}
            super().on_log(args, state, control, model, logs, **kwargs)


def init_distributed():
    # Initialize the process group
    torch.distributed.init_process_group(
        backend="nccl", timeout=datetime.timedelta(seconds=5400)
    )  # Use "gloo" backend for CPU
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    return local_rank


def download_model(model_name):
    print("Downloading model ", model_name)

    os.makedirs("/tmp/tmp_folder", exist_ok=True)

    snapshot_download(repo_id=model_name, local_dir="/tmp/tmp_folder")

    print(f"Model {model_name} downloaded under /tmp/tmp_folder")


def set_custom_env(env_vars: Dict[str, str]) -> None:
    """
    Set custom environment variables.

    Args:
        env_vars (Dict[str, str]): A dictionary of environment variables to set.
                                   Keys are variable names, values are their corresponding values.

    Returns:
        None

    Raises:
        TypeError: If env_vars is not a dictionary.
        ValueError: If any key or value in env_vars is not a string.
    """
    if not isinstance(env_vars, dict):
        raise TypeError("env_vars must be a dictionary")

    for key, value in env_vars.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("All keys and values in env_vars must be strings")

    os.environ.update(env_vars)

    # Optionally, print the updated environment variables
    print("Updated environment variables:")
    for key, value in env_vars.items():
        print(f"  {key}: {value}")


def is_mlflow_enabled(script_args: ScriptArguments) -> bool:
    """
    Check if MLflow is enabled based on script arguments.

    Args:
        script_args: Script arguments

    Returns:
        True if MLflow is enabled, False otherwise
    """
    return (
        script_args.mlflow_uri is not None
        and script_args.mlflow_experiment_name is not None
        and script_args.mlflow_uri != ""
        and script_args.mlflow_experiment_name != ""
    )


def setup_mlflow(script_args: ScriptArguments) -> None:
    """
    Set up MLflow tracking.

    Args:
        script_args: Script arguments
    """
    if not is_mlflow_enabled(script_args):
        return

    logger.info("Initializing MLflow")
    mlflow.enable_system_metrics_logging()
    mlflow.autolog()
    mlflow.set_tracking_uri(script_args.mlflow_uri)
    mlflow.set_experiment(script_args.mlflow_experiment_name)

    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M")
    set_custom_env(
        {
            "MLFLOW_RUN_NAME": f"Fine-tuning-{formatted_datetime}",
            "MLFLOW_EXPERIMENT_NAME": script_args.mlflow_experiment_name,
        }
    )


def setup_wandb(script_args: ScriptArguments) -> None:
    """
    Set up Weights & Biases tracking.

    Args:
        script_args: Script arguments

    Returns:
        List of callbacks or None
    """
    if script_args.wandb_token and script_args.wandb_token != "":
        logger.info("Initializing Wandb")
        set_custom_env({"WANDB_API_KEY": script_args.wandb_token})
        wandb.init(project=script_args.wandb_project)
        return [CustomWandbCallback()]
    else:
        set_custom_env({"WANDB_DISABLED": "true"})
        return None


def get_model_config(
    training_args: TrainingArguments,
) -> Tuple[torch.dtype, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Get model configuration based on training arguments.

    Args:
        training_args: Training arguments

    Returns:
        Tuple containing torch dtype, model configs, BnB config params, and trainer configs
    """
    # Set up model configuration based on training arguments
    if training_args.bf16:
        logger.info("Using flash_attention_2")
        torch_dtype = torch.bfloat16
        model_configs = {
            "attn_implementation": "flash_attention_2",
            "torch_dtype": torch_dtype,
        }
    else:
        logger.info("Using torch.float32")
        torch_dtype = torch.float32
        model_configs = {}

    # Set up FSDP configuration if enabled
    if (
        training_args.fsdp is not None
        and training_args.fsdp != ""
        and training_args.fsdp_config is not None
        and len(training_args.fsdp_config) > 0
    ):
        logger.info("Using FSDP configuration")
        bnb_config_params = {"bnb_4bit_quant_storage": torch_dtype}
        trainer_configs = {
            "fsdp": training_args.fsdp,
            "fsdp_config": training_args.fsdp_config,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
        }
    else:
        logger.info("Using DDP in case of distribution")
        bnb_config_params = {}
        trainer_configs = {
            "gradient_checkpointing": training_args.gradient_checkpointing,
        }

    return torch_dtype, model_configs, bnb_config_params, trainer_configs


def load_model_and_tokenizer(
    script_args: ScriptArguments, training_args: TrainingArguments
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load model and tokenizer.

    Args:
        script_args: Script arguments
        training_args: Training arguments

    Returns:
        Tuple containing model and tokenizer
    """
    # Get model configuration
    torch_dtype, model_configs, bnb_config_params, _ = get_model_config(training_args)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id)

    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        **bnb_config_params,
    )

    # Load model with quantization
    try:
        model = AutoModelForCausalLM.from_pretrained(
            script_args.model_id,
            trust_remote_code=True,
            quantization_config=bnb_config,
            use_cache=not training_args.gradient_checkpointing,
            cache_dir="/tmp/.cache",
            **model_configs,
        )

        # Apply gradient checkpointing configuration
        if training_args.fsdp is None and training_args.fsdp_config is None:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

            if training_args.gradient_checkpointing:
                model.gradient_checkpointing_enable()
        else:
            if training_args.gradient_checkpointing:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )

        return model, tokenizer

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def apply_lora_config(
    model: AutoModelForCausalLM, script_args: ScriptArguments
) -> AutoModelForCausalLM:
    """
    Apply LoRA configuration to the model.

    Args:
        model: The model to apply LoRA to
        script_args: Script arguments

    Returns:
        Model with LoRA applied
    """
    config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        target_modules="all-linear",
        lora_dropout=script_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    return get_peft_model(model, config)


def setup_trainer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_ds: Dataset,
    script_args: ScriptArguments,
    training_args: TrainingArguments,
    test_ds: Optional[Dataset] = None,
    callbacks: Optional[List] = None,
) -> DPOTrainer:
    """
    Set up the Trainer.

    Args:
        model: Model to train
        tokenizer: Tokenizer to use in the training loop
        train_ds: Training dataset
        script_args: Script arguments
        training_args: Training arguments
        test_ds: Evaluation dataset
        callbacks: List of callbacks

    Returns:
        Configured Trainer
    """
    _, _, _, trainer_configs = get_model_config(training_args)

    return DPOTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=test_ds if test_ds is not None else None,
        args=DPOConfig(
            per_device_train_batch_size=training_args.per_device_train_batch_size,
            per_device_eval_batch_size=training_args.per_device_eval_batch_size,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            logging_strategy="steps",
            logging_steps=1,
            log_on_each_node=False,
            num_train_epochs=training_args.num_train_epochs,
            learning_rate=training_args.learning_rate,
            bf16=training_args.bf16,
            ddp_find_unused_parameters=False,
            save_strategy="steps",
            save_steps=training_args.save_steps,
            save_total_limit=1,
            output_dir=script_args.checkpoint_dir,
            ignore_data_skip=True,
            warmup_steps=training_args.warmup_steps,
            weight_decay=training_args.weight_decay,
            **trainer_configs,
        ),
        callbacks=callbacks,
        processing_class=tokenizer,
    )


def save_model(
    trainer: DPOTrainer,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    script_args: ScriptArguments,
    training_args: TrainingArguments,
    accelerator: Accelerator,
    mlflow_enabled: bool,
) -> None:
    """
    Save the trained model.

    Args:
        trainer: The trainer instance
        model: The model to save
        tokenizer: The tokenizer to save
        script_args: Script arguments
        training_args: Training arguments
        accelerator: Accelerator instance
        mlflow_enabled: Whether MLflow is enabled
    """
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    if script_args.merge_weights:
        output_dir = "/tmp/model"

        # Save adapter weights with base model
        trainer.model.save_pretrained(output_dir, safe_serialization=False)

        if accelerator.is_main_process:
            # Clear memory
            del model
            del trainer
            torch.cuda.empty_cache()

            try:
                # Load PEFT model
                model = AutoPeftModelForCausalLM.from_pretrained(
                    output_dir,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )

                # Merge LoRA and base model and save
                model = model.merge_and_unload()
                model = model.to(torch.float16)

                model.save_pretrained(
                    training_args.output_dir,
                    safe_serialization=True,
                    max_shard_size="2GB",
                )
            except Exception as e:
                logger.error(f"Error merging model weights: {e}")
                raise
    else:
        trainer.model.save_pretrained(training_args.output_dir, safe_serialization=True)

    if accelerator.is_main_process:
        tokenizer.save_pretrained(training_args.output_dir)

        if mlflow_enabled:
            register_model_in_mlflow(model, tokenizer, script_args)


def register_model_in_mlflow(
    model: AutoModelForCausalLM, tokenizer: AutoTokenizer, script_args: ScriptArguments
) -> None:
    """
    Register the model in MLflow.

    Args:
        model: The model to register
        tokenizer: The tokenizer to register
        script_args: Script arguments
    """
    logger.info(f"MLflow model registration under {script_args.mlflow_experiment_name}")

    try:
        params = {
            "top_p": 0.9,
            "temperature": 0.2,
            "max_new_tokens": 2048,
        }
        signature = infer_signature("inputs", "generated_text", params=params)

        mlflow.transformers.log_model(
            transformers_model={"model": model, "tokenizer": tokenizer},
            signature=signature,
            artifact_path="model",
            model_config=params,
            task="text-generation",
            registered_model_name=f"model-{os.environ.get('MLFLOW_RUN_NAME', '').split('Fine-tuning-')[-1]}",
        )
    except Exception as e:
        logger.error(f"Error registering model in MLflow: {e}")
        raise


def train(script_args, training_args, train_ds, test_ds):
    """
    Train the model.

    Args:
        script_args: Script arguments
        training_args: Training arguments
        train_ds: Training dataset
        test_ds: Evaluation dataset
    """
    # Set random seed for reproducibility
    set_seed(training_args.seed)

    # Check if MLflow is enabled
    mlflow_enabled = is_mlflow_enabled(script_args)

    # Initialize accelerator
    accelerator = Accelerator()

    # Set up Hugging Face token if provided
    if script_args.token is not None:
        os.environ.update({"HF_TOKEN": script_args.token})
        accelerator.wait_for_everyone()

    # Download model
    download_model(script_args.model_id)
    accelerator.wait_for_everyone()

    # Update model path to local directory
    script_args.model_id = "/tmp/tmp_folder"

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(script_args, training_args)

    # Apply LoRA configuration
    model = apply_lora_config(model, script_args)

    # Set up Weights & Biases
    callbacks = setup_wandb(script_args)

    # Set up trainer
    trainer = setup_trainer(
        model,
        tokenizer,
        train_ds,
        script_args,
        training_args,
        test_ds,
        callbacks,
    )

    # Print trainable parameters
    if trainer.accelerator.is_main_process:
        trainer.model.print_trainable_parameters()

    # Create checkpoint directory if needed
    if script_args.checkpoint_dir is not None:
        os.makedirs(script_args.checkpoint_dir, exist_ok=True)

    # Start training
    if mlflow_enabled:
        logger.info(f"MLflow tracking under {script_args.mlflow_experiment_name}")
        with mlflow.start_run(run_name=os.environ.get("MLFLOW_RUN_NAME", None)) as run:
            # Log training dataset
            try:
                train_dataset_mlflow = mlflow.data.from_pandas(
                    train_ds.to_pandas(), name="train_dataset"
                )
                mlflow.log_input(train_dataset_mlflow, context="train")
            except Exception as e:
                logger.warning(f"Failed to log dataset to MLflow: {e}")

            # Resume training from checkpoint if available
            if get_last_checkpoint(script_args.checkpoint_dir) is not None:
                trainer.train(resume_from_checkpoint=True)
            else:
                trainer.train()
    else:
        # Resume training from checkpoint if available
        if get_last_checkpoint(script_args.checkpoint_dir) is not None:
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()

    # Save and register model
    save_model(
        trainer,
        model,
        tokenizer,
        script_args,
        training_args,
        accelerator,
        mlflow_enabled,
    )

    # Wait for all processes to finish
    accelerator.wait_for_everyone()


def load_datasets(script_args: ScriptArguments) -> Tuple[Dataset, Optional[Dataset]]:
    """
    Load training and test datasets.

    Args:
        script_args: Script arguments

    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    try:
        logger.info(f"Loading training dataset from {script_args.train_dataset_path}")
        train_ds = load_dataset(
            "json",
            data_files=os.path.join(script_args.train_dataset_path, "dataset.json"),
            split="train",
        )

        test_ds = None
        if script_args.val_dataset_path:
            logger.info(f"Loading test dataset from {script_args.val_dataset_path}")
            test_ds = load_dataset(
                "json",
                data_files=os.path.join(script_args.val_dataset_path, "dataset.json"),
                split="train",
            )

        return train_ds, test_ds
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        raise


def main() -> None:
    """Main function to parse arguments and start training."""
    # Call this function at the beginning of your script
    local_rank = init_distributed()

    # Now you can use distributed functionalities
    torch.distributed.barrier(device_ids=[local_rank])

    # Parse arguments
    parser = TrlParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_and_config()

    # Set up environment
    set_custom_env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})

    # Set up MLflow if enabled
    setup_mlflow(script_args)

    # Set up Weights & Biases
    setup_wandb(script_args)

    # Load datasets
    train_ds, test_ds = load_datasets(script_args)

    # Launch training
    train(script_args, training_args, train_ds, test_ds)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
