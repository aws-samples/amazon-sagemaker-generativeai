from accelerate import Accelerator
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
)
import torch
import torch.distributed as dist
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Mxfp4Config,
    Trainer,
    TrainingArguments,
    set_seed,
)
from trl import TrlParser
import transformers
from transformers.trainer_utils import get_last_checkpoint
from transformers.integrations import WandbCallback
from typing import Any, Dict, List, Optional, Tuple
import wandb
from distutils.util import strtobool

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

    apply_truncation: Optional[bool] = field(
        default=False, metadata={"help": "Whether to apply truncation"}
    )

    attn_implementation: Optional[str] = field(
        default="flash_attention_2",
        metadata={"help": "Attention implementation (flash_attention_2, sdpa, eager)"},
    )

    checkpoint_dir: str = field(default=None, metadata={"help": "Checkpoint directory"})

    use_checkpoints: bool = field(
        default=False, metadata={"help": "Whether to use checkpointing"}
    )

    load_in_4bit: bool = field(
        default=True, metadata={"help": "Load model in 4-bit quantization"}
    )

    lora_r: Optional[int] = field(default=8, metadata={"help": "lora_r"})

    lora_alpha: Optional[int] = field(default=16, metadata={"help": "lora_dropout"})

    lora_dropout: Optional[float] = field(
        default=0.1, metadata={"help": "lora_dropout"}
    )

    max_length: Optional[int] = field(
        default=None, metadata={"help": "max_length used for truncation"}
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

    use_mxfp4: bool = field(
        default=False,
        metadata={"help": "Use MXFP4 quantization instead of BitsAndBytes"},
    )

    use_snapshot_download: bool = field(
        default=True,
        metadata={"help": "Use snapshot download instead of Hugging Face Hub"},
    )

    val_dataset_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the vaò dataset"}
    )

    wandb_token: str = field(default="", metadata={"help": "Wandb API token"})

    wandb_project: str = field(
        default="project", metadata={"help": "Wandb project name"}
    )

    torch_dtype: Optional[str] = field(
        default="auto",
        metadata={"help": "Torch dtype (auto, bfloat16, float16, float32)"},
    )


class CustomWandbCallback(WandbCallback):
    """Custom Wandb callback that logs metrics for all GPUs."""

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if state.is_world_process_zero and logs:
            # Format logs to include GPU index
            logs = {f"gpu_{i}_{k}": v for i in range(8) for k, v in logs.items()}
            super().on_log(args, state, control, model, logs, **kwargs)


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
    script_args: ScriptArguments,
) -> Tuple[torch.dtype, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Get model configuration based on training arguments.

    Args:
        training_args: Training arguments

    Returns:
        Tuple containing torch dtype, model configs, BnB config params, and trainer configs
    """
    # Determine torch dtype from script args
    if script_args.torch_dtype in ["auto", None]:
        torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    else:
        torch_dtype = getattr(torch, script_args.torch_dtype)

    # Set up model configuration
    model_configs = {
        "attn_implementation": script_args.attn_implementation,
        "torch_dtype": torch_dtype,
        "use_cache": not training_args.gradient_checkpointing,
    }

    # Set low_cpu_mem_usage based on DeepSpeed usage
    use_deepspeed = strtobool(os.environ.get("ACCELERATE_USE_DEEPSPEED", "false"))
    use_fsdp = strtobool(os.environ.get("ACCELERATE_USE_FSDP", "false"))

    logger.info(f"DeepSpeed enabled: {use_deepspeed}")
    logger.info(f"FSDP enabled: {use_fsdp}")

    if not use_deepspeed:
        model_configs["low_cpu_mem_usage"] = True

    if use_fsdp or (training_args.fsdp and training_args.fsdp != ""):
        logger.info("Using FSDP configuration")
        if training_args.gradient_checkpointing_kwargs is not None:
            trainer_configs = {}
        else:
            trainer_configs = {
                "gradient_checkpointing_kwargs": {"use_reentrant": False},
            }
    elif use_deepspeed:
        logger.info("Using DeepSpeed configuration")
        trainer_configs = {}
    else:
        logger.info("Using DDP configuration")
        if training_args.gradient_checkpointing_kwargs is not None:
            trainer_configs = {}
        else:
            trainer_configs = {
                "gradient_checkpointing_kwargs": {"use_reentrant": False},
            }

    return torch_dtype, model_configs, trainer_configs


def load_model(
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
    torch_dtype, model_configs, _ = get_model_config(training_args, script_args)

    # Configure quantization
    if script_args.load_in_4bit:
        if script_args.use_mxfp4:
            quantization_config = Mxfp4Config(dequantize=True)
            logger.info("Using MXFP4 quantization")
        else:
            # Only add bnb_4bit_quant_storage for FSDP
            bnb_params = {
                "load_in_4bit": True,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch_dtype,
                "bnb_4bit_quant_storage": torch_dtype,
            }

            quantization_config = BitsAndBytesConfig(**bnb_params)
            logger.info("Using BitsAndBytes quantization")
    else:
        quantization_config = None
        logger.info("No quantization")

    # Load model with quantization
    try:
        model = AutoModelForCausalLM.from_pretrained(
            script_args.model_id,
            trust_remote_code=True,
            quantization_config=quantization_config,
            cache_dir="/tmp/.cache",
            **model_configs,
        )

        # Apply gradient checkpointing configuration
        if training_args.gradient_checkpointing:
            # Use reentrant=False for better compatibility with DeepSpeed and FSDP
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        return model

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def load_tokenizer(
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
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(script_args.model_id)

        if tokenizer.pad_token is None:
            # Define PAD token
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

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
) -> Trainer:
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
    _, _, trainer_configs = get_model_config(training_args, script_args)

    # Update training_args with trainer configs
    for key, value in trainer_configs.items():
        setattr(training_args, key, value)

    # Set report_to based on enabled tracking services
    report_to = []
    if os.environ.get("WANDB_DISABLED", "false").lower() != "true":
        report_to.append("wandb")
    if is_mlflow_enabled(script_args):
        report_to.append("mlflow")
    training_args.report_to = report_to

    return Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=test_ds if test_ds is not None else None,
        args=training_args,
        callbacks=callbacks,
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False
        ),
    )


def save_model(
    trainer: Trainer,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    script_args: ScriptArguments,
    training_args: TrainingArguments,
    accelerator: Accelerator,
    mlflow_enabled: bool,
) -> None:
    """Save the trained model."""
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    if script_args.merge_weights:
        # Save and merge PEFT model
        temp_dir = "/tmp/model"
        trainer.model.save_pretrained(temp_dir, safe_serialization=False)

        if accelerator.is_main_process:
            del model, trainer
            torch.cuda.empty_cache()

            model = AutoPeftModelForCausalLM.from_pretrained(
                temp_dir,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            model = model.merge_and_unload()
            model = model.to(torch.float16)

            model.save_pretrained(
                "/opt/ml/model", safe_serialization=True, max_shard_size="2GB"
            )
    else:
        trainer.model.save_pretrained("/opt/ml/model", safe_serialization=True)

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


def calculate_optimal_max_length(
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    sample_size: int = 1000,
    percentile: float = 0.95,
) -> int:
    """
    Calculate optimal max_length by actually tokenizing a sample of the dataset.

    Args:
        tokenizer: The tokenizer to use
        dataset: Dataset to analyze
        sample_size: Number of samples to tokenize for analysis
        percentile: Percentile to use for max_length (0.95 = 95th percentile)

    Returns:
        Optimal max_length based on actual token counts
    """
    # Sample dataset for analysis
    sample_indices = torch.randperm(len(dataset))[: min(sample_size, len(dataset))]
    sample_data = dataset.select(sample_indices)

    # Tokenize samples to get actual token lengths
    token_lengths = []
    for sample in sample_data:
        tokens = tokenizer(sample["text"], add_special_tokens=True)["input_ids"]
        token_lengths.append(len(tokens))

    # Calculate statistics
    avg_length = sum(token_lengths) / len(token_lengths)
    max_length = int(sorted(token_lengths)[int(percentile * len(token_lengths))])

    logger.info(f"Analyzed {len(token_lengths)} samples")
    logger.info(f"Average token length: {avg_length:.1f}")
    logger.info(f"{percentile*100}th percentile token length: {max_length}")

    return max_length


def prepare_dataset(
    tokenizer: AutoTokenizer,
    script_args: ScriptArguments,
    train_ds: Dataset,
    test_ds: Optional[Dataset] = None,
):
    """
    Prepare the dataset for training with optimal tokenization.

    Args:
        tokenizer: Tokenizer to use
        train_ds: Training dataset
        test_ds: Test dataset
        max_length: Optional fixed max_length, if None will be calculated

    Returns:
        Prepared tokenized datasets
    """
    # Calculate optimal max_length if not provided and truncation is enabled
    if script_args.apply_truncation:
        if script_args.max_length is None:
            max_length = calculate_optimal_max_length(tokenizer, train_ds)
        else:
            max_length = script_args.max_length
    else:
        max_length = None

    logger.info(f"Using max_length: {max_length}")
    logger.info(f"Truncation enabled: {script_args.apply_truncation}")

    # Tokenize training dataset
    lm_train_dataset = train_ds.map(
        lambda sample: tokenizer(
            sample["text"],
            padding=False,
            truncation=script_args.apply_truncation,
            max_length=max_length if script_args.apply_truncation else None,
        ),
        remove_columns=list(train_ds.features),
        batched=True,
        batch_size=1000,
    )

    # Tokenize test dataset if provided
    if test_ds is not None:
        lm_test_dataset = test_ds.map(
            lambda sample: tokenizer(
                sample["text"],
                padding=False,
                truncation=script_args.apply_truncation,
                max_length=max_length if script_args.apply_truncation else None,
            ),
            remove_columns=list(test_ds.features),
            batched=True,
            batch_size=1000,
        )
        logger.info(f"Total number of test samples: {len(lm_test_dataset)}")
    else:
        lm_test_dataset = None

    logger.info(f"Total number of train samples: {len(lm_train_dataset)}")
    return lm_train_dataset, lm_test_dataset


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

    # Set up Hugging Face token if provided
    if script_args.token is not None:
        os.environ.update({"HF_TOKEN": script_args.token})
        if dist.is_initialized():
            logger.info("Waiting for all processes after setting HF token")
            dist.barrier()

    if script_args.use_snapshot_download:
        # Download model
        download_model(script_args.model_id)
        if dist.is_initialized():
            logger.info("Waiting for all processes after model download")
            dist.barrier()

        # Update model path to local directory
        script_args.model_id = "/tmp/tmp_folder"

    # Load model and tokenizer
    model = load_model(script_args, training_args)
    tokenizer = load_tokenizer(script_args, training_args)

    train_ds, test_ds = prepare_dataset(tokenizer, script_args, train_ds, test_ds)

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
        training_args.output_dir = script_args.checkpoint_dir

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
            if (
                get_last_checkpoint(script_args.checkpoint_dir) is not None
                and script_args.use_checkpoints
            ):
                train_result = trainer.train(resume_from_checkpoint=True)
            else:
                train_result = trainer.train()
    else:
        # Resume training from checkpoint if available
        if (
            get_last_checkpoint(script_args.checkpoint_dir) is not None
            and script_args.use_checkpoints
        ):
            train_result = trainer.train(resume_from_checkpoint=True)
        else:
            train_result = trainer.train()

    # Log training metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_ds)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Save and register model
    save_model(
        trainer,
        model,
        tokenizer,
        script_args,
        training_args,
        trainer.accelerator,
        mlflow_enabled,
    )

    # Wait for all processes to finish
    trainer.accelerator.wait_for_everyone()


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

        # Support both JSON and JSONL files
        if script_args.train_dataset_path.endswith(".jsonl"):
            train_ds = load_dataset(
                "json", data_files=script_args.train_dataset_path, split="train"
            )
        else:
            train_ds = load_dataset(
                "json",
                data_files=os.path.join(script_args.train_dataset_path, "dataset.json"),
                split="train",
            )

        test_ds = None
        if script_args.val_dataset_path:
            logger.info(f"Loading test dataset from {script_args.val_dataset_path}")
            if script_args.val_dataset_path.endswith(".jsonl"):
                test_ds = load_dataset(
                    "json", data_files=script_args.val_dataset_path, split="train"
                )
            else:
                test_ds = load_dataset(
                    "json",
                    data_files=os.path.join(
                        script_args.val_dataset_path, "dataset.json"
                    ),
                    split="train",
                )

        return train_ds, test_ds
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        raise


def main() -> None:
    """Main function to parse arguments and start training."""
    # Parse arguments
    parser = TrlParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_and_config()

    # Set up environment
    set_custom_env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})

    # Set up MLflow if enabled
    setup_mlflow(script_args)

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
