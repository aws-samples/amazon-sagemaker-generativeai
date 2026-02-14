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
import subprocess
import sys
import textwrap
import torch
import torch.distributed as dist
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Mxfp4Config,
    Trainer,
    TrainingArguments,
    set_seed,
)
from trl import TrlParser
import transformers
from transformers.trainer_utils import get_last_checkpoint
from transformers.integrations import WandbCallback
import contextlib
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
    """Arguments for the script execution."""

    apply_truncation: Optional[bool] = field(
        default=False, metadata={"help": "Whether to apply truncation"}
    )
    attn_implementation: Optional[str] = field(
        default="flash_attention_2", metadata={"help": "Attention implementation"}
    )
    checkpoint_dir: str = field(default=None, metadata={"help": "Checkpoint directory"})
    use_checkpoints: bool = field(
        default=False, metadata={"help": "Whether to use checkpointing"}
    )
    early_stopping: bool = field(
        default=False, metadata={"help": "Whether to use early stopping"}
    )
    load_in_4bit: bool = field(
        default=True, metadata={"help": "Load model in 4-bit quantization"}
    )
    lora_r: Optional[int] = field(default=8, metadata={"help": "lora_r"})
    lora_alpha: Optional[int] = field(default=16, metadata={"help": "lora_alpha"})
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
    text_field: str = field(
        default="text",
        metadata={
            "help": "Name of the text field in dataset (e.g., 'text', 'content', 'message')"
        },
    )
    token: str = field(default=None, metadata={"help": "Hugging Face API token"})
    train_dataset_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the training dataset"}
    )
    use_mxfp4: bool = field(
        default=False,
        metadata={"help": "Use MXFP4 quantization instead of BitsAndBytes"},
    )
    use_peft: bool = field(default=True, metadata={"help": "Use PEFT for training"})
    use_snapshot_download: bool = field(
        default=True,
        metadata={"help": "Use snapshot download instead of Hugging Face Hub"},
    )
    val_dataset_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the val dataset"}
    )
    wandb_token: str = field(default="", metadata={"help": "Wandb API token"})
    wandb_project: str = field(
        default="project", metadata={"help": "Wandb project name"}
    )
    target_modules: Optional[List[str]] = field(
        default=None, metadata={"help": "Target modules for LoRA"}
    )
    torch_dtype: Optional[str] = field(
        default="auto",
        metadata={"help": "Torch dtype (auto, bfloat16, float16, float32)"},
    )


class ModelConfigBuilder:
    """Centralized model configuration builder to eliminate duplicate logic."""

    def __init__(self, script_args: ScriptArguments, training_args: TrainingArguments):
        self.script_args = script_args
        self.training_args = training_args
        self._torch_dtype = None
        self._quantization_config = None
        self._use_deepspeed = None
        self._use_fsdp = None

    @property
    def torch_dtype(self) -> torch.dtype:
        """Get torch dtype with single source of truth."""
        if self._torch_dtype is None:
            if self.script_args.torch_dtype in ["auto", None]:
                self._torch_dtype = (
                    torch.bfloat16 if self.training_args.bf16 else torch.float32
                )
            else:
                self._torch_dtype = getattr(torch, self.script_args.torch_dtype)
        return self._torch_dtype

    @property
    def use_deepspeed(self) -> bool:
        """Check if DeepSpeed is enabled."""
        if self._use_deepspeed is None:
            self._use_deepspeed = strtobool(
                os.environ.get("ACCELERATE_USE_DEEPSPEED", "false")
            )
        return self._use_deepspeed

    @property
    def use_fsdp(self) -> bool:
        """Check if FSDP is enabled."""
        if self._use_fsdp is None:
            self._use_fsdp = strtobool(os.environ.get("ACCELERATE_USE_FSDP", "false"))
        return self._use_fsdp

    @property
    def quantization_config(self) -> Optional[Any]:
        """Get quantization configuration."""
        if self._quantization_config is None and self.script_args.load_in_4bit:
            if self.script_args.use_mxfp4:
                self._quantization_config = Mxfp4Config(dequantize=True)
                logger.info("Using MXFP4 quantization")
            else:
                self._quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=self.torch_dtype,
                    bnb_4bit_quant_storage=self.torch_dtype,
                )
                logger.info("Using BitsAndBytes quantization")
        return self._quantization_config

    def build_model_kwargs(self) -> Dict[str, Any]:
        """Build complete model loading arguments."""
        if (
            self.script_args.attn_implementation is not None
            and self.script_args.attn_implementation != ""
        ):
            model_kwargs = {
                "attn_implementation": self.script_args.attn_implementation,
                "torch_dtype": self.torch_dtype,
                "trust_remote_code": True,
                "cache_dir": "/tmp/.cache",
            }
        else:
            model_kwargs = {
                "torch_dtype": self.torch_dtype,
                "trust_remote_code": True,
                "cache_dir": "/tmp/.cache",
            }

        # Set low_cpu_mem_usage based on DeepSpeed usage
        if not self.use_deepspeed:
            model_kwargs["low_cpu_mem_usage"] = True

        # Add quantization config if enabled
        if self.quantization_config is not None:
            model_kwargs["quantization_config"] = self.quantization_config

        return model_kwargs

    def build_trainer_kwargs(self) -> Dict[str, Any]:
        """Build trainer-specific configuration."""
        trainer_kwargs = {}

        if self.use_fsdp or (self.training_args.fsdp and self.training_args.fsdp != ""):
            logger.info("Using FSDP configuration")
            if self.training_args.gradient_checkpointing_kwargs is None:
                trainer_kwargs["gradient_checkpointing_kwargs"] = {
                    "use_reentrant": False
                }
        elif self.use_deepspeed:
            logger.info("Using DeepSpeed configuration")
        else:
            logger.info("Using DDP configuration")
            if self.training_args.gradient_checkpointing_kwargs is None:
                trainer_kwargs["gradient_checkpointing_kwargs"] = {
                    "use_reentrant": False
                }

        return trainer_kwargs


class CustomWandbCallback(WandbCallback):
    """Custom Wandb callback that logs metrics for all GPUs."""

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if state.is_world_process_zero and logs:
            logs = {f"gpu_{i}_{k}": v for i in range(8) for k, v in logs.items()}
            super().on_log(args, state, control, model, logs, **kwargs)


@contextlib.contextmanager
def gpu_memory_manager():
    """Context manager for GPU memory cleanup."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(
                f"GPU memory freed: {torch.cuda.memory_allocated() / 1e9:.2f}GB allocated"
            )


@contextlib.contextmanager
def model_lifecycle(model_name: str):
    """Context manager for model loading/cleanup lifecycle."""
    model = None
    try:
        logger.info(f"Loading model: {model_name}")
        yield model
    except Exception as e:
        logger.error(f"Error in model lifecycle for {model_name}: {e}")
        raise
    finally:
        if model is not None:
            logger.info(f"Cleaning up model: {model_name}")
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def download_model(model_name):
    print("Downloading model ", model_name)
    os.makedirs("/tmp/tmp_folder", exist_ok=True)
    snapshot_download(repo_id=model_name, local_dir="/tmp/tmp_folder")
    print(f"Model {model_name} downloaded under /tmp/tmp_folder")


def set_custom_env(env_vars: Dict[str, str]) -> None:
    """Set custom environment variables."""
    if not isinstance(env_vars, dict):
        raise TypeError("env_vars must be a dictionary")

    for key, value in env_vars.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("All keys and values in env_vars must be strings")

    os.environ.update(env_vars)
    print("Updated environment variables:")
    for key, value in env_vars.items():
        print(f"  {key}: {value}")


def is_mlflow_enabled(script_args: ScriptArguments) -> bool:
    """Check if MLflow is enabled based on script arguments."""
    return (
        script_args.mlflow_uri is not None
        and script_args.mlflow_experiment_name is not None
        and script_args.mlflow_uri != ""
        and script_args.mlflow_experiment_name != ""
    )


def setup_mlflow(script_args: ScriptArguments) -> None:
    """Set up MLflow tracking."""
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
    """Set up Weights & Biases tracking."""
    if script_args.wandb_token and script_args.wandb_token != "":
        logger.info("Initializing Wandb")
        set_custom_env({"WANDB_API_KEY": script_args.wandb_token})
        wandb.init(project=script_args.wandb_project)
        return [CustomWandbCallback()]
    else:
        set_custom_env({"WANDB_DISABLED": "true"})
        return None


def load_model(
    config_builder: ModelConfigBuilder, script_args: ScriptArguments
) -> AutoModelForCausalLM:
    """Load model using centralized configuration."""
    model_kwargs = config_builder.build_model_kwargs()

    try:
        model = AutoModelForCausalLM.from_pretrained(
            script_args.model_id, **model_kwargs
        )

        # Apply gradient checkpointing configuration
        if config_builder.training_args.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        return model
    except Exception as e:
        logger.error(f"Error loading model {script_args.model_id}: {e}")
        raise


def load_tokenizer(script_args: ScriptArguments) -> AutoTokenizer:
    """Load tokenizer."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(script_args.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except Exception as e:
        logger.error(f"Error loading tokenizer {script_args.model_id}: {e}")
        raise


def load_processor(script_args: ScriptArguments):
    """Load processor for multimodal models. Returns None if unavailable."""
    try:
        processor = AutoProcessor.from_pretrained(script_args.model_id)
        logger.info(f"Loaded processor for {script_args.model_id}")
        return processor
    except Exception as e:
        logger.warning(f"No processor found for {script_args.model_id}: {e}")
        return None


def apply_lora_config(
    model: AutoModelForCausalLM, script_args: ScriptArguments
) -> AutoModelForCausalLM:
    """Apply LoRA configuration to the model."""
    config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        target_modules=(
            "all-linear"
            if script_args.target_modules is None
            else script_args.target_modules
        ),
        lora_dropout=script_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def setup_trainer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_ds: Dataset,
    config_builder: ModelConfigBuilder,
    test_ds: Optional[Dataset] = None,
    callbacks: Optional[List] = None,
) -> Trainer:
    """Set up the Trainer using centralized configuration."""
    trainer_kwargs = config_builder.build_trainer_kwargs()

    # Update training_args with trainer configs
    for key, value in trainer_kwargs.items():
        setattr(config_builder.training_args, key, value)

    # Set report_to based on enabled tracking services
    report_to = []
    if os.environ.get("WANDB_DISABLED", "false").lower() != "true":
        report_to.append("wandb")
    if is_mlflow_enabled(config_builder.script_args):
        report_to.append("mlflow")
    config_builder.training_args.report_to = report_to

    return Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=test_ds if test_ds is not None else None,
        args=config_builder.training_args,
        callbacks=callbacks,
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False
        ),
    )


def _merge_adapter_in_process(
    temp_dir: str, final_output_dir: str
) -> AutoModelForCausalLM:
    """Merge LoRA adapter in the current process (for FSDP/DDP)."""
    with gpu_memory_manager():
        model = AutoPeftModelForCausalLM.from_pretrained(
            temp_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model = model.merge_and_unload()
        model.save_pretrained(
            final_output_dir, safe_serialization=True, max_shard_size="2GB"
        )
        return model


def _merge_adapter_via_subprocess(temp_dir: str, final_output_dir: str) -> None:
    """Merge LoRA adapter in a clean subprocess to avoid DeepSpeed env conflicts."""
    merge_script = textwrap.dedent(
        f"""\
        import torch
        from peft import AutoPeftModelForCausalLM

        print("Loading adapter for merging...")
        model = AutoPeftModelForCausalLM.from_pretrained(
            "{temp_dir}",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        print("Merging LoRA weights...")
        model = model.merge_and_unload()

        print("Saving merged model...")
        model.save_pretrained(
            "{final_output_dir}",
            safe_serialization=True,
            max_shard_size="2GB",
        )

        print("Merge complete!")
    """
    )

    clean_env = {
        k: v
        for k, v in os.environ.items()
        if "DEEPSPEED" not in k and "ACCELERATE" not in k
    }

    result = subprocess.run(
        [sys.executable, "-c", merge_script],
        env=clean_env,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"Merge subprocess failed: {result.stderr}")
        raise RuntimeError(f"Merge failed: {result.stderr}")

    logger.info(f"Merge subprocess output: {result.stdout}")


def _detect_distributed_strategy(trainer: Trainer) -> Tuple[bool, bool]:
    """Detect whether DeepSpeed or FSDP is active."""
    use_deepspeed = (
        hasattr(trainer.accelerator.state, "deepspeed_plugin")
        and trainer.accelerator.state.deepspeed_plugin is not None
    )
    use_fsdp = trainer.is_fsdp_enabled
    return use_deepspeed, use_fsdp


def _save_artifacts_on_main(
    tokenizer: AutoTokenizer,
    processor,
    final_output_dir: str,
) -> None:
    """Save tokenizer and processor to the output directory."""
    tokenizer.save_pretrained(final_output_dir)
    if processor is not None:
        processor.save_pretrained(final_output_dir)


def save_model(
    trainer: Trainer,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    processor,
    script_args: ScriptArguments,
    accelerator: Accelerator,
    mlflow_enabled: bool,
    final_output_dir: str,
) -> None:
    """Save the trained model with proper DeepSpeed ZeRO-3 handling and online merging."""
    logger.info("STARTING MODEL SAVE PROCESS")

    accelerator.wait_for_everyone()

    use_deepspeed, use_fsdp = _detect_distributed_strategy(trainer)
    logger.info(f"Distributed strategy - DeepSpeed: {use_deepspeed}, FSDP: {use_fsdp}")

    if use_fsdp:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    if script_args.use_peft and script_args.merge_weights:
        temp_dir = "/tmp/adapter_temp"
        os.makedirs(temp_dir, exist_ok=True)

        if use_deepspeed:
            # Trainer.save_model handles ZeRO-3 state dict gathering
            trainer.save_model(temp_dir)
            accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                torch.cuda.empty_cache()
                _merge_adapter_via_subprocess(temp_dir, final_output_dir)
                _save_artifacts_on_main(tokenizer, processor, final_output_dir)
                if mlflow_enabled:
                    logger.info(
                        "Skipping MLflow registration (model merged in subprocess)"
                    )
        else:
            trainer.model.save_pretrained(temp_dir, safe_serialization=False)
            accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                del model, trainer
                merged_model = _merge_adapter_in_process(temp_dir, final_output_dir)
                _save_artifacts_on_main(tokenizer, processor, final_output_dir)
                if mlflow_enabled:
                    register_model_in_mlflow(merged_model, tokenizer, script_args)

        accelerator.wait_for_everyone()

    else:
        # Covers both PEFT without merge and non-PEFT models
        trainer.save_model(final_output_dir)
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            _save_artifacts_on_main(tokenizer, processor, final_output_dir)
            if mlflow_enabled:
                register_model_in_mlflow(trainer.model, tokenizer, script_args)

        accelerator.wait_for_everyone()

    logger.info("MODEL SAVE PROCESS COMPLETED SUCCESSFULLY")


def register_model_in_mlflow(
    model: AutoModelForCausalLM, tokenizer: AutoTokenizer, script_args: ScriptArguments
) -> None:
    """Register the model in MLflow."""
    logger.info(f"MLflow model registration under {script_args.mlflow_experiment_name}")

    try:
        params = {"top_p": 0.9, "temperature": 0.2, "max_new_tokens": 1024 * 4}
        signature = infer_signature("inputs", "generated_text", params=params)

        mlflow.transformers.log_model(
            transformers_model={"model": model, "tokenizer": tokenizer},
            signature=signature,
            name="model",
            task="text-generation",
            registered_model_name=f"model-{os.environ.get('MLFLOW_RUN_NAME', '').split('Fine-tuning-')[-1]}",
        )
    except Exception as e:
        logger.error(f"Error registering model in MLflow: {e}")
        raise


def calculate_optimal_max_length(
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    script_args: ScriptArguments,
    sample_size: int = 1000,
    percentile: float = 0.95,
    max_absolute_length: int = 32768,  # Hard cap to prevent extreme outliers
) -> int:
    """Calculate optimal max_length by tokenizing a sample of the dataset."""
    sample_indices = torch.randperm(len(dataset))[: min(sample_size, len(dataset))]
    sample_data = dataset.select(sample_indices)

    token_lengths = []
    outliers_removed = 0

    for sample in sample_data:
        text = sample[script_args.text_field]

        # Skip extremely long samples that cause NaN
        if len(text) > 100000:  # Character-level filter
            outliers_removed += 1
            continue

        tokens = tokenizer(text, add_special_tokens=True)["input_ids"]
        token_len = len(tokens)

        # Cap at max_absolute_length
        if token_len > max_absolute_length:
            outliers_removed += 1
            continue

        token_lengths.append(token_len)

    if not token_lengths:
        logger.warning("No valid samples found, using default max_length=2048")
        return 2048

    avg_length = sum(token_lengths) / len(token_lengths)
    max_length = int(sorted(token_lengths)[int(percentile * len(token_lengths))])

    logger.info(
        f"Analyzed {len(token_lengths)} samples ({outliers_removed} outliers removed)"
    )
    logger.info(f"Average token length: {avg_length:.1f}")
    logger.info(f"{percentile*100}th percentile token length: {max_length}")

    return max_length


def prepare_dataset(
    tokenizer: AutoTokenizer,
    script_args: ScriptArguments,
    train_ds: Dataset,
    test_ds: Optional[Dataset] = None,
):
    """Prepare the dataset for training with optimal tokenization."""

    if script_args.apply_truncation:
        # Filter extreme outliers before calculating max_length
        logger.info(f"Original training samples: {len(train_ds)}")
        lengths = [len(x[script_args.text_field]) for x in train_ds]
        threshold = sorted(lengths)[int(0.995 * len(lengths))]
        logger.info(f"Filtering samples > {threshold} characters (99.5th percentile)")
        train_ds = train_ds.filter(
            lambda x: len(x[script_args.text_field]) <= threshold
        )
        logger.info(f"Filtered training samples: {len(train_ds)}")

        if test_ds is not None:
            logger.info(f"Original test samples: {len(test_ds)}")
            test_ds = test_ds.filter(
                lambda x: len(x[script_args.text_field]) <= threshold
            )
            logger.info(f"Filtered test samples: {len(test_ds)}")

        if script_args.max_length is None:
            max_length = calculate_optimal_max_length(tokenizer, train_ds, script_args)
        else:
            max_length = script_args.max_length
    else:
        max_length = None

    logger.info(f"Using max_length: {max_length}")
    logger.info(f"Truncation enabled: {script_args.apply_truncation}")

    lm_train_dataset = train_ds.map(
        lambda sample: tokenizer(
            sample[script_args.text_field],
            padding=False,
            truncation=script_args.apply_truncation,
            max_length=max_length if script_args.apply_truncation else None,
        ),
        remove_columns=list(train_ds.features),
        batched=True,
        batch_size=1000,
    )

    if test_ds is not None:
        lm_test_dataset = test_ds.map(
            lambda sample: tokenizer(
                sample[script_args.text_field],
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
    """Train the model using centralized configuration."""
    set_seed(training_args.seed)

    # Create centralized config builder
    config_builder = ModelConfigBuilder(script_args, training_args)
    mlflow_enabled = is_mlflow_enabled(script_args)

    if script_args.token is not None:
        os.environ.update({"HF_TOKEN": script_args.token})
        if dist.is_initialized():
            logger.info("Waiting for all processes after setting HF token")
            dist.barrier()

    if script_args.use_snapshot_download:
        download_model(script_args.model_id)
        if dist.is_initialized():
            logger.info("Waiting for all processes after model download")
            dist.barrier()
        script_args.model_id = "/tmp/tmp_folder"

    # Load model, tokenizer, and processor using centralized config
    model = load_model(config_builder, script_args)
    tokenizer = load_tokenizer(script_args)
    processor = load_processor(script_args)

    train_ds, test_ds = prepare_dataset(tokenizer, script_args, train_ds, test_ds)

    if script_args.use_peft:
        model = apply_lora_config(model, script_args)

    callbacks = setup_wandb(script_args)
    if script_args.early_stopping:
        if callbacks is None:
            callbacks = []
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))

        training_args.load_best_model_at_end = True
        training_args.metric_for_best_model = "eval_loss"
        training_args.greater_is_better = False
    trainer = setup_trainer(
        model, tokenizer, train_ds, config_builder, test_ds, callbacks
    )

    if trainer.accelerator.is_main_process:
        trainer.model.print_trainable_parameters()

    if script_args.checkpoint_dir is not None:
        os.makedirs(script_args.checkpoint_dir, exist_ok=True)

        original_output_dir = training_args.output_dir
        training_args.output_dir = script_args.checkpoint_dir
    else:
        original_output_dir = training_args.output_dir

    # Start training
    if mlflow_enabled:
        logger.info(f"MLflow tracking under {script_args.mlflow_experiment_name}")
        mlflow.set_system_metrics_node_id(
            f"node_{trainer.accelerator.process_index // torch.cuda.device_count()}"
        )
        if trainer.accelerator.is_main_process:
            mlflow.start_run(run_name=os.environ.get("MLFLOW_RUN_NAME", None))
            mlflow.log_params(
                {
                    "total_gpus": trainer.accelerator.num_processes,
                    "nodes": trainer.accelerator.num_processes
                    // torch.cuda.device_count(),
                    "gpus_per_node": torch.cuda.device_count(),
                }
            )
            try:
                train_dataset_mlflow = mlflow.data.from_pandas(
                    train_ds.to_pandas(), name="train_dataset"
                )
                mlflow.log_input(train_dataset_mlflow, context="train")
            except Exception as e:
                logger.warning(f"Failed to log dataset to MLflow: {e}")

    if (
        get_last_checkpoint(script_args.checkpoint_dir) is not None
        and script_args.use_checkpoints
    ):
        train_result = trainer.train(resume_from_checkpoint=True)
    else:
        train_result = trainer.train()

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_ds)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    save_model(
        trainer,
        model,
        tokenizer,
        processor,
        script_args,
        trainer.accelerator,
        mlflow_enabled,
        original_output_dir,
    )
    trainer.accelerator.wait_for_everyone()


def load_datasets(script_args: ScriptArguments) -> Tuple[Dataset, Optional[Dataset]]:
    """Load training and test datasets."""
    try:
        logger.info(f"Loading training dataset from {script_args.train_dataset_path}")

        if script_args.train_dataset_path.endswith(
            ".jsonl"
        ) or script_args.train_dataset_path.endswith(".json"):
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
            if script_args.val_dataset_path.endswith(
                ".jsonl"
            ) or script_args.val_dataset_path.endswith(".json"):
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
    parser = TrlParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_and_config()

    set_custom_env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    setup_mlflow(script_args)

    train_ds, test_ds = load_datasets(script_args)
    train(script_args, training_args, train_ds, test_ds)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
