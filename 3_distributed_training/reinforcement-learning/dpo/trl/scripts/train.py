from accelerate import Accelerator
from dataclasses import dataclass, field
from datasets import Dataset, load_dataset
import datetime
from huggingface_hub import snapshot_download
import json
import logging
import mlflow
from mlflow.models import infer_signature
import os
from peft import (
    AutoPeftModelForCausalLM,
    LoraConfig,
)
import subprocess
import sys
import textwrap
import torch
import torch.distributed as dist
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Mxfp4Config,
    set_seed,
)
from trl import DPOConfig, DPOTrainer, TrlParser
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

    attn_implementation: Optional[str] = field(
        default="flash_attention_2", metadata={"help": "Attention implementation"}
    )
    auto_calculate_lengths: bool = field(
        default=False,
        metadata={
            "help": "Auto-calculate max_length, max_prompt_length, max_completion_length from dataset"
        },
    )
    checkpoint_dir: str = field(default=None, metadata={"help": "Checkpoint directory"})
    deserialize_messages: bool = field(
        default=False,
        metadata={"help": "Deserialize JSON-encoded prompt, chosen, rejected fields"},
    )
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
        default=None, metadata={"help": "Model ID to use for DPO training"}
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

    def __init__(self, script_args: ScriptArguments, training_args: DPOConfig):
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
                "use_cache": not self.training_args.gradient_checkpointing,
                "trust_remote_code": True,
                "cache_dir": "/tmp/.cache",
            }
        else:
            model_kwargs = {
                "torch_dtype": self.torch_dtype,
                "use_cache": not self.training_args.gradient_checkpointing,
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


def patch_dpo_trainer_dtype(trainer):
    """Patch DPOTrainer to fix input_ids dtype issue with tool calling.

    This is a known bug in DPOTrainer where concatenated_forward incorrectly
    casts input_ids to the model's dtype (bfloat16) instead of keeping them as long.
    See: https://github.com/huggingface/trl/issues/2101
    """
    original_concatenated_forward = trainer.concatenated_forward

    def safe_concatenated_forward(model, batch, is_ref_model=False):
        # Fix all tensor dtypes in the batch before forward pass
        for key in batch.keys():
            if batch[key] is not None and isinstance(batch[key], torch.Tensor):
                # Input IDs, labels, and attention masks must be long
                if any(x in key for x in ["input_ids", "labels", "attention_mask"]):
                    if batch[key].dtype != torch.long:
                        batch[key] = batch[key].long()

        return original_concatenated_forward(model, batch, is_ref_model)

    trainer.concatenated_forward = safe_concatenated_forward
    return trainer


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
            "MLFLOW_RUN_NAME": f"DPO-{formatted_datetime}",
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


def ensure_uniform_dtype(
    model: AutoModelForCausalLM, target_dtype: torch.dtype
) -> AutoModelForCausalLM:
    """Ensure all model parameters have uniform dtype for FSDP compatibility.

    This is especially important for MoE (Mixture of Experts) models like Nemotron
    where router/gate modules may initialize in float32 even when bfloat16 is specified.
    """
    mixed_dtypes = set()
    for name, param in model.named_parameters():
        if param.dtype != target_dtype:
            mixed_dtypes.add((name, param.dtype))

    if mixed_dtypes:
        logger.warning(
            f"Found {len(mixed_dtypes)} parameters with non-uniform dtype. Converting to {target_dtype}"
        )
        for name, dtype in list(mixed_dtypes)[:5]:  # Log first 5 examples
            logger.warning(f"  - {name}: {dtype}")
        if len(mixed_dtypes) > 5:
            logger.warning(f"  ... and {len(mixed_dtypes) - 5} more")

        # Cast all parameters to target dtype
        model = model.to(target_dtype)

    return model


def load_model(
    config_builder: ModelConfigBuilder, script_args: ScriptArguments
) -> AutoModelForCausalLM:
    """Load model using centralized configuration."""
    model_kwargs = config_builder.build_model_kwargs()

    try:
        model = AutoModelForCausalLM.from_pretrained(
            script_args.model_id, **model_kwargs
        )

        # # Ensure uniform dtype for FSDP compatibility (critical for MoE models like Nemotron)
        # if not script_args.load_in_4bit:
        #     model = ensure_uniform_dtype(model, config_builder.torch_dtype)

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


def extract_tools_from_dataset(dataset: Dataset) -> Optional[List[Dict]]:
    """Extract tools from first sample if available."""
    if "tools" in dataset.column_names and dataset[0]["tools"]:
        return dataset[0]["tools"]
    return None


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


def _detect_distributed_strategy(trainer: DPOTrainer) -> Tuple[bool, bool]:
    """Detect whether DeepSpeed or FSDP is active."""
    use_deepspeed = (
        hasattr(trainer.accelerator.state, "deepspeed_plugin")
        and trainer.accelerator.state.deepspeed_plugin is not None
    )
    use_fsdp = trainer.is_fsdp_enabled
    return use_deepspeed, use_fsdp


def save_model(
    trainer: DPOTrainer,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
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
                tokenizer.save_pretrained(final_output_dir)
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
                tokenizer.save_pretrained(final_output_dir)
                if mlflow_enabled:
                    register_model_in_mlflow(merged_model, tokenizer, script_args)

        accelerator.wait_for_everyone()

    else:
        # Covers both PEFT without merge and non-PEFT models
        trainer.save_model(final_output_dir)
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            tokenizer.save_pretrained(final_output_dir)
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
            registered_model_name=f"model-{os.environ.get('MLFLOW_RUN_NAME', '').split('DPO-')[-1]}",
        )
    except Exception as e:
        logger.error(f"Error registering model in MLflow: {e}")
        raise


def calculate_optimal_dpo_lengths(
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    sample_size: int = 1000,
    percentile: float = 0.95,
) -> Tuple[int, int, int]:
    """Calculate optimal max_length, max_prompt_length, and max_completion_length for DPO."""
    sample_indices = torch.randperm(len(dataset))[: min(sample_size, len(dataset))]
    sample_data = dataset.select(sample_indices)

    prompt_lengths = []
    chosen_lengths = []
    rejected_lengths = []
    total_lengths = []

    for sample in sample_data:
        # Calculate prompt length (user messages)
        prompt_text = ""
        if "system" in sample and sample["system"]:
            prompt_text += sample["system"] + "\n"
        for msg in sample["chosen"]:
            if msg["role"] == "user":
                prompt_text += msg["content"]

        prompt_tokens = tokenizer(prompt_text, add_special_tokens=True)["input_ids"]
        prompt_lengths.append(len(prompt_tokens))

        # Calculate chosen completion length
        chosen_text = ""
        for msg in sample["chosen"]:
            if msg["role"] == "assistant":
                chosen_text += msg["content"]
        chosen_tokens = tokenizer(chosen_text, add_special_tokens=False)["input_ids"]
        chosen_lengths.append(len(chosen_tokens))

        # Calculate rejected completion length
        rejected_text = ""
        for msg in sample["rejected"]:
            if msg["role"] == "assistant":
                rejected_text += msg["content"]
        rejected_tokens = tokenizer(rejected_text, add_special_tokens=False)[
            "input_ids"
        ]
        rejected_lengths.append(len(rejected_tokens))

        # Total length (prompt + max of chosen/rejected)
        total_lengths.append(
            len(prompt_tokens) + max(len(chosen_tokens), len(rejected_tokens))
        )

    # Calculate percentiles
    max_prompt_length = int(
        sorted(prompt_lengths)[int(percentile * len(prompt_lengths))]
    )
    max_completion_length = int(
        sorted(chosen_lengths + rejected_lengths)[
            int(percentile * len(chosen_lengths + rejected_lengths))
        ]
    )
    max_length = int(sorted(total_lengths)[int(percentile * len(total_lengths))])

    logger.info(f"Analyzed {len(sample_data)} samples")
    logger.info(
        f"Average prompt length: {sum(prompt_lengths) / len(prompt_lengths):.1f}"
    )
    logger.info(f"{percentile*100}th percentile prompt length: {max_prompt_length}")
    logger.info(
        f"Average completion length: {sum(chosen_lengths + rejected_lengths) / len(chosen_lengths + rejected_lengths):.1f}"
    )
    logger.info(
        f"{percentile*100}th percentile completion length: {max_completion_length}"
    )
    logger.info(f"{percentile*100}th percentile total length: {max_length}")

    return max_length, max_prompt_length, max_completion_length


def load_json_file(file_path: str) -> List[Dict]:
    """Load JSON or JSONL file manually to avoid schema inference issues."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        # Try to parse as JSON array first
        if content.startswith("["):
            data = json.loads(content)
        else:
            # Parse as JSONL (one JSON object per line)
            for line in content.split("\n"):
                if line.strip():
                    data.append(json.loads(line))
    return data


def deserialize_and_load(data: List[Dict]) -> Dataset:
    """Deserialize JSON-encoded fields and create Dataset."""
    for record in data:
        if "prompt" in record and isinstance(record["prompt"], str):
            record["prompt"] = json.loads(record["prompt"])
        if "chosen" in record and isinstance(record["chosen"], str):
            record["chosen"] = json.loads(record["chosen"])
        if "rejected" in record and isinstance(record["rejected"], str):
            record["rejected"] = json.loads(record["rejected"])
    return Dataset.from_list(data)


def load_datasets(script_args: ScriptArguments) -> Tuple[Dataset, Optional[Dataset]]:
    """Load training and test datasets."""
    try:
        logger.info(f"Loading training dataset from {script_args.train_dataset_path}")

        # Determine the file path
        if script_args.train_dataset_path.endswith(
            ".jsonl"
        ) or script_args.train_dataset_path.endswith(".json"):
            train_file = script_args.train_dataset_path
        else:
            train_file = os.path.join(script_args.train_dataset_path, "dataset.json")

        # Load based on deserialize_messages flag
        if script_args.deserialize_messages:
            logger.info("Loading and deserializing JSON-encoded message fields")
            train_data = load_json_file(train_file)
            train_ds = deserialize_and_load(train_data)
        else:
            train_ds = load_dataset("json", data_files=train_file, split="train")

        test_ds = None
        if script_args.val_dataset_path:
            logger.info(f"Loading test dataset from {script_args.val_dataset_path}")

            if script_args.val_dataset_path.endswith(
                ".jsonl"
            ) or script_args.val_dataset_path.endswith(".json"):
                val_file = script_args.val_dataset_path
            else:
                val_file = os.path.join(script_args.val_dataset_path, "dataset.json")

            if script_args.deserialize_messages:
                logger.info(
                    "Loading and deserializing JSON-encoded message fields for test dataset"
                )
                test_data = load_json_file(val_file)
                test_ds = deserialize_and_load(test_data)
            else:
                test_ds = load_dataset("json", data_files=val_file, split="train")

        return train_ds, test_ds
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        raise


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

    # Load model and tokenizer using centralized config
    model = load_model(config_builder, script_args)
    tokenizer = load_tokenizer(script_args)

    # Auto-calculate lengths if enabled
    if script_args.auto_calculate_lengths:
        logger.info("Auto-calculating optimal lengths from dataset...")
        max_length, max_prompt_length, max_completion_length = (
            calculate_optimal_dpo_lengths(tokenizer, train_ds)
        )
        training_args.max_length = max_length
        training_args.max_prompt_length = max_prompt_length
        training_args.max_completion_length = max_completion_length
        logger.info(
            f"Set max_length={max_length}, max_prompt_length={max_prompt_length}, max_completion_length={max_completion_length}"
        )

    # Extract tools from dataset if available
    tools = extract_tools_from_dataset(train_ds)
    if tools:
        logger.info(f"Found {len(tools)} tools in dataset")
        training_args.tools = tools

    # Configure PEFT
    peft_config = None
    if script_args.use_peft:
        peft_config = LoraConfig(
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

    callbacks = setup_wandb(script_args)
    if script_args.early_stopping:
        if callbacks is None:
            callbacks = []
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))

        training_args.load_best_model_at_end = True
        training_args.metric_for_best_model = "eval_loss"
        training_args.greater_is_better = False

    # Apply trainer kwargs from centralized config
    trainer_kwargs = config_builder.build_trainer_kwargs()
    for key, value in trainer_kwargs.items():
        setattr(training_args, key, value)

    # Set report_to based on enabled tracking services
    report_to = []
    if os.environ.get("WANDB_DISABLED", "false").lower() != "true":
        report_to.append("wandb")
    if is_mlflow_enabled(script_args):
        report_to.append("mlflow")
    training_args.report_to = report_to

    # Initialize DPO trainer
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        peft_config=peft_config,
        callbacks=callbacks,
    )

    # Patch trainer to fix input_ids dtype bug with tool calling
    trainer = patch_dpo_trainer_dtype(trainer)

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
        script_args.checkpoint_dir
        and get_last_checkpoint(script_args.checkpoint_dir) is not None
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
        script_args,
        trainer.accelerator,
        mlflow_enabled,
        original_output_dir,
    )
    trainer.accelerator.wait_for_everyone()


def main() -> None:
    """Main function to parse arguments and start training."""
    parser = TrlParser((ScriptArguments, DPOConfig))
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
