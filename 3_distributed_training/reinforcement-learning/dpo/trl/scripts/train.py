from accelerate import Accelerator
from dataclasses import dataclass, field
from datasets import Dataset, load_dataset
import contextlib
import datetime
from huggingface_hub import snapshot_download
import logging
import mlflow
from mlflow.models import infer_signature
import os
from peft import AutoPeftModelForCausalLM, LoraConfig
import torch
import torch.distributed as dist
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Mxfp4Config,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from trl import DPOConfig, DPOTrainer, TrlParser
from typing import Any, Dict, List, Optional, Tuple
from distutils.util import strtobool

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
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
    use_checkpoints: bool = field(
        default=False, metadata={"help": "Whether to use checkpointing"}
    )
    load_in_4bit: bool = field(
        default=True, metadata={"help": "Load model in 4-bit quantization"}
    )
    lora_r: Optional[int] = field(default=8, metadata={"help": "LoRA r"})
    lora_alpha: Optional[int] = field(default=16, metadata={"help": "LoRA alpha"})
    lora_dropout: Optional[float] = field(
        default=0.1, metadata={"help": "LoRA dropout"}
    )
    merge_weights: Optional[bool] = field(
        default=False, metadata={"help": "Merge adapter with base model"}
    )
    mlflow_uri: Optional[str] = field(
        default=None, metadata={"help": "MLflow tracking URI"}
    )
    mlflow_experiment_name: Optional[str] = field(
        default=None, metadata={"help": "MLflow experiment name"}
    )
    model_id: str = field(default=None, metadata={"help": "Model ID"})
    token: str = field(default=None, metadata={"help": "Hugging Face API token"})
    train_dataset_path: Optional[str] = field(
        default=None, metadata={"help": "Training dataset path"}
    )
    val_dataset_path: Optional[str] = field(
        default=None, metadata={"help": "Validation dataset path"}
    )
    target_modules: Optional[List[str]] = field(
        default=None, metadata={"help": "Target modules for LoRA"}
    )
    torch_dtype: Optional[str] = field(default="auto", metadata={"help": "Torch dtype"})
    use_mxfp4: bool = field(default=False, metadata={"help": "Use MXFP4 quantization"})
    use_peft: bool = field(default=True, metadata={"help": "Use PEFT for training"})
    use_snapshot_download: bool = field(
        default=True, metadata={"help": "Use snapshot download"}
    )


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


def download_model(model_name):
    logger.info(f"Downloading model {model_name}")
    os.makedirs("/tmp/tmp_folder", exist_ok=True)
    snapshot_download(repo_id=model_name, local_dir="/tmp/tmp_folder")
    logger.info(f"Model {model_name} downloaded to /tmp/tmp_folder")


def set_custom_env(env_vars: Dict[str, str]) -> None:
    """Set custom environment variables."""
    os.environ.update(env_vars)
    for key, value in env_vars.items():
        logger.info(f"  {key}: {value}")


def is_mlflow_enabled(script_args: ScriptArguments) -> bool:
    return script_args.mlflow_uri and script_args.mlflow_experiment_name


def setup_mlflow(script_args: ScriptArguments) -> None:
    if not is_mlflow_enabled(script_args):
        return
    logger.info("Initializing MLflow")
    mlflow.enable_system_metrics_logging()
    mlflow.autolog()
    mlflow.set_tracking_uri(script_args.mlflow_uri)
    mlflow.set_experiment(script_args.mlflow_experiment_name)
    formatted_datetime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    set_custom_env(
        {
            "MLFLOW_RUN_NAME": f"DPO-{formatted_datetime}",
            "MLFLOW_EXPERIMENT_NAME": script_args.mlflow_experiment_name,
        }
    )


def load_model_and_tokenizer(script_args: ScriptArguments, training_args: DPOConfig):
    torch_dtype = (
        torch.bfloat16
        if training_args.bf16
        else (
            torch.float32
            if script_args.torch_dtype == "auto"
            else getattr(torch, script_args.torch_dtype)
        )
    )

    use_deepspeed = strtobool(os.environ.get("ACCELERATE_USE_DEEPSPEED", "false"))

    model_kwargs = {
        "attn_implementation": script_args.attn_implementation,
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
        "cache_dir": "/tmp/.cache",
    }

    if not use_deepspeed:
        model_kwargs["low_cpu_mem_usage"] = True

    if script_args.load_in_4bit:
        if script_args.use_mxfp4:
            model_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
            logger.info("Using MXFP4 quantization")
        else:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_storage=torch_dtype,
            )
            logger.info("Using BitsAndBytes quantization")

    model = AutoModelForCausalLM.from_pretrained(script_args.model_id, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    return model, tokenizer


def register_model_in_mlflow(
    model: AutoModelForCausalLM, tokenizer: AutoTokenizer, script_args: ScriptArguments
) -> None:
    """Register the model in MLflow."""
    logger.info(f"MLflow model registration under {script_args.mlflow_experiment_name}")
    try:
        params = {"top_p": 0.9, "temperature": 0.2, "max_new_tokens": 4096}
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


def save_model(
    trainer: DPOTrainer,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    script_args: ScriptArguments,
    training_args: DPOConfig,
    accelerator,
    mlflow_enabled: bool,
    final_output_dir: str,
) -> None:
    """Save the trained model."""
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    if script_args.use_peft and script_args.merge_weights:
        temp_dir = "/tmp/model"
        trainer.model.save_pretrained(temp_dir, safe_serialization=False)

        if accelerator.is_main_process:
            with gpu_memory_manager():
                del model, trainer
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
                tokenizer.save_pretrained(final_output_dir)

                if mlflow_enabled:
                    register_model_in_mlflow(model, tokenizer, script_args)
    else:
        trainer.save_model(final_output_dir)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(final_output_dir)
            if mlflow_enabled:
                register_model_in_mlflow(trainer.model, tokenizer, script_args)


def is_mlflow_enabled(script_args: ScriptArguments) -> bool:
    return script_args.mlflow_uri and script_args.mlflow_experiment_name


def setup_mlflow(script_args: ScriptArguments) -> None:
    if not is_mlflow_enabled(script_args):
        return
    logger.info("Initializing MLflow")
    mlflow.enable_system_metrics_logging()
    mlflow.autolog()
    mlflow.set_tracking_uri(script_args.mlflow_uri)
    mlflow.set_experiment(script_args.mlflow_experiment_name)
    formatted_datetime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    os.environ.update(
        {
            "MLFLOW_RUN_NAME": f"DPO-{formatted_datetime}",
            "MLFLOW_EXPERIMENT_NAME": script_args.mlflow_experiment_name,
        }
    )


class ModelConfigBuilder:
    """Centralized model configuration builder."""

    def __init__(self, script_args: ScriptArguments, training_args: DPOConfig):
        self.script_args = script_args
        self.training_args = training_args
        self._torch_dtype = None
        self._quantization_config = None
        self._use_deepspeed = None

    @property
    def torch_dtype(self) -> torch.dtype:
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
        if self._use_deepspeed is None:
            self._use_deepspeed = strtobool(
                os.environ.get("ACCELERATE_USE_DEEPSPEED", "false")
            )
        return self._use_deepspeed

    @property
    def quantization_config(self) -> Optional[Any]:
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
        model_kwargs = {
            "attn_implementation": self.script_args.attn_implementation,
            "torch_dtype": self.torch_dtype,
            "use_cache": not self.training_args.gradient_checkpointing,
            "trust_remote_code": True,
            "cache_dir": "/tmp/.cache",
        }
        if not self.use_deepspeed:
            model_kwargs["low_cpu_mem_usage"] = True
        if self.quantization_config is not None:
            model_kwargs["quantization_config"] = self.quantization_config
        return model_kwargs


def load_model(
    config_builder: ModelConfigBuilder, script_args: ScriptArguments
) -> AutoModelForCausalLM:
    """Load model using centralized configuration."""
    model_kwargs = config_builder.build_model_kwargs()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            script_args.model_id, **model_kwargs
        )
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


def train(
    script_args: ScriptArguments,
    training_args: DPOConfig,
    train_ds: Dataset,
    test_ds: Optional[Dataset],
):
    set_seed(training_args.seed)
    mlflow_enabled = is_mlflow_enabled(script_args)

    if script_args.token:
        os.environ["HF_TOKEN"] = script_args.token
        if dist.is_initialized():
            logger.info("Waiting for all processes after setting HF token")
            dist.barrier()

    if script_args.use_snapshot_download:
        download_model(script_args.model_id)
        if dist.is_initialized():
            logger.info("Waiting for all processes after model download")
            dist.barrier()
        script_args.model_id = "/tmp/tmp_folder"

    # Create centralized config builder
    config_builder = ModelConfigBuilder(script_args, training_args)

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

    # Set report_to based on enabled tracking
    report_to = []
    if mlflow_enabled:
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
    )

    if trainer.accelerator.is_main_process:
        logger.info(
            f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )

    # Handle checkpoint directory
    if script_args.checkpoint_dir:
        os.makedirs(script_args.checkpoint_dir, exist_ok=True)
        original_output_dir = training_args.output_dir
        training_args.output_dir = script_args.checkpoint_dir
    else:
        original_output_dir = training_args.output_dir

    # Start training with MLflow tracking
    if mlflow_enabled:
        # All processes set their node ID for system metrics
        mlflow.set_system_metrics_node_id(
            f"node_{trainer.accelerator.process_index // torch.cuda.device_count()}"
        )

        # Only main process manages the MLflow run
        if trainer.accelerator.is_main_process:
            mlflow.start_run(run_name=os.environ.get("MLFLOW_RUN_NAME"))
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

    # Train with checkpoint resume support
    if (
        script_args.checkpoint_dir
        and get_last_checkpoint(script_args.checkpoint_dir)
        and script_args.use_checkpoints
    ):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    save_model(
        trainer,
        model,
        tokenizer,
        script_args,
        training_args,
        trainer.accelerator,
        mlflow_enabled,
        original_output_dir,
    )
    trainer.accelerator.wait_for_everyone()


def main() -> None:
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
