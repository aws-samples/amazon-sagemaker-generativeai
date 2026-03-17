from accelerate import Accelerator
import base64
from dataclasses import dataclass, field
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
import datetime
from huggingface_hub import snapshot_download
import io
import json
import logging
import mlflow
from mlflow.models import infer_signature
import os
from peft import (
    AutoPeftModelForCausalLM,
    LoraConfig,
    get_peft_model,
)
from PIL import Image
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

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    try:
        from transformers import AutoModelForVision2Seq as AutoModelForImageTextToText
    except ImportError:
        AutoModelForImageTextToText = None
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
    modality_type: str = field(
        default="text",
        metadata={
            "help": (
                "Input modality: 'text' (text-only, default), "
                "'image' (image+text multi-modal). "
                "When set to 'image', the script uses a vision-aware data collator "
                "and loads the model with AutoModelForVision2Seq."
            )
        },
    )
    model_id: str = field(
        default=None, metadata={"help": "Model ID to use for SFT training"}
    )
    dataset_format: str = field(
        default="auto",
        metadata={
            "help": (
                "Dataset format: 'auto' (detect from columns), 'text' (pre-rendered text column), "
                "'messages' (conversational format - applies chat template), 'pretokenized' (input_ids column)"
            )
        },
    )
    messages_field: str = field(
        default="messages",
        metadata={
            "help": "Name of the messages field for conversational datasets (e.g., 'messages', 'conversation')"
        },
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
    patch_peft_fsdp_auto_wrap_policy: bool = field(
        default=False,
        metadata={
            "help": (
                "Patch PEFT's FSDP auto-wrap policy for architectures PEFT doesn't "
                "recognize (e.g. Qwen3.5). FSDP + LoRA only."
            )
        },
    )
    cast_parameters_to_uniform_dtype: bool = field(
        default=False,
        metadata={
            "help": (
                "Cast all model parameters to uniform dtype. Required for models "
                "with mixed float32/bfloat16 parameters (e.g. Qwen3.5 inv_freq). "
                "Needed for both FSDP and DeepSpeed."
            )
        },
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


def load_model(config_builder: ModelConfigBuilder, script_args: ScriptArguments):
    """Load model using centralized configuration.

    When modality_type is 'image', loads with AutoModelForImageTextToText.
    Otherwise loads with AutoModelForCausalLM.
    """
    model_kwargs = config_builder.build_model_kwargs()

    try:
        if (
            script_args.modality_type == "image"
            and AutoModelForImageTextToText is not None
        ):
            model = AutoModelForImageTextToText.from_pretrained(
                script_args.model_id, **model_kwargs
            )
            logger.info(f"Loaded model with {AutoModelForImageTextToText.__name__}")
        else:
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


def process_vision_info(messages: List[Dict]) -> List[Image.Image]:
    """Extract images from message content blocks.

    Supports:
    - base64 data URIs: {"type": "image_url", "image_url": {"url": "data:image/...;base64,..."}}
    - Local file paths: {"type": "image_url", "image_url": {"url": "file:///path/to/image.png"}}
    - Plain image type markers: {"type": "image"} (image data must come from dataset column)
    """
    images = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for element in content:
            if not isinstance(element, dict):
                continue
            if element.get("type") == "image_url":
                url = (element.get("image_url") or {}).get("url", "")
                if url.startswith("data:image"):
                    b64_data = url.split(",", 1)[1]
                    img = Image.open(io.BytesIO(base64.b64decode(b64_data))).convert(
                        "RGB"
                    )
                    images.append(img)
                elif url.startswith("file://"):
                    path = url.replace("file://", "")
                    images.append(Image.open(path).convert("RGB"))
    return images


def collect_images(sample: Dict, messages: List[Dict]) -> Optional[List[Image.Image]]:
    """Collect images from HF dataset columns or inline base64 in messages.

    Priority: 'images' column > 'image' column > inline base64 in messages.
    Returns None if no images are found (text-only sample).
    """
    if "images" in sample and sample["images"]:
        imgs = sample["images"]
        return imgs if isinstance(imgs, list) else [imgs]
    if "image" in sample and sample["image"]:
        img = sample["image"]
        return [img] if not isinstance(img, list) else img
    inline = process_vision_info(messages)
    return inline if inline else None


def get_vision_special_token_ids(processor) -> List[int]:
    """Get token IDs for vision special tokens that should be masked in labels.

    Scans the tokenizer's special tokens for vision-related markers
    (e.g., <image>, <|vision_start|>, <|image_pad|>) and returns their IDs.
    """
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    substrings = (
        "image",
        "vision",
        "video",
        "boi",
        "eoi",
        "<image>",
        "<video>",
        "image_pad",
    )

    special_vals = []
    for v in (tokenizer.special_tokens_map or {}).values():
        if isinstance(v, str):
            special_vals.append(v)
        elif isinstance(v, (list, tuple)):
            special_vals.extend(s for s in v if isinstance(s, str))

    addl = getattr(tokenizer, "additional_special_tokens", None)
    if isinstance(addl, (list, tuple)):
        special_vals.extend(s for s in addl if isinstance(s, str))

    token_ids = []
    for tok in special_vals:
        if any(s in tok.lower() for s in substrings):
            tid = tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid >= 0:
                token_ids.append(tid)

    return list(set(token_ids))


def patch_peft_fsdp_auto_wrap_policy():
    """Patch PEFT's fsdp_auto_wrap_policy for model architectures that PEFT doesn't recognize.

    PEFT's implementation inspects the model to find the transformer layer class but fails
    on newer architectures (e.g. Qwen3.5). This patch catches the exception and auto-detects
    the decoder layer class by scanning for modules with 'DecoderLayer' in their class name.

    This is safe to call unconditionally — if PEFT's original function works, the patch
    is a no-op pass-through.
    """
    import functools
    from torch.distributed.fsdp.wrap import (
        transformer_auto_wrap_policy,
        _or_policy,
        lambda_auto_wrap_policy,
    )
    import peft.utils.other

    _original_fsdp_auto_wrap_policy = peft.utils.other.fsdp_auto_wrap_policy

    def _patched_fsdp_auto_wrap_policy(model):
        try:
            return _original_fsdp_auto_wrap_policy(model)
        except Exception:
            base = model.base_model.model if hasattr(model, "base_model") else model
            decoder_layer_cls = None
            for _, module in base.named_modules():
                cls_name = type(module).__name__
                if "DecoderLayer" in cls_name:
                    decoder_layer_cls = type(module)
                    break
            if decoder_layer_cls is None:
                raise
            logger.info(
                f"Patched FSDP auto-wrap policy to use {decoder_layer_cls.__name__}"
            )
            from peft.tuners import PrefixEncoder, PromptEmbedding, PromptEncoder

            peft_cls = (PrefixEncoder, PromptEmbedding, PromptEncoder)
            lambda_policy = functools.partial(
                lambda_auto_wrap_policy,
                lambda_fn=lambda module: isinstance(module, peft_cls),
            )
            transformer_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={decoder_layer_cls},
            )
            return functools.partial(
                _or_policy, policies=[lambda_policy, transformer_policy]
            )

    peft.utils.other.fsdp_auto_wrap_policy = _patched_fsdp_auto_wrap_policy
    logger.info("PEFT FSDP auto-wrap policy patch applied")


def cast_parameters_to_uniform_dtype(model, target_dtype: torch.dtype) -> int:
    """Cast all model parameters to a uniform dtype for FSDP compatibility.

    Some model architectures (e.g. Qwen3.5) have parameters like inv_freq in
    rotary embeddings that remain float32 even when loaded with torch_dtype=bfloat16.
    FSDP requires all parameters to have the same dtype, and mixed dtypes also cause
    gradient checkpointing recomputation mismatches (PyTorch issue #159359).

    Returns the number of parameters that were cast.
    """
    cast_count = 0
    for name, param in model.named_parameters():
        if param.dtype != target_dtype:
            param.data = param.data.to(target_dtype)
            cast_count += 1
    if cast_count > 0:
        logger.info(
            f"Cast {cast_count} parameters from mixed dtypes to {target_dtype} for FSDP"
        )
    return cast_count


def apply_lora_config(model, script_args: ScriptArguments, is_vlm: bool = False):
    """Apply LoRA configuration to the model.

    For VLMs with target_modules='all-linear', the vision encoder is excluded
    to avoid gradient checkpointing recomputation mismatches — LoRA on vision
    encoder layers causes shape/dtype conflicts during FSDP recomputation
    (PyTorch issue #159359).
    """
    lora_kwargs = dict(
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

    if is_vlm and script_args.target_modules is None:
        vision_prefixes = [
            "visual",
            "vision_tower",
            "vision_model",
            "img_processor",
            "vpm",
        ]
        lora_kwargs["exclude_modules"] = vision_prefixes
        logger.info(
            f"VLM detected: excluding vision encoder from LoRA targets ({vision_prefixes})"
        )

    config = LoraConfig(**lora_kwargs)
    return get_peft_model(model, config)


class MultiModalDataCollator:
    """Data collator for vision-language models.

    Processes raw dataset samples at collation time: applies chat template,
    collects images, runs the processor to produce input_ids + pixel_values,
    and builds labels with proper masking for padding and vision special tokens.
    """

    def __init__(
        self,
        processor,
        script_args: ScriptArguments,
        vision_token_ids: Optional[List[int]] = None,
    ):
        self.processor = processor
        self.script_args = script_args
        self.vision_token_ids = set(vision_token_ids or [])
        self.tokenizer = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = []
        all_images = []

        for sample in features:
            messages = sample[self.script_args.messages_field]
            if isinstance(messages, str):
                messages = json.loads(messages)

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)

            images = collect_images(sample, messages)
            all_images.append(images)

        has_images = any(imgs is not None for imgs in all_images)

        if has_images:
            flat_images = []
            for imgs in all_images:
                if imgs is not None:
                    flat_images.extend(imgs)
            batch = self.processor(
                text=texts,
                images=flat_images if flat_images else None,
                return_tensors="pt",
                padding=True,
                truncation=self.script_args.apply_truncation,
                max_length=self.script_args.max_length,
            )
        else:
            batch = self.processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=self.script_args.apply_truncation,
                max_length=self.script_args.max_length,
            )

        labels = batch["input_ids"].clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        for tid in self.vision_token_ids:
            labels[labels == tid] = -100
        batch["labels"] = labels

        return batch


def setup_trainer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_ds: Dataset,
    config_builder: ModelConfigBuilder,
    test_ds: Optional[Dataset] = None,
    callbacks: Optional[List] = None,
    processor=None,
    script_args: Optional[ScriptArguments] = None,
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

    if (
        script_args is not None
        and script_args.modality_type == "image"
        and processor is not None
    ):
        vision_token_ids = get_vision_special_token_ids(processor)
        logger.info(
            f"Using multi-modal data collator (vision tokens to mask: {len(vision_token_ids)})"
        )
        data_collator = MultiModalDataCollator(
            processor=processor,
            script_args=script_args,
            vision_token_ids=vision_token_ids,
        )
    else:
        data_collator = transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False
        )

    return Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=test_ds if test_ds is not None else None,
        args=config_builder.training_args,
        callbacks=callbacks,
        data_collator=data_collator,
    )


def _was_trained_as_vlm(adapter_dir: str) -> bool:
    """Check if the adapter was trained on a VLM by inspecting adapter weight keys.

    When a model is loaded with AutoModelForImageTextToText, the language model
    layers are nested under 'language_model' (e.g. model.language_model.layers.X).
    When loaded with AutoModelForCausalLM, they are directly under 'model'
    (e.g. model.layers.X). We check the saved adapter weights for this prefix.
    """
    if AutoModelForImageTextToText is None:
        return False
    try:
        import glob

        # Check safetensors files first
        safetensor_files = glob.glob(
            os.path.join(adapter_dir, "adapter_model*.safetensors")
        )
        if safetensor_files:
            from safetensors import safe_open

            with safe_open(safetensor_files[0], framework="pt") as sf:
                for key in sf.keys():
                    if "language_model" in key:
                        return True
                return False

        # Fall back to pytorch bin files
        bin_files = glob.glob(os.path.join(adapter_dir, "adapter_model*.bin"))
        if bin_files:
            state_dict = torch.load(bin_files[0], map_location="cpu", weights_only=True)
            for key in state_dict.keys():
                if "language_model" in key:
                    return True
            return False
    except Exception as e:
        logger.warning(f"Could not inspect adapter weights: {e}")
    return False


def _is_vlm_from_config(model_id: str) -> bool:
    """Check if a model is a VLM by inspecting its config."""
    if AutoModelForImageTextToText is None:
        return False
    try:
        from transformers import AutoConfig
        from transformers.models.auto.modeling_auto import (
            MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
        )

        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        return config.model_type in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
    except Exception:
        return False


def _transplant_into_vlm(
    merged_causal_state: dict,
    base_model_id: str,
    torch_dtype: torch.dtype,
):
    """Transplant merged CausalLM weights into a full VLM to preserve vision encoder.

    When an adapter is trained with AutoModelForCausalLM on a VLM base, the merged
    language model weights use CausalLM key paths (model.layers.X...). This function
    loads the full VLM and replaces the language model weights with the merged ones,
    keeping the vision encoder and projector intact.
    """
    logger.info(
        "Transplanting merged CausalLM weights into full VLM to preserve vision encoder"
    )
    vlm_model = AutoModelForImageTextToText.from_pretrained(
        base_model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    vlm_state = vlm_model.state_dict()

    # Find the key prefix mapping between CausalLM and VLM structures.
    # CausalLM: model.layers.0... / VLM: model.language_model.layers.0...
    # We detect the VLM prefix by finding a known layer key.
    causal_layer_key = next((k for k in merged_causal_state if ".layers.0." in k), None)
    if causal_layer_key is None:
        logger.warning(
            "Could not find layer keys in merged state dict, skipping transplant"
        )
        return vlm_model

    causal_prefix = causal_layer_key.split("layers.0.")[0]  # e.g. "model."
    vlm_layer_key = next(
        (k for k in vlm_state if ".layers.0." in k and "language_model" in k), None
    )
    if vlm_layer_key is None:
        logger.warning(
            "Could not find language_model layer keys in VLM, skipping transplant"
        )
        return vlm_model

    vlm_prefix = vlm_layer_key.split("layers.0.")[0]  # e.g. "model.language_model."
    logger.info(f"Key mapping: CausalLM '{causal_prefix}*' -> VLM '{vlm_prefix}*'")

    # Replace VLM language model weights with merged CausalLM weights
    updated = 0
    for causal_key, value in merged_causal_state.items():
        if causal_key.startswith(causal_prefix):
            vlm_key = vlm_prefix + causal_key[len(causal_prefix) :]
        else:
            # Keys outside the main prefix (e.g. lm_head.weight)
            # Search for matching suffix in VLM state dict
            vlm_key = next((vk for vk in vlm_state if vk.endswith(causal_key)), None)
        if vlm_key and vlm_key in vlm_state:
            vlm_state[vlm_key] = value
            updated += 1

    logger.info(
        f"Transplanted {updated}/{len(merged_causal_state)} language model weights into VLM"
    )
    vlm_model.load_state_dict(vlm_state)
    return vlm_model


def _load_and_merge_adapter(adapter_dir: str, torch_dtype: torch.dtype):
    """Load, merge, and return the final model ready for saving.

    Handles three cases:
    1. Adapter trained on VLM (modality_type=image) → merge directly on VLM
    2. Adapter trained on CausalLM, base is VLM → merge CausalLM, transplant into VLM
    3. Adapter trained on CausalLM, base is text-only → merge directly on CausalLM
    """
    from peft import PeftConfig, PeftModel

    peft_config = PeftConfig.from_pretrained(adapter_dir)
    base_model_id = peft_config.base_model_name_or_path
    trained_as_vlm = _was_trained_as_vlm(adapter_dir)
    base_is_vlm = _is_vlm_from_config(base_model_id)

    if trained_as_vlm:
        # Case 1: direct VLM merge
        logger.info(
            f"Adapter trained on VLM: loading with {AutoModelForImageTextToText.__name__}"
        )
        base_model = AutoModelForImageTextToText.from_pretrained(
            base_model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, adapter_dir)
        return model.merge_and_unload()

    # Merge adapter as CausalLM first
    logger.info("Adapter trained on CausalLM: merging with AutoPeftModelForCausalLM")
    causal_model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_dir,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    merged_causal = causal_model.merge_and_unload()

    if base_is_vlm:
        # Case 2: transplant merged LM weights into full VLM
        merged_state = merged_causal.state_dict()
        del causal_model, merged_causal
        torch.cuda.empty_cache()
        return _transplant_into_vlm(merged_state, base_model_id, torch_dtype)

    # Case 3: text-only model, return merged CausalLM directly
    return merged_causal


def _merge_adapter_in_process(
    temp_dir: str,
    final_output_dir: str,
    torch_dtype: torch.dtype = torch.bfloat16,
):
    """Merge LoRA adapter in the current process (for FSDP/DDP)."""
    with gpu_memory_manager():
        model = _load_and_merge_adapter(temp_dir, torch_dtype)
        model.save_pretrained(
            final_output_dir, safe_serialization=True, max_shard_size="2GB"
        )
        return model


def _merge_adapter_via_subprocess(
    temp_dir: str,
    final_output_dir: str,
    torch_dtype_str: str = "bfloat16",
) -> None:
    """Merge LoRA adapter in a clean subprocess to avoid DeepSpeed env conflicts.

    Auto-detects whether the base model is a VLM from the adapter config and
    loads with the correct auto class to preserve vision encoder weights.
    """
    merge_script = textwrap.dedent(
        f"""\
        import glob
        import os
        import torch
        from peft import PeftConfig, PeftModel, AutoPeftModelForCausalLM
        from transformers import AutoConfig

        adapter_dir = "{temp_dir}"
        output_dir = "{final_output_dir}"
        dtype = getattr(torch, "{torch_dtype_str}")

        peft_config = PeftConfig.from_pretrained(adapter_dir)
        base_model_id = peft_config.base_model_name_or_path

        # Check if adapter was trained on a VLM by inspecting weight keys
        trained_as_vlm = False
        try:
            sf_files = glob.glob(os.path.join(adapter_dir, "adapter_model*.safetensors"))
            if sf_files:
                from safetensors import safe_open
                with safe_open(sf_files[0], framework="pt") as sf:
                    trained_as_vlm = any("language_model" in k for k in sf.keys())
            else:
                bin_files = glob.glob(os.path.join(adapter_dir, "adapter_model*.bin"))
                if bin_files:
                    sd = torch.load(bin_files[0], map_location="cpu", weights_only=True)
                    trained_as_vlm = any("language_model" in k for k in sd.keys())
        except Exception as e:
            print(f"Warning: could not inspect adapter weights: {{e}}")

        # Check if base model is a VLM
        base_is_vlm = False
        try:
            from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
            config = AutoConfig.from_pretrained(base_model_id, trust_remote_code=True)
            base_is_vlm = config.model_type in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
        except Exception:
            pass

        try:
            from transformers import AutoModelForImageTextToText
            vlm_auto_cls = AutoModelForImageTextToText
        except ImportError:
            try:
                from transformers import AutoModelForVision2Seq as vlm_auto_cls
            except ImportError:
                vlm_auto_cls = None

        if trained_as_vlm and vlm_auto_cls:
            # Case 1: adapter trained on VLM -> merge directly
            print(f"Adapter trained on VLM: loading with {{vlm_auto_cls.__name__}}")
            base_model = vlm_auto_cls.from_pretrained(
                base_model_id, torch_dtype=dtype,
                low_cpu_mem_usage=True, trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(base_model, adapter_dir)
            model = model.merge_and_unload()
        else:
            # Merge as CausalLM first
            print("Merging adapter as CausalLM...")
            causal_model = AutoPeftModelForCausalLM.from_pretrained(
                adapter_dir, torch_dtype=dtype,
                low_cpu_mem_usage=True, trust_remote_code=True,
            )
            model = causal_model.merge_and_unload()

            if base_is_vlm and vlm_auto_cls:
                # Case 2: CausalLM adapter on VLM base -> transplant into full VLM
                print("Base is VLM: transplanting merged weights into full VLM...")
                merged_state = model.state_dict()
                del causal_model, model
                torch.cuda.empty_cache()

                vlm_model = vlm_auto_cls.from_pretrained(
                    base_model_id, torch_dtype=dtype,
                    low_cpu_mem_usage=True, trust_remote_code=True,
                )
                vlm_state = vlm_model.state_dict()

                # Find key prefix mapping: CausalLM "model." -> VLM "model.language_model."
                causal_lk = next((k for k in merged_state if ".layers.0." in k), None)
                vlm_lk = next((k for k in vlm_state if ".layers.0." in k and "language_model" in k), None)
                if causal_lk and vlm_lk:
                    c_prefix = causal_lk.split("layers.0.")[0]
                    v_prefix = vlm_lk.split("layers.0.")[0]
                    print(f"Key mapping: '{{c_prefix}}*' -> '{{v_prefix}}*'")
                    updated = 0
                    for ck, val in merged_state.items():
                        if ck.startswith(c_prefix):
                            vk = v_prefix + ck[len(c_prefix):]
                        else:
                            vk = next((x for x in vlm_state if x.endswith(ck)), None)
                        if vk and vk in vlm_state:
                            vlm_state[vk] = val
                            updated += 1
                    print(f"Transplanted {{updated}}/{{len(merged_state)}} weights")
                    vlm_model.load_state_dict(vlm_state)
                model = vlm_model

        print("Saving merged model...")
        model.save_pretrained(
            output_dir,
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
    model_id: str = None,
) -> None:
    """Save tokenizer and processor to the output directory.

    If processor is None but the saved model is a VLM (has a config with a
    model_type registered for image-text-to-text), loads and saves the processor
    from the base model so the output is complete for multi-modal inference.
    """
    tokenizer.save_pretrained(final_output_dir)
    if processor is not None:
        processor.save_pretrained(final_output_dir)
        if (
            hasattr(processor, "image_processor")
            and processor.image_processor is not None
        ):
            processor.image_processor.save_pretrained(final_output_dir)
        if (
            hasattr(processor, "video_processor")
            and processor.video_processor is not None
        ):
            processor.video_processor.save_pretrained(final_output_dir)
    elif model_id and _is_vlm_from_config(model_id):
        # No processor provided but base model is a VLM — load processor from base model
        try:
            logger.info(
                f"No processor provided but base model is a VLM. "
                f"Loading processor from: {model_id}"
            )
            base_processor = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=True
            )
            base_processor.save_pretrained(final_output_dir)
            # Also save sub-processors as separate files for compatibility (e.g. Ollama)
            if (
                hasattr(base_processor, "image_processor")
                and base_processor.image_processor is not None
            ):
                base_processor.image_processor.save_pretrained(final_output_dir)
            if (
                hasattr(base_processor, "video_processor")
                and base_processor.video_processor is not None
            ):
                base_processor.video_processor.save_pretrained(final_output_dir)
        except Exception as e:
            logger.warning(f"Could not auto-save processor for VLM: {e}")


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
                dtype_str = (
                    script_args.torch_dtype
                    if script_args.torch_dtype not in ["auto", None]
                    else "bfloat16"
                )
                _merge_adapter_via_subprocess(
                    temp_dir,
                    final_output_dir,
                    torch_dtype_str=dtype_str,
                )
                _save_artifacts_on_main(
                    tokenizer, processor, final_output_dir, script_args.model_id
                )
                if mlflow_enabled:
                    logger.info(
                        "Skipping MLflow registration (model merged in subprocess)"
                    )
        else:
            trainer.model.save_pretrained(temp_dir, safe_serialization=False)
            accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                del model, trainer
                save_dtype = (
                    getattr(torch, script_args.torch_dtype)
                    if script_args.torch_dtype not in ["auto", None]
                    else torch.bfloat16
                )
                merged_model = _merge_adapter_in_process(
                    temp_dir,
                    final_output_dir,
                    torch_dtype=save_dtype,
                )
                _save_artifacts_on_main(
                    tokenizer, processor, final_output_dir, script_args.model_id
                )
                if mlflow_enabled:
                    register_model_in_mlflow(merged_model, tokenizer, script_args)

        accelerator.wait_for_everyone()

    else:
        # Covers both PEFT without merge and non-PEFT models
        trainer.save_model(final_output_dir)
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            _save_artifacts_on_main(
                tokenizer, processor, final_output_dir, script_args.model_id
            )
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


def detect_dataset_format(dataset: Dataset, script_args: ScriptArguments) -> str:
    """Detect dataset format from column names.

    Returns one of: 'pretokenized', 'messages', 'text'.
    """
    if script_args.dataset_format != "auto":
        fmt = script_args.dataset_format
        logger.info(f"Using explicitly configured dataset format: {fmt}")
        return fmt

    columns = dataset.column_names

    if "input_ids" in columns:
        logger.info(
            "Auto-detected dataset format: pretokenized (input_ids column found)"
        )
        return "pretokenized"

    if script_args.messages_field in columns:
        logger.info(
            f"Auto-detected dataset format: messages "
            f"('{script_args.messages_field}' column found)"
        )
        return "messages"

    if script_args.text_field in columns:
        logger.info(
            f"Auto-detected dataset format: text "
            f"('{script_args.text_field}' column found)"
        )
        return "text"

    raise ValueError(
        f"Cannot detect dataset format. Dataset columns: {columns}. "
        f"Expected one of: 'input_ids' (pretokenized), "
        f"'{script_args.messages_field}' (conversational messages), "
        f"or '{script_args.text_field}' (pre-rendered text). "
        f"Use --dataset_format to specify explicitly, or "
        f"--messages_field / --text_field to match your column names."
    )


def _get_chat_template_callable(tokenizer, processor):
    """Return the best available apply_chat_template callable.

    Prefers the processor (handles multimodal content like images/video)
    and falls back to the tokenizer for text-only models.
    """
    if processor is not None and hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template
    return tokenizer.apply_chat_template


def _tokenize_sample_for_length(
    sample: dict,
    tokenizer: AutoTokenizer,
    processor,
    dataset_format: str,
    script_args: ScriptArguments,
) -> Optional[int]:
    """Tokenize a single sample and return its token length, or None to skip."""
    if dataset_format == "messages":
        messages = sample[script_args.messages_field]
        if isinstance(messages, str):
            messages = json.loads(messages)
        kwargs = {}
        if "tools" in sample and sample["tools"]:
            tools = sample["tools"]
            if isinstance(tools, str):
                tools = json.loads(tools)
            kwargs["tools"] = tools
        apply_chat_template = _get_chat_template_callable(tokenizer, processor)
        tokens = apply_chat_template(messages, tokenize=True, **kwargs)
        return len(tokens)
    else:
        text = sample[script_args.text_field]
        if len(text) > 100000:
            return None
        tokens = tokenizer(text, add_special_tokens=True)["input_ids"]
        return len(tokens)


def calculate_optimal_max_length(
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    script_args: ScriptArguments,
    dataset_format: str,
    processor=None,
    sample_size: int = 1000,
    percentile: float = 0.95,
    max_absolute_length: int = 32768,
) -> int:
    """Calculate optimal max_length by tokenizing a sample of the dataset."""
    sample_indices = torch.randperm(len(dataset))[: min(sample_size, len(dataset))]
    sample_data = dataset.select(sample_indices)

    token_lengths = []
    outliers_removed = 0

    for sample in sample_data:
        token_len = _tokenize_sample_for_length(
            sample, tokenizer, processor, dataset_format, script_args
        )

        if token_len is None or token_len > max_absolute_length:
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


def _tokenize_text_dataset(
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    script_args: ScriptArguments,
    max_length: Optional[int],
) -> Dataset:
    """Tokenize a text-column dataset."""
    return dataset.map(
        lambda sample: tokenizer(
            sample[script_args.text_field],
            padding=False,
            truncation=script_args.apply_truncation,
            max_length=max_length if script_args.apply_truncation else None,
        ),
        remove_columns=list(dataset.features),
        batched=True,
        batch_size=1000,
    )


def _tokenize_messages_dataset(
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    script_args: ScriptArguments,
    max_length: Optional[int],
    processor=None,
) -> Dataset:
    """Tokenize a conversational messages dataset using apply_chat_template.

    Uses the processor's apply_chat_template when available (multimodal models),
    falling back to the tokenizer's for text-only models.
    """
    has_tools = "tools" in dataset.column_names
    apply_chat_template = _get_chat_template_callable(tokenizer, processor)

    def tokenize_conversation(sample):
        messages = sample[script_args.messages_field]
        if isinstance(messages, str):
            messages = json.loads(messages)

        kwargs = {}
        if has_tools and sample["tools"]:
            tools = sample["tools"]
            if isinstance(tools, str):
                tools = json.loads(tools)
            kwargs["tools"] = tools

        token_ids = apply_chat_template(messages, tokenize=True, **kwargs)

        if script_args.apply_truncation and max_length is not None:
            token_ids = token_ids[:max_length]

        return {"input_ids": token_ids, "attention_mask": [1] * len(token_ids)}

    return dataset.map(
        tokenize_conversation,
        remove_columns=list(dataset.features),
    )


def prepare_dataset(
    tokenizer: AutoTokenizer,
    script_args: ScriptArguments,
    train_ds: Dataset,
    test_ds: Optional[Dataset] = None,
    processor=None,
):
    """Prepare the dataset for training with optimal tokenization.

    Supports three dataset formats (auto-detected or explicitly set via --dataset_format):
      - 'pretokenized': Dataset already has 'input_ids' column, skip tokenization.
      - 'text': Dataset has a text column (default 'text'), tokenize directly.
      - 'messages': Dataset has a messages column (default 'messages') with
        conversational format (list of message dicts). Uses apply_chat_template
        to render and tokenize. For multimodal models, the processor's
        apply_chat_template is used when available. Optionally supports a
        'tools' column (JSON string).
    """
    dataset_format = detect_dataset_format(train_ds, script_args)

    if dataset_format == "pretokenized":
        logger.info(
            "Dataset already contains 'input_ids' - skipping tokenization "
            f"(columns: {train_ds.column_names})"
        )
        return train_ds, test_ds

    if script_args.apply_truncation and dataset_format == "text":
        # Character-level outlier filtering only for text format
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

    if script_args.apply_truncation:
        if script_args.max_length is None:
            script_args.max_length = calculate_optimal_max_length(
                tokenizer, train_ds, script_args, dataset_format, processor
            )
    max_length = script_args.max_length

    logger.info(f"Using max_length: {max_length}")
    logger.info(f"Truncation enabled: {script_args.apply_truncation}")
    logger.info(f"Dataset format: {dataset_format}")

    if script_args.modality_type == "image":
        logger.info(
            "Image modality: skipping pre-tokenization (processing happens in data collator)"
        )
        return train_ds, test_ds

    if dataset_format == "messages":
        lm_train_dataset = _tokenize_messages_dataset(
            tokenizer, train_ds, script_args, max_length, processor
        )
        lm_test_dataset = (
            _tokenize_messages_dataset(
                tokenizer, test_ds, script_args, max_length, processor
            )
            if test_ds is not None
            else None
        )
    else:
        lm_train_dataset = _tokenize_text_dataset(
            tokenizer, train_ds, script_args, max_length
        )
        lm_test_dataset = (
            _tokenize_text_dataset(tokenizer, test_ds, script_args, max_length)
            if test_ds is not None
            else None
        )

    logger.info(f"Total number of train samples: {len(lm_train_dataset)}")
    if lm_test_dataset is not None:
        logger.info(f"Total number of test samples: {len(lm_test_dataset)}")

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
    is_vlm = script_args.modality_type == "image"

    train_ds, test_ds = prepare_dataset(
        tokenizer, script_args, train_ds, test_ds, processor
    )

    if script_args.use_peft:
        model = apply_lora_config(model, script_args, is_vlm=is_vlm)

    if (
        script_args.patch_peft_fsdp_auto_wrap_policy
        and script_args.use_peft
        and training_args.fsdp
        and training_args.fsdp != ""
    ):

        patch_peft_fsdp_auto_wrap_policy()
    if script_args.cast_parameters_to_uniform_dtype:
        cast_parameters_to_uniform_dtype(model, config_builder.torch_dtype)

    callbacks = setup_wandb(script_args)
    if script_args.early_stopping:
        if callbacks is None:
            callbacks = []
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=3, early_stopping_threshold=0.01
            )
        )

        training_args.load_best_model_at_end = True
        training_args.metric_for_best_model = "eval_loss"
        training_args.greater_is_better = False
    trainer = setup_trainer(
        model,
        tokenizer,
        train_ds,
        config_builder,
        test_ds,
        callbacks,
        processor=processor,
        script_args=script_args,
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


def _is_hf_dataset_dir(path: str) -> bool:
    """Check if path is a HuggingFace dataset directory (Arrow format)."""
    return os.path.isdir(path) and os.path.exists(
        os.path.join(path, "dataset_info.json")
    )


def _load_dataset_auto(path: str) -> Dataset:
    """Load a dataset from path, automatically detecting format (JSON, JSONL, or Arrow)."""
    if path.endswith(".jsonl") or path.endswith(".json"):
        return load_dataset("json", data_files=path, split="train")
    if path.endswith(".arrow"):
        logger.info(f"Loading Arrow file from {path}")
        return load_dataset("arrow", data_files=path, split="train")
    if _is_hf_dataset_dir(path):
        logger.info(f"Detected HuggingFace Arrow dataset format at {path}")
        ds = load_from_disk(path)
        if isinstance(ds, DatasetDict):
            split = "train" if "train" in ds else list(ds.keys())[0]
            logger.info(f"DatasetDict detected, using split '{split}'")
            ds = ds[split]
        return ds
    # Fallback: look for JSON/JSONL files in directory
    import glob as _glob

    json_files = sorted(
        _glob.glob(os.path.join(path, "*.json"))
        + _glob.glob(os.path.join(path, "*.jsonl"))
    )
    if json_files:
        logger.info(f"Found JSON file(s) in directory: {json_files}")
        return load_dataset("json", data_files=json_files, split="train")
    raise FileNotFoundError(
        f"No supported dataset files found in '{path}'. "
        "Expected .json, .jsonl, .arrow files or a HuggingFace dataset directory."
    )


def load_datasets(script_args: ScriptArguments) -> Tuple[Dataset, Optional[Dataset]]:
    """Load training and test datasets."""
    try:
        logger.info(f"Loading training dataset from {script_args.train_dataset_path}")
        train_ds = _load_dataset_auto(script_args.train_dataset_path)
        logger.info(
            f"Training dataset loaded: {len(train_ds)} samples, "
            f"columns: {train_ds.column_names}"
        )

        test_ds = None
        if script_args.val_dataset_path:
            logger.info(f"Loading test dataset from {script_args.val_dataset_path}")
            test_ds = _load_dataset_auto(script_args.val_dataset_path)
            logger.info(
                f"Test dataset loaded: {len(test_ds)} samples, "
                f"columns: {test_ds.column_names}"
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
