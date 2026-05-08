#!/usr/bin/env python3
"""
SAMA SFT Recipe Generator MCP Server
Generates YAML training recipes for PEFT/LoRA, Spectrum, and Full fine-tuning
strategies. Writes recipes to ../supervised_finetuning/sagemaker_code/hf_recipes/.
"""

import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)

from fastmcp import FastMCP

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

mcp = FastMCP("sama-sft-recipe-generator-mcp-server")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_CHECKPOINT_ROOT = "/opt/ml/checkpoints"
BASE_TRAIN_DATA_ROOT = "/opt/ml/input/data/training"

VALID_MODALITIES = {"text", "image", "video", "audio"}
VALID_ATTN_IMPLS = {"eager", "kernels-community/flash-attn2", "kernels-community/vllm-flash-attn3"}
VALID_STRATEGIES = {"peft", "spectrum", "full"}


def _attn_for_instance(instance_type: str) -> str:
    """Return the correct attention implementation for a SageMaker instance type.

    - P5, P6 (H100/H200 class) → kernels-community/vllm-flash-attn3
    - G5, G6, G6e, P4, P4d, P4de → kernels-community/flash-attn2
    - Unknown / None → eager
    """
    if not instance_type:
        return "eager"
    it = instance_type.lower()
    # H100/H200 class instances → flash-attn3
    if any(x in it for x in ["p5.", "p5e.", "p6."]):
        return "kernels-community/vllm-flash-attn3"
    # A10G / A100 class instances → flash-attn2
    if any(x in it for x in ["g5.", "g6.", "g6e.", "p4.", "p4d.", "p4de."]):
        return "kernels-community/flash-attn2"
    return "eager"

# ---------------------------------------------------------------------------
# YAML Templates (matching sft_recipe_generator.py patterns)
# ---------------------------------------------------------------------------

PEFT_TEMPLATE = """\
# Model arguments
model_name_or_path: {model_name}
tokenizer_name_or_path: {tokenizer_name}
model_revision: {model_revision}
torch_dtype: {torch_dtype}
attn_implementation: {attn_implementation}
use_liger: {use_liger}
bf16: {bf16}
tf32: {tf32}
output_dir: {output_dir}

# Dataset arguments
dataset_id_or_path: {dataset_path}
max_seq_length: {max_seq_length}
packing: {packing}

# Modality type
modality_type: {modality_type}

# LoRA arguments
use_peft: true
load_in_4bit: {load_in_4bit}
lora_target_modules: {lora_target_modules}
lora_r: {lora_r}
lora_alpha: {lora_alpha}

# Training arguments
num_train_epochs: {num_train_epochs}
per_device_train_batch_size: {per_device_train_batch_size}
gradient_accumulation_steps: {gradient_accumulation_steps}
gradient_checkpointing: {gradient_checkpointing}
gradient_checkpointing_kwargs:
  use_reentrant: {use_reentrant}
learning_rate: {learning_rate}
lr_scheduler_type: {lr_scheduler_type}
warmup_ratio: {warmup_ratio}

# Logging arguments
logging_strategy: {logging_strategy}
logging_steps: {logging_steps}
report_to:
- {report_to}
run_name: {run_name}
save_strategy: {save_strategy}
seed: {seed}
"""

SPECTRUM_TEMPLATE = """\
# Model arguments
model_name_or_path: {model_name}
tokenizer_name_or_path: {tokenizer_name}
model_revision: {model_revision}
torch_dtype: {torch_dtype}
attn_implementation: {attn_implementation}
use_liger: {use_liger}
bf16: {bf16}
tf32: {tf32}
output_dir: {output_dir}

# Dataset arguments
dataset_id_or_path: {dataset_path}
max_seq_length: {max_seq_length}
packing: {packing}

# Spectrum Config
spectrum_config_path: {spectrum_config_path}

# Modality type
modality_type: {modality_type}

# Training arguments
num_train_epochs: {num_train_epochs}
per_device_train_batch_size: {per_device_train_batch_size}
gradient_accumulation_steps: {gradient_accumulation_steps}
gradient_checkpointing: {gradient_checkpointing}
gradient_checkpointing_kwargs:
  use_reentrant: {use_reentrant}
learning_rate: {learning_rate}
lr_scheduler_type: {lr_scheduler_type}
warmup_ratio: {warmup_ratio}
max_steps: {max_steps}

# Logging arguments
logging_strategy: {logging_strategy}
logging_steps: {logging_steps}
report_to:
- {report_to}
run_name: {run_name}
save_strategy: {save_strategy}
seed: {seed}
"""

FULL_TEMPLATE = """\
# Model arguments
model_name_or_path: {model_name}
tokenizer_name_or_path: {tokenizer_name}
model_revision: {model_revision}
torch_dtype: {torch_dtype}
attn_implementation: {attn_implementation}
use_liger: {use_liger}
bf16: {bf16}
tf32: {tf32}
output_dir: {output_dir}

# Dataset arguments
dataset_id_or_path: {dataset_path}
max_seq_length: {max_seq_length}
packing: {packing}

# Modality type
modality_type: {modality_type}

# Training arguments
num_train_epochs: {num_train_epochs}
per_device_train_batch_size: {per_device_train_batch_size}
gradient_accumulation_steps: {gradient_accumulation_steps}
gradient_checkpointing: {gradient_checkpointing}
gradient_checkpointing_kwargs:
  use_reentrant: {use_reentrant}
learning_rate: {learning_rate}
lr_scheduler_type: {lr_scheduler_type}
warmup_ratio: {warmup_ratio}

# Logging arguments
logging_strategy: {logging_strategy}
logging_steps: {logging_steps}
report_to:
- {report_to}
run_name: {run_name}
save_strategy: {save_strategy}
seed: {seed}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _split_model_id(model_id: str):
    """Return (org, model_name) from a HF model id."""
    if "/" in model_id:
        org, name = model_id.split("/", 1)
    else:
        org, name = "models", model_id
    return org, name


def _resolve_recipe_dir(base_dir: str, org: str) -> Path:
    """Resolve and create the recipe output directory."""
    p = Path(base_dir) / org
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def generate_sft_recipe(
    model_id: str,
    strategy: str = "peft",
    dataset_name: str = "AI-MO--NuminaMath-CoT",
    modality_type: str = "text",
    attn_implementation: str = "auto",
    instance_type: Optional[str] = None,
    use_liger: bool = False,
    max_seq_length: int = 4096,
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 2,
    learning_rate: str = "1.0e-4",
    lr_scheduler_type: str = "cosine",
    warmup_ratio: str = "0.1",
    report_to: str = "mlflow",
    packing: bool = False,
    # PEFT-specific
    load_in_4bit: bool = True,
    lora_target_modules: str = '["q_proj", "k_proj", "v_proj", "o_proj"]',
    lora_r: int = 8,
    lora_alpha: int = 16,
    # Spectrum-specific
    spectrum_config_path: Optional[str] = None,
    max_steps: int = 32,
    # Output control
    recipes_base_dir: str = "../supervised_finetuning/sagemaker_code/hf_recipes",
) -> Dict[str, Any]:
    """
    Generate an SFT training recipe YAML for a HuggingFace model.

    This is STEP 3 of the SFT workflow. Call this AFTER get_model_details confirms
    the model selection. Supports three strategies: 'peft' (LoRA/QLoRA), 'spectrum'
    (selective unfreezing), and 'full' (all parameters). The recipe is written to
    supervised_finetuning/sagemaker_code/hf_recipes/<org>/<model>--<flavor>-<strategy>.yaml

    WORKFLOW: search/filter → get_model_details → generate_sft_recipe → generate_training_script

    Args:
        model_id: HuggingFace model identifier (e.g. 'meta-llama/Llama-3.2-3B-Instruct').
        strategy: Fine-tuning strategy - 'peft', 'spectrum', or 'full'.
        dataset_name: Dataset name (without path or .jsonl extension).
        modality_type: Data modality - 'text', 'image', 'video', or 'audio'.
        attn_implementation: Attention impl - 'eager', 'kernels-community/flash-attn2', 'kernels-community/vllm-flash-attn3', or 'auto' (resolved from instance_type).
        instance_type: SageMaker instance type (e.g. 'ml.g6e.2xlarge'). Used to auto-resolve attn_implementation when set to 'auto'.
        use_liger: Whether to use Liger kernel optimizations.
        max_seq_length: Maximum sequence length in tokens.
        num_train_epochs: Number of training epochs.
        per_device_train_batch_size: Batch size per GPU.
        gradient_accumulation_steps: Gradient accumulation steps.
        learning_rate: Learning rate (string, e.g. '1.0e-4').
        lr_scheduler_type: LR scheduler type (e.g. 'cosine', 'linear').
        warmup_ratio: Warmup ratio as string.
        report_to: Logging backend ('mlflow', 'wandb', 'tensorboard').
        packing: Whether to pack multiple examples per sequence.
        load_in_4bit: (PEFT) Load model in 4-bit quantization.
        lora_target_modules: (PEFT) LoRA target modules as YAML list string.
        lora_r: (PEFT) LoRA rank.
        lora_alpha: (PEFT) LoRA alpha scaling factor.
        spectrum_config_path: (Spectrum) Path to spectrum config YAML.
        max_steps: (Spectrum) Maximum training steps.
        recipes_base_dir: Base directory for writing recipe files.

    Returns:
        Dictionary with status, recipe file path, and generated YAML content.
    """
    try:
        # Validate inputs
        strategy = strategy.lower()
        if strategy not in VALID_STRATEGIES:
            return {"status": "error", "message": f"Invalid strategy '{strategy}'. Must be one of: {VALID_STRATEGIES}"}

        modality_plain = modality_type.strip('"').strip("'")
        if modality_plain not in VALID_MODALITIES:
            return {"status": "error", "message": f"Invalid modality_type '{modality_type}'. Must be one of: {VALID_MODALITIES}"}

        if attn_implementation not in VALID_ATTN_IMPLS:
            if attn_implementation == "auto":
                attn_implementation = _attn_for_instance(instance_type)
            elif attn_implementation == "flash_attention_2":
                # Legacy alias — map to kernels-community variant
                attn_implementation = "kernels-community/flash-attn2"
            else:
                return {"status": "error", "message": f"Invalid attn_implementation '{attn_implementation}'. Must be one of: {VALID_ATTN_IMPLS} or 'auto'."}

        if strategy == "spectrum" and not spectrum_config_path:
            return {"status": "error", "message": "spectrum_config_path is required for spectrum strategy."}

        org, model_name = _split_model_id(model_id)

        # Derive paths
        strategy_dir = {"peft": "peft-qlora", "spectrum": "spectrum", "full": "full-finetuning"}[strategy]
        output_dir = f"{BASE_CHECKPOINT_ROOT}/{org}/{model_name}/{strategy_dir}/"
        dataset_path = f"{BASE_TRAIN_DATA_ROOT}/{dataset_name}.jsonl"
        flavor = "liger" if use_liger else "vanilla"
        strategy_suffix = {"peft": "peft-qlora", "spectrum": "spectrum", "full": "full"}[strategy]
        run_name = f"{model_name}-{strategy_suffix}-{dataset_name}"

        # Common template values
        fmt = {
            "model_name": model_id,
            "tokenizer_name": model_id,
            "model_revision": "main",
            "torch_dtype": "bfloat16",
            "attn_implementation": attn_implementation,
            "use_liger": str(use_liger).lower(),
            "bf16": "true",
            "tf32": "false",
            "output_dir": output_dir,
            "dataset_path": dataset_path,
            "max_seq_length": str(max_seq_length),
            "packing": str(packing).lower(),
            "modality_type": f'"{modality_plain}"',
            "num_train_epochs": str(num_train_epochs),
            "per_device_train_batch_size": str(per_device_train_batch_size),
            "gradient_accumulation_steps": str(gradient_accumulation_steps),
            "gradient_checkpointing": "true",
            "use_reentrant": "true",
            "learning_rate": learning_rate,
            "lr_scheduler_type": lr_scheduler_type,
            "warmup_ratio": warmup_ratio,
            "logging_strategy": "steps",
            "logging_steps": "2",
            "report_to": report_to,
            "run_name": run_name,
            "save_strategy": "epoch",
            "seed": "42",
        }

        if strategy == "peft":
            fmt.update({
                "load_in_4bit": str(load_in_4bit).lower(),
                "lora_target_modules": lora_target_modules,
                "lora_r": str(lora_r),
                "lora_alpha": str(lora_alpha),
            })
            yaml_text = PEFT_TEMPLATE.format(**fmt)
        elif strategy == "spectrum":
            fmt.update({
                "spectrum_config_path": spectrum_config_path,
                "max_steps": str(max_steps),
            })
            yaml_text = SPECTRUM_TEMPLATE.format(**fmt)
        else:
            yaml_text = FULL_TEMPLATE.format(**fmt)

        # Write to disk
        recipe_dir = _resolve_recipe_dir(recipes_base_dir, org)
        recipe_filename = f"{model_name}--{flavor}-{strategy_suffix}.yaml"
        recipe_path = recipe_dir / recipe_filename

        recipe_path.write_text(yaml_text, encoding="utf-8")
        logger.info(f"Recipe written to {recipe_path}")

        return {
            "status": "success",
            "message": f"SFT recipe generated successfully.",
            "recipe_path": str(recipe_path),
            "recipe_relative_path": f"hf_recipes/{org}/{recipe_filename}",
            "strategy": strategy,
            "model_id": model_id,
            "yaml_content": yaml_text,
            "next_step": "Show the recipe summary to the user. Then proceed to generate_training_script (sft_training_workflow server) with the same model_id, strategy, and dataset_name. Pass param_count from the earlier get_model_details call.",
        }

    except Exception as e:
        logger.error(f"Failed to generate recipe: {e}")
        return {"status": "error", "message": f"Recipe generation failed: {e}"}


@mcp.tool()
def list_existing_recipes(
    recipes_base_dir: str = "../supervised_finetuning/sagemaker_code/hf_recipes",
) -> Dict[str, Any]:
    """
    List all existing SFT recipe YAML files organized by model family.

    Args:
        recipes_base_dir: Base directory containing recipe files.

    Returns:
        Dictionary with status and a tree of existing recipes by family.
    """
    try:
        base = Path(recipes_base_dir)
        if not base.exists():
            return {"status": "success", "recipes": {}, "message": "No recipes directory found."}

        recipes: Dict[str, list] = {}
        for yaml_file in sorted(base.rglob("*.yaml")):
            family = yaml_file.parent.name
            recipes.setdefault(family, []).append({
                "filename": yaml_file.name,
                "path": str(yaml_file),
                "relative_path": str(yaml_file.relative_to(base)),
            })

        return {
            "status": "success",
            "total_recipes": sum(len(v) for v in recipes.values()),
            "families": list(recipes.keys()),
            "recipes": recipes,
        }
    except Exception as e:
        logger.error(f"Failed to list recipes: {e}")
        return {"status": "error", "message": f"Failed to list recipes: {e}"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    """Main entry point for the MCP server."""
    mcp.run(show_banner=False)


if __name__ == "__main__":
    main()
