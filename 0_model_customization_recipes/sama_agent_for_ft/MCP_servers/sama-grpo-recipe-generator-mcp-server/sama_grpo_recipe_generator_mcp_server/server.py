#!/usr/bin/env python3
"""
SAMA GRPO Recipe Generator MCP Server
Generates YAML recipes for GRPO RLVR (Reinforcement Learning with Verifiable Rewards)
training. Writes to ../preference_optimization/grpo_rlvr/sagemaker_code/hf_recipes/.
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

mcp = FastMCP("sama-grpo-recipe-generator-mcp-server")

VALID_ATTN_IMPLS = {"eager", "kernels-community/flash-attn2", "kernels-community/vllm-flash-attn3"}


def _attn_for_instance(instance_type: str) -> str:
    """Return the correct attention implementation for a SageMaker instance type.

    - P5, P6 (H100/H200 class) → kernels-community/vllm-flash-attn3
    - G5, G6, G6e, P4, P4d, P4de → kernels-community/flash-attn2
    - Unknown / None → eager
    """
    if not instance_type:
        return "eager"
    it = instance_type.lower()
    if any(x in it for x in ["p5.", "p5e.", "p6."]):
        return "kernels-community/vllm-flash-attn3"
    if any(x in it for x in ["g5.", "g6.", "g6e.", "p4.", "p4d.", "p4de."]):
        return "kernels-community/flash-attn2"
    return "eager"

BASE_CHECKPOINT_ROOT = "/opt/ml/checkpoints"
BASE_TRAIN_DATA_ROOT = "/opt/ml/input/data/training"

GRPO_TEMPLATE = """\
model_name_or_path: {model_id}
model_revision: {model_revision}
dtype: {dtype}
attn_implementation: {attn_implementation}
trust_remote_code: {trust_remote_code}
output_dir: {output_dir}

dataset_id_or_path: {dataset_path}

# Generation settings
num_generations: {num_generations}
max_grpo_completion_length: {max_grpo_completion_length}
mask_truncated_completions: {mask_truncated_completions}
learning_rate: {learning_rate}
lr_scheduler_type: "{lr_scheduler_type}"
warmup_steps: {warmup_steps}
num_train_epochs: {num_train_epochs}
per_device_train_batch_size: {per_device_train_batch_size}
gradient_accumulation_steps: {gradient_accumulation_steps}
gradient_checkpointing: {gradient_checkpointing}
save_strategy: "{save_strategy}"
save_steps: {save_steps}
save_total_limit: {save_total_limit}
logging_steps: {logging_steps}
report_to:
  - "{report_to}"
seed: {seed}
"""


@mcp.tool()
def generate_grpo_recipe(
    model_id: str,
    dataset_name: str = "grpo_financial_train",
    attn_implementation: str = "auto",
    instance_type: Optional[str] = None,
    num_generations: int = 8,
    max_grpo_completion_length: int = 1024,
    learning_rate: str = "5.0e-6",
    lr_scheduler_type: str = "constant",
    warmup_steps: int = 0,
    num_train_epochs: int = 15,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    save_steps: int = 25,
    save_total_limit: int = 3,
    report_to: str = "mlflow",
    recipes_base_dir: str = "../preference_optimization/grpo_rlvr/sagemaker_code/hf_recipes",
) -> Dict[str, Any]:
    """
    Generate a GRPO RLVR training recipe YAML.

    GRPO recipes differ from SFT: no PEFT/LoRA, uses num_generations,
    constant LR, higher epochs, step-based checkpointing.

    WORKFLOW: model selection → generate_grpo_recipe → generate_grpo_training_script

    Args:
        model_id: HuggingFace model identifier (e.g. 'Qwen/Qwen3-4B').
        dataset_name: Dataset filename without path or .jsonl extension.
        attn_implementation: Attention impl - 'eager', 'kernels-community/flash-attn2', 'kernels-community/vllm-flash-attn3', or 'auto' (resolved from instance_type).
        instance_type: SageMaker instance type. Used to auto-resolve attn_implementation when set to 'auto'.
        num_generations: Completions per prompt for GRPO (default 8).
        max_grpo_completion_length: Max tokens per generation (default 1024).
        learning_rate: Learning rate (default '5.0e-6').
        lr_scheduler_type: LR scheduler (default 'constant').
        warmup_steps: Warmup steps (default 0).
        num_train_epochs: Training epochs (default 15, GRPO needs more).
        per_device_train_batch_size: Batch size per GPU.
        gradient_accumulation_steps: Gradient accumulation (default 8).
        save_steps: Checkpoint every N steps.
        save_total_limit: Max checkpoints to keep.
        report_to: Logging backend.
        recipes_base_dir: Output directory for recipe files.

    Returns:
        Status, recipe path, and YAML content.
    """
    try:
        # Resolve attention implementation
        if attn_implementation == "auto":
            attn_implementation = _attn_for_instance(instance_type)
        elif attn_implementation == "flash_attention_2":
            attn_implementation = "kernels-community/flash-attn2"
        elif attn_implementation not in VALID_ATTN_IMPLS:
            return {"status": "error", "message": f"Invalid attn_implementation '{attn_implementation}'. Must be one of: {VALID_ATTN_IMPLS} or 'auto'."}

        if "/" in model_id:
            org, model_name = model_id.split("/", 1)
        else:
            org, model_name = "models", model_id

        output_dir = f"{BASE_CHECKPOINT_ROOT}/{org}/{model_name}/grpo-rlvr/"
        dataset_path = f"{BASE_TRAIN_DATA_ROOT}/{dataset_name}.jsonl"

        yaml_text = GRPO_TEMPLATE.format(
            model_id=model_id,
            model_revision="main",
            dtype="bfloat16",
            attn_implementation=attn_implementation,
            trust_remote_code="true",
            output_dir=output_dir,
            dataset_path=dataset_path,
            num_generations=num_generations,
            max_grpo_completion_length=max_grpo_completion_length,
            mask_truncated_completions="true",
            learning_rate=learning_rate,
            lr_scheduler_type=lr_scheduler_type,
            warmup_steps=warmup_steps,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing="true",
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            logging_steps=1,
            report_to=report_to,
            seed=42,
        )

        recipe_dir = Path(recipes_base_dir) / org
        recipe_dir.mkdir(parents=True, exist_ok=True)
        recipe_filename = f"{model_name}--grpo.yaml"
        recipe_path = recipe_dir / recipe_filename
        recipe_path.write_text(yaml_text, encoding="utf-8")

        return {
            "status": "success",
            "recipe_path": str(recipe_path),
            "recipe_relative_path": f"hf_recipes/{org}/{recipe_filename}",
            "model_id": model_id,
            "yaml_content": yaml_text,
            "next_step": "Proceed to generate_grpo_training_script (sagemaker_job_launcher server).",
        }
    except Exception as e:
        return {"status": "error", "message": f"GRPO recipe generation failed: {e}"}


@mcp.tool()
def list_grpo_recipes(
    recipes_base_dir: str = "../preference_optimization/grpo_rlvr/sagemaker_code/hf_recipes",
) -> Dict[str, Any]:
    """List all existing GRPO recipe YAML files."""
    try:
        base = Path(recipes_base_dir)
        if not base.exists():
            return {"status": "success", "recipes": {}, "message": "No GRPO recipes directory found."}
        recipes: Dict[str, list] = {}
        for f in sorted(base.rglob("*.yaml")):
            family = f.parent.name
            recipes.setdefault(family, []).append({
                "filename": f.name,
                "path": str(f),
            })
        return {
            "status": "success",
            "total_recipes": sum(len(v) for v in recipes.values()),
            "recipes": recipes,
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to list recipes: {e}"}


def main():
    mcp.run(show_banner=False)

if __name__ == "__main__":
    main()
