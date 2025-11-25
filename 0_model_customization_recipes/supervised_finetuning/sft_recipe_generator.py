#!/usr/bin/env python3
"""
Interactive CLI to generate SFT YAML recipes (LoRA / Spectrum / Full).

Modes
-----
1) Full mode (default):
   - Detailed guided flow (strategy -> modality -> model -> dataset -> training -> logging).

2) Easy mode (--easy):
   - Uses example defaults, only asks for:
        0. Fine-tuning strategy   (PEFT / Spectrum / Full)
        1. model_name_or_path
        2. attn_implementation
        3. dataset_id_or_path     (dataset name; path is derived)
        4. modality_type          (text / image / video / audio; text for reasoning)
        5. report_to              (logging backend)

Shared behavior
---------------
- Base checkpoint path: /opt/ml/checkpoints
- Base dataset path:    /opt/ml/input/data/training
- Saves recipe to:
    sagemaker_code/hf_recipes/<model-family>/<model-name>--<vanilla|liger>-<strategy>.yaml
"""

from pathlib import Path
import textwrap
import os
import argparse
from typing import Optional, Tuple

# Static base paths
BASE_CHECKPOINT_ROOT = "/opt/ml/checkpoints"
BASE_TRAIN_DATA_ROOT = "/opt/ml/input/data/training"


# ---------- Small helpers ----------

def banner(title: str) -> None:
    box_width = max(len(title) + 4, 60)
    print("\n" + "┌" + "─" * (box_width - 2) + "┐")
    print("│ " + title.center(box_width - 4) + " │")
    print("└" + "─" * (box_width - 2) + "┘")


def ask(prompt: str, default: Optional[str] = None) -> str:
    if default is not None:
        full = f"{prompt} [{default}]: "
    else:
        full = f"{prompt}: "
    val = input(full).strip()
    return val if val else (default if default is not None else "")


def yes_no(prompt: str, default: bool = True) -> bool:
    d = "Y/n" if default else "y/N"
    val = input(f"{prompt} ({d}): ").strip().lower()
    if not val:
        return default
    return val.startswith("y")


def split_model_for_paths(model_id: str) -> Tuple[str, str]:
    """Return (org, model_name_part) for building output_dir and recipe path."""
    if "/" in model_id:
        org, model = model_id.split("/", 1)
    else:
        org, model = "models", model_id
    return org, model


def model_display_name(model_id: str) -> str:
    """Display name used in run_name (e.g., org/model -> model)."""
    if "/" in model_id:
        return model_id.split("/", 1)[1]
    return model_id


# ---------- YAML templates ----------

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
{lora_modules_to_save_block}lora_r: {lora_r}
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


# ---------- Common choice helpers ----------

def select_method() -> str:
    # Step 0: SFT strategy
    banner("Step 0: Choose Supervised Fine-Tuning Strategy")
    print("This controls *how many* parameters are updated and how heavy the training is.")
    print("  1) PEFT / LoRA      (parameter-efficient; adapters only)")
    print("  2) Spectrum         (selective unfreezing; between LoRA and full)")
    print("  3) Full fine-tuning (update all parameters)")
    while True:
        choice = input("Select strategy 1/2/3 [1]: ").strip() or "1"
        if choice in {"1", "2", "3"}:
            return {"1": "peft", "2": "spectrum", "3": "full"}[choice]
        print("Invalid choice, please enter 1, 2, or 3.")


def choose_modality(default: str = '"text"') -> str:
    banner("Modality Selection")
    print("Supported modality_type values: \"text\", \"image\", \"video\", \"audio\".")
    print("• Use \"text\" for standard language, instruction-following, and reasoning fine-tuning")
    print("  (including chain-of-thought, tool use, or code completion tasks).")
    print("• Use \"image\", \"video\", or \"audio\" when your primary input/output involves that modality.")
    while True:
        value = ask('modality_type ("text" | "image" | "video" | "audio")', default)
        plain = value.strip('"').strip("'")
        if plain in {"text", "image", "video", "audio"}:
            return f"\"{plain}\""
        print("Please choose one of: text, image, video, audio.")


def choose_attn_implementation(default: str = "eager") -> str:
    print("\n[attn_implementation]")
    print("    • Controls which attention kernel implementation is used.")
    print("      - eager              : reference PyTorch implementation (safest, most compatible).")
    print("      - flash_attention_2  : faster, memory-efficient attention (requires compatible GPUs).")
    print("      - kernels-community/vllm-flash-attn3 : specialized FlashAttention v3 kernels.")
    print("        NOTE: For 'kernels-community/vllm-flash-attn3', the instance type should be H100")
    print("              (or newer) to fully support these kernels.")
    print("    Options:")
    print("      1) eager")
    print("      2) flash_attention_2")
    print("      3) kernels-community/vllm-flash-attn3")
    while True:
        choice = input("Select attention implementation 1/2/3 [1]: ").strip() or "1"
        if choice == "1":
            return "eager"
        elif choice == "2":
            return "flash_attention_2"
        elif choice == "3":
            print("\n⚠  Reminder: 'kernels-community/vllm-flash-attn3' is optimized for H100 or newer GPUs.")
            print("   Ensure your training instance type provides H100-class hardware.")
            return "kernels-community/vllm-flash-attn3"
        else:
            print("Invalid choice, please enter 1, 2, or 3.")


# ---------- FULL MODE FLOW (detailed) ----------

def collect_model_section(base_defaults: dict, method: str) -> dict:
    banner("Model & Compute Configuration")

    # 1) model_name_or_path
    print("\n[1] model_name_or_path")
    print("    • The base model you want to fine-tune.")
    print("    • Use a Hugging Face ID (e.g., meta-llama/Llama-3.2-3B-Instruct)")
    print("      or a local path to a model directory.")
    while True:
        model_name = ask("Enter model_name_or_path", None)
        if model_name:
            break
        print("model_name_or_path is required.")

    # 2) tokenizer_name_or_path
    print("\n[2] tokenizer_name_or_path")
    print("    • Usually the same as model_name_or_path.")
    print("    • Press Enter to reuse the model name, or override if you use a custom tokenizer.")
    tokenizer_name = ask("Enter tokenizer_name_or_path", model_name)

    # 3) attn_implementation
    attn_impl = choose_attn_implementation(base_defaults["attn_implementation"])

    # 4) use_liger
    print("\n[4] use_liger")
    print("    • Enables 'liger' kernel optimizations (if integrated in your stack).")
    print("    • Keep this as 'false' unless you know you have proper support.")
    use_liger = ask("use_liger (true/false)", base_defaults["use_liger"])

    # 5) bf16
    print("\n[5] bf16")
    print("    • Whether to train in bfloat16 (bf16) precision.")
    print("    • Recommended: 'true' on modern hardware (A100/H100/Trn1) for speed and stability.")
    bf16 = ask("bf16 (true/false)", base_defaults["bf16"])

    # 6) tf32
    print("\n[6] tf32")
    print("    • TensorFloat-32 (TF32) is a mixed-precision mode on Ampere GPUs.")
    print("    • Often safe to leave as 'false' when using bf16.")
    tf32 = ask("tf32 (true/false)", base_defaults["tf32"])

    # 7) output_dir (derived)
    print("\n[7] output_dir")
    print("    • Where checkpoints and final model artifacts will be written.")
    print(f"    • Base path is fixed to: {BASE_CHECKPOINT_ROOT}")
    print("    • The tool derives a subdirectory from your model name + strategy.")
    org, model_part = split_model_for_paths(model_name)
    method_dir = {
        "peft": "peft-qlora",
        "spectrum": "spectrum",
        "full": "full-finetuning",
    }[method]
    output_default = f"{BASE_CHECKPOINT_ROOT}/{org}/{model_part}/{method_dir}/"
    print(f"      -> Suggested: {output_default}")
    print("    • Press Enter to accept, or override if you want a custom location.")
    output_dir = ask("output_dir", output_default)

    return {
        "model_name": model_name,
        "tokenizer_name": tokenizer_name,
        "model_revision": base_defaults["model_revision"],
        "torch_dtype": base_defaults["torch_dtype"],
        "attn_implementation": attn_impl,
        "use_liger": use_liger,
        "bf16": bf16,
        "tf32": tf32,
        "output_dir": output_dir,
    }


def collect_dataset_section(base_defaults: dict) -> dict:
    # Soft delineation message
    banner("Dataset Configuration")
    print("Now let's get your dataset information.")

    # 8) dataset_id_or_path (json name)
    print("\n[8] dataset_id_or_path (JSON file name only)")
    print("    • This is the *file name* of your training dataset under:")
    print(f"      {BASE_TRAIN_DATA_ROOT}")
    print("    • IMPORTANT: Please provide only the dataset name, NOT a full path.")
    print("      Examples:")
    print("        AI-MO--NuminaMath-CoT")
    print("        mesolitica--AudioSet-Audio-Instructions")
    print("    • The tool will automatically turn this into:")
    print("        /opt/ml/input/data/training/<name>.jsonl")
    raw_name = ask("Dataset name (without path and without .jsonl)", base_defaults["dataset_name"])

    base_name = os.path.basename(raw_name)
    if base_name.endswith(".jsonl"):
        base_name = base_name[:-len(".jsonl")]
    dataset_name = base_name
    dataset_path = f"{BASE_TRAIN_DATA_ROOT}/{dataset_name}.jsonl"

    # 9) max_seq_length
    print("\n[9] max_seq_length")
    print("    • Maximum sequence length (in tokens) for each training example.")
    print("    • Higher values capture more context but use more memory.")
    print("    • 4096 is a good default for many LLMs.")
    max_seq_length = ask("max_seq_length", base_defaults["max_seq_length"])

    # 10) packing
    print("\n[10] packing")
    print("    • Whether to pack multiple short examples into a single sequence window.")
    print("    • 'false' (default): each example is tokenized separately.")
    print("    • 'true'           : can increase throughput, but changes example boundaries.")
    packing = ask("packing (true/false)", base_defaults["packing"])

    return {
        "dataset_name": dataset_name,
        "dataset_path": dataset_path,
        "max_seq_length": max_seq_length,
        "packing": packing,
    }


def collect_training_section(base_defaults: dict) -> dict:
    banner("Training Hyperparameters")
    print("These control how long and how aggressively the model is trained.")

    print("\n[Training] num_train_epochs")
    print("    • How many passes over the full dataset to make.")
    num_epochs = ask("num_train_epochs", base_defaults["num_train_epochs"])

    print("\n[Training] per_device_train_batch_size")
    print("    • Batch size per GPU/instance.")
    print("    • Increase until you hit GPU memory limits.")
    bs = ask("per_device_train_batch_size", base_defaults["per_device_train_batch_size"])

    print("\n[Training] gradient_accumulation_steps")
    print("    • Virtually increases batch size by accumulating gradients over multiple steps.")
    print("    • Effective batch size = batch_size * gradient_accumulation_steps.")
    gas = ask("gradient_accumulation_steps", base_defaults["gradient_accumulation_steps"])

    print("\n[Training] gradient_checkpointing")
    print("    • 'true' saves memory by recomputing activations during backward pass.")
    print("    • Typically 'true' for large models.")
    gc = ask("gradient_checkpointing (true/false)", base_defaults["gradient_checkpointing"])

    print("\n[Training] learning_rate")
    print("    • Step size for the optimizer.")
    print("    • 1e-4 is a common starting point for adapter-based tuning.")
    lr = ask("learning_rate", base_defaults["learning_rate"])

    print("\n[Training] lr_scheduler_type")
    print("    • Learning rate schedule over time.")
    print("    • Common options: linear, cosine, constant.")
    sched = ask("lr_scheduler_type", base_defaults["lr_scheduler_type"])

    print("\n[Training] warmup_ratio")
    print("    • Fraction of total steps used to linearly warm up the learning rate.")
    warmup = ask("warmup_ratio", base_defaults["warmup_ratio"])

    return {
        "num_train_epochs": num_epochs,
        "per_device_train_batch_size": bs,
        "gradient_accumulation_steps": gas,
        "gradient_checkpointing": gc,
        "learning_rate": lr,
        "lr_scheduler_type": sched,
        "warmup_ratio": warmup,
        "use_reentrant": base_defaults["use_reentrant"],
    }


def collect_logging_section(base_defaults: dict, run_name_default: str) -> dict:
    banner("Logging & Experiment Tracking")
    print("These settings control how training progress is tracked and saved.")

    print("\n[Logging] report_to")
    print("    • Which tracking backend to use, e.g. mlflow, wandb, tensorboard.")
    report_to = ask("report_to backend", base_defaults["report_to"])

    print("\n[Logging] logging_strategy")
    print("    • When to log metrics (e.g., 'steps', 'epoch').")
    logging_strategy = ask("logging_strategy", base_defaults["logging_strategy"])

    print("\n[Logging] logging_steps")
    print("    • How often to log when using 'steps' strategy.")
    logging_steps = ask("logging_steps", base_defaults["logging_steps"])

    print("\n[Logging] save_strategy")
    print("    • When to save checkpoints (e.g., 'epoch', 'steps').")
    save_strategy = ask("save_strategy", base_defaults["save_strategy"])

    print("\n[Logging] seed")
    print("    • Random seed for reproducibility.")
    seed = ask("seed", base_defaults["seed"])

    print("\n[Logging] run_name")
    print("    • Human-readable name for this experiment run.")
    print("    • Auto-generated suggestion combines model, strategy, and dataset.")
    run_name = ask("run_name", run_name_default)

    return {
        "report_to": report_to,
        "logging_strategy": logging_strategy,
        "logging_steps": logging_steps,
        "save_strategy": save_strategy,
        "seed": seed,
        "run_name": run_name,
    }


def collect_peft_specific(base_defaults: dict) -> dict:
    banner("LoRA / PEFT Settings")
    print("These control how the low-rank adapters are configured.")

    print("\n[LoRA] load_in_4bit")
    print("    • 'true' will load the model in 4-bit quantized weights (QLoRA-style).")
    load_in_4bit = ask("load_in_4bit (true/false)", base_defaults["load_in_4bit"])

    print("\n[LoRA] lora_target_modules")
    print("    • Which modules to inject LoRA adapters into (YAML list as string).")
    print("    • Typical: [\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\"] for attention.")
    lora_target_modules = ask("lora_target_modules", base_defaults["lora_target_modules"])

    print("\n[LoRA] lora_modules_to_save (optional)")
    print("    • Optional modules to keep outside adapters (e.g., output head).")
    print("    • If you leave this empty, it will be omitted from the YAML.")
    lora_modules_to_save = ask("lora_modules_to_save (YAML list as string, optional)", "")

    print("\n[LoRA] lora_r")
    print("    • Rank of the low-rank adaptation matrices (capacity of LoRA).")
    lora_r = ask("lora_r", base_defaults["lora_r"])

    print("\n[LoRA] lora_alpha")
    print("    • Scaling factor applied to LoRA updates.")
    lora_alpha = ask("lora_alpha", base_defaults["lora_alpha"])

    # Build optional block
    if lora_modules_to_save.strip():
        block = f"# lora_modules_to_save: {lora_modules_to_save}\n"
    else:
        block = ""

    return {
        "load_in_4bit": load_in_4bit,
        "lora_target_modules": lora_target_modules,
        "lora_modules_to_save_block": block,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
    }


def collect_spectrum_specific(base_defaults: dict) -> dict:
    banner("Spectrum Settings (Selective Unfreezing)")
    print("Spectrum uses a config file to decide which layers to unfreeze.")
    print("Please provide the FULL path to the spectrum config YAML.")
    print("For example:")
    print("  sagemaker_code/hf_recipes/Qwen/Qwen3-VL-2B-Instruct-vanilla-spectrum.yaml")
    print("  sagemaker_code/hf_recipes/meta-llama/Llama-3.3-70B-Instruct--vanilla-spectrum.yaml")

    print("\n[Spectrum] spectrum_config_path (full path)")
    spectrum_config_path = ask(
        "spectrum_config_path",
        base_defaults["spectrum_config_path"],
    )

    print("\n[Spectrum] max_steps")
    print("    • Optional cap on total training steps.")
    max_steps = ask("max_steps", base_defaults["max_steps"])

    return {
        "spectrum_config_path": spectrum_config_path,
        "max_steps": max_steps,
    }


def collect_full_specific(base_defaults: dict) -> dict:
    # For full FT we don't need extra prompts beyond modality
    return {}


def run_full_flow() -> None:
    # Strategy
    method = select_method()
    # Global modality (text/image/video/audio)
    modality_type = choose_modality('"text"')

    # Shared defaults
    base_defaults = {
        "model_revision": "main",
        "torch_dtype": "bfloat16",
        "use_liger": "false",
        "bf16": "true",
        "tf32": "false",
        "attn_implementation": "eager",
        "dataset_name": "AI-MO--NuminaMath-CoT",
        "max_seq_length": "4096",
        "packing": "false",
        "num_train_epochs": "1",
        "per_device_train_batch_size": "2",
        "gradient_accumulation_steps": "2",
        "gradient_checkpointing": "true",
        "use_reentrant": "true",
        "learning_rate": "1.0e-4",
        "lr_scheduler_type": "cosine",
        "warmup_ratio": "0.1",
        "logging_strategy": "steps",
        "logging_steps": "2",
        "report_to": "mlflow",
        "save_strategy": "epoch",
        "seed": "42",
    }

    if method == "peft":
        method_defaults = {
            "load_in_4bit": "true",
            "lora_target_modules": '["q_proj", "k_proj", "v_proj", "o_proj"]',
            "lora_r": "8",
            "lora_alpha": "16",
        }
    elif method == "spectrum":
        method_defaults = {
            "spectrum_config_path": "sagemaker_code/hf_recipes/meta-llama/Llama-3.3-70B-Instruct--vanilla-spectrum.yaml",
            "max_steps": "32",
        }
    else:  # full
        method_defaults = {}

    defaults = {**base_defaults, **method_defaults}

    # Model & compute
    model_cfg = collect_model_section(defaults, method)

    # Dataset
    dataset_cfg = collect_dataset_section(defaults)

    # Adaptation-specific
    if method == "peft":
        adapt_cfg = collect_peft_specific(defaults)
    elif method == "spectrum":
        adapt_cfg = collect_spectrum_specific(defaults)
    else:
        adapt_cfg = collect_full_specific(defaults)
        adapt_cfg["max_steps"] = ""  # not used but placeholder for template consistency

    # Training + logging
    train_cfg = collect_training_section(defaults)

    # Derived run_name and recipe file info
    model_disp = model_display_name(model_cfg["model_name"])
    dataset_slug = dataset_cfg["dataset_name"]
    method_tag = {"peft": "peft-qlora", "spectrum": "spectrum", "full": "full"}[method]
    auto_run_name = f"{model_disp}-{method_tag}-{dataset_slug}"

    log_cfg = collect_logging_section(defaults, auto_run_name)

    # PEFT optional block handling
    if method != "peft":
        adapt_cfg.setdefault("lora_modules_to_save_block", "")

    # Assemble YAML
    banner("Validate & Generate YAML")

    common_fmt = {
        **model_cfg,
        **dataset_cfg,
        **train_cfg,
        **log_cfg,
        "modality_type": modality_type,
    }

    if method == "peft":
        yaml_text = PEFT_TEMPLATE.format(
            **common_fmt,
            **adapt_cfg,
        )
    elif method == "spectrum":
        yaml_text = SPECTRUM_TEMPLATE.format(
            **common_fmt,
            **adapt_cfg,
        )
    else:
        yaml_text = FULL_TEMPLATE.format(
            **common_fmt,
        )

    print("\nGenerated YAML preview:\n")
    print(textwrap.indent(yaml_text, "  "))

    if not yes_no("\nWrite this YAML to a recipe file?", default=True):
        print("Aborted. No file written.")
        return

    # Auto recipe path: sagemaker_code/hf_recipes/<org>/<model-name>--<vanilla|liger>-<strategy>.yaml
    org, model_part = split_model_for_paths(model_cfg["model_name"])
    flavor = "liger" if model_cfg["use_liger"].lower() == "true" else "vanilla"
    strategy_suffix = {"peft": "peft-qlora", "spectrum": "spectrum", "full": "full"}[method]
    recipe_dir = Path(f"sagemaker_code/hf_recipes/{org}")
    recipe_dir.mkdir(parents=True, exist_ok=True)
    recipe_path = recipe_dir / f"{model_part}--{flavor}-{strategy_suffix}.yaml"

    recipe_path.write_text(yaml_text, encoding="utf-8")
    print(f"\n✅ YAML written to: {recipe_path}")
    print("You can now run your trainer with, for example:")
    print(f"  python train.py --config {recipe_path}")


# ---------- EASY MODE FLOW ----------

def run_easy_flow() -> None:
    """
    Easy mode:

    Asks only:
      0. fine-tuning strategy
      1. model_name_or_path
      2. attn_implementation
      3. dataset_id_or_path (dataset name)
      4. modality_type
      5. report_to

    Everything else is inherited from the example YAMLs.
    """
    banner("EASY MODE: Minimal Questions, Example Defaults")

    # 0. Strategy
    method = select_method()

    # 4. modality_type
    modality_type = choose_modality('"text"')

    # 1. model_name_or_path
    print("\n[Easy] model_name_or_path")
    while True:
        model_name = ask("Enter model_name_or_path (e.g. org/model)", None)
        if model_name:
            break
        print("model_name_or_path is required.")
    tokenizer_name = model_name  # easy mode: same as model

    # 2. attn_implementation
    attn_impl = choose_attn_implementation("eager")

    # 3. dataset_id_or_path (dataset name; path derived)
    print("\n[Easy] dataset_id_or_path (dataset name only)")
    print(f"    Base path is fixed: {BASE_TRAIN_DATA_ROOT}")
    print("    Please provide just the dataset name (no path, no .jsonl).")
    raw_name = ask("Dataset name", "AI-MO--NuminaMath-CoT")
    base_name = os.path.basename(raw_name)
    if base_name.endswith(".jsonl"):
        base_name = base_name[:-len(".jsonl")]
    dataset_name = base_name
    dataset_path = f"{BASE_TRAIN_DATA_ROOT}/{dataset_name}.jsonl"

    # 5. report_to
    print("\n[Easy] report_to")
    print("    e.g., mlflow, wandb, tensorboard.")
    report_to = ask("report_to backend", "mlflow")

    # Shared example defaults
    common_defaults = {
        "model_revision": "main",
        "torch_dtype": "bfloat16",
        "use_liger": "false",
        "bf16": "true",
        "tf32": "false",
        "max_seq_length": "4096",
        "packing": "false",
        "gradient_checkpointing": "true",
        "use_reentrant": "true",
        "learning_rate": "1.0e-4",
        "lr_scheduler_type": "cosine",
        "warmup_ratio": "0.1",
        "logging_strategy": "steps",
        "save_strategy": "epoch",
        "seed": "42",
    }

    org, model_part = split_model_for_paths(model_name)
    model_disp = model_display_name(model_name)

    if method == "peft":
        method_dir = "peft-qlora"
        output_dir = f"{BASE_CHECKPOINT_ROOT}/{org}/{model_part}/{method_dir}/"

        yaml_text = PEFT_TEMPLATE.format(
            model_name=model_name,
            tokenizer_name=tokenizer_name,
            model_revision=common_defaults["model_revision"],
            torch_dtype=common_defaults["torch_dtype"],
            attn_implementation=attn_impl,
            use_liger=common_defaults["use_liger"],
            bf16=common_defaults["bf16"],
            tf32=common_defaults["tf32"],
            output_dir=output_dir,
            dataset_path=dataset_path,
            max_seq_length=common_defaults["max_seq_length"],
            packing=common_defaults["packing"],
            modality_type=modality_type,
            load_in_4bit="true",
            lora_target_modules='["q_proj", "k_proj", "v_proj", "o_proj"]',
            lora_modules_to_save_block="",
            lora_r="8",
            lora_alpha="16",
            num_train_epochs="10",
            per_device_train_batch_size="2",
            gradient_accumulation_steps="2",
            gradient_checkpointing=common_defaults["gradient_checkpointing"],
            use_reentrant=common_defaults["use_reentrant"],
            learning_rate=common_defaults["learning_rate"],
            lr_scheduler_type=common_defaults["lr_scheduler_type"],
            warmup_ratio=common_defaults["warmup_ratio"],
            logging_strategy=common_defaults["logging_strategy"],
            logging_steps="2",
            report_to=report_to,
            run_name=f"{model_disp}-peft-qlora-{dataset_name}",
            save_strategy=common_defaults["save_strategy"],
            seed=common_defaults["seed"],
        )

    elif method == "spectrum":
        method_dir = "spectrum"
        output_dir = f"{BASE_CHECKPOINT_ROOT}/{org}/{model_part}/{method_dir}/"

        banner("Easy: Spectrum Config")
        print("Please provide the FULL path to the spectrum config YAML.")
        print("For example:")
        print("  sagemaker_code/hf_recipes/Qwen/Qwen3-VL-2B-Instruct-vanilla-spectrum.yaml")
        print("  sagemaker_code/hf_recipes/meta-llama/Llama-3.3-70B-Instruct--vanilla-spectrum.yaml")
        spectrum_config_path = ask(
            "spectrum_config_path",
            "sagemaker_code/hf_recipes/meta-llama/Llama-3.3-70B-Instruct--vanilla-spectrum.yaml",
        )

        yaml_text = SPECTRUM_TEMPLATE.format(
            model_name=model_name,
            tokenizer_name=tokenizer_name,
            model_revision=common_defaults["model_revision"],
            torch_dtype=common_defaults["torch_dtype"],
            attn_implementation=attn_impl,
            use_liger=common_defaults["use_liger"],
            bf16=common_defaults["bf16"],
            tf32=common_defaults["tf32"],
            output_dir=output_dir,
            dataset_path=dataset_path,
            max_seq_length=common_defaults["max_seq_length"],
            packing=common_defaults["packing"],
            spectrum_config_path=spectrum_config_path,
            modality_type=modality_type,
            num_train_epochs="1",
            per_device_train_batch_size="2",
            gradient_accumulation_steps="2",
            gradient_checkpointing=common_defaults["gradient_checkpointing"],
            use_reentrant=common_defaults["use_reentrant"],
            learning_rate=common_defaults["learning_rate"],
            lr_scheduler_type=common_defaults["lr_scheduler_type"],
            warmup_ratio=common_defaults["warmup_ratio"],
            max_steps="32",
            logging_strategy=common_defaults["logging_strategy"],
            logging_steps="2",
            report_to=report_to,
            run_name=f"{model_disp}-spectrum-{dataset_name}",
            save_strategy=common_defaults["save_strategy"],
            seed=common_defaults["seed"],
        )

    else:  # full
        method_dir = "full-finetuning"
        output_dir = f"{BASE_CHECKPOINT_ROOT}/{org}/{model_part}/{method_dir}/"

        yaml_text = FULL_TEMPLATE.format(
            model_name=model_name,
            tokenizer_name=tokenizer_name,
            model_revision=common_defaults["model_revision"],
            torch_dtype=common_defaults["torch_dtype"],
            attn_implementation=attn_impl,
            use_liger=common_defaults["use_liger"],
            bf16=common_defaults["bf16"],
            tf32=common_defaults["tf32"],
            output_dir=output_dir,
            dataset_path=dataset_path,
            max_seq_length=common_defaults["max_seq_length"],
            packing=common_defaults["packing"],
            modality_type=modality_type,
            num_train_epochs="1",
            per_device_train_batch_size="2",
            gradient_accumulation_steps="2",
            gradient_checkpointing=common_defaults["gradient_checkpointing"],
            use_reentrant=common_defaults["use_reentrant"],
            learning_rate=common_defaults["learning_rate"],
            lr_scheduler_type=common_defaults["lr_scheduler_type"],
            warmup_ratio=common_defaults["warmup_ratio"],
            logging_strategy=common_defaults["logging_strategy"],
            logging_steps="1",
            report_to=report_to,
            run_name=f"{model_disp}-full-{dataset_name}",
            save_strategy=common_defaults["save_strategy"],
            seed=common_defaults["seed"],
        )

    banner("Generated YAML (Easy Mode)")
    print("\nYAML preview:\n")
    print(textwrap.indent(yaml_text, "  "))

    if not yes_no("\nWrite this YAML to a recipe file?", default=True):
        print("Aborted. No file written.")
        return

    flavor = "vanilla"  # easy mode doesn't ask for liger
    strategy_suffix = {"peft": "peft-qlora", "spectrum": "spectrum", "full": "full"}[method]
    recipe_dir = Path(f"sagemaker_code/hf_recipes/{org}")
    recipe_dir.mkdir(parents=True, exist_ok=True)
    recipe_path = recipe_dir / f"{model_part}--{flavor}-{strategy_suffix}.yaml"

    recipe_path.write_text(yaml_text, encoding="utf-8")
    print(f"\n✅ YAML written to: {recipe_path}")
    print("You can now run your trainer with, for example:")
    print(f"  python train.py --config {recipe_path}")


# ---------- Entry point ----------

def main() -> None:
    parser = argparse.ArgumentParser(description="SFT Recipe Generator CLI")
    parser.add_argument(
        "--easy",
        action="store_true",
        help="Enable easy mode (minimal questions, example defaults).",
    )
    args = parser.parse_args()

    if args.easy:
        run_easy_flow()
    else:
        run_full_flow()


if __name__ == "__main__":
    main()
