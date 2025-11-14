"""
Standalone inference script for Base and Fine-tuned (PEFT / Spectrum / Full) models.

This script supports:
- Uses TRL TrlParser to parse the same recipe YAML used for training
- 90/10 split for local .jsonl datasets (like training), or HuggingFace splits if provided
- Runs inference using vLLM (each model loaded once)
- Supports MXFP4 quantization if enabled in ScriptArguments
- Saves results in JSONL format under output_dir:
    <output_dir>/<model_basename>--<dataset_name>__base.jsonl
    <output_dir>/<model_basename>--<dataset_name>__target.jsonl
- At the end, writes an evaluation config YAML with paths, metrics, and model judge info
"""

import gc
import time
import json
import logging
import os
import shutil
import yaml
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset, load_dataset
from peft import AutoPeftModelForCausalLM
from tqdm import tqdm
# from transformers import AutoProcessor, AutoTokenizer, set_seed
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
)
from trl import ModelConfig, SFTConfig, TrlParser
from vllm import LLM, SamplingParams

try:
    from transformers import (
        Qwen3VLConfig,
        Qwen3VLForConditionalGeneration,
    )

    # fixes issue: ValueError: Unrecognized configuration class <class'transformers.models.qwen3_vl.configuration_qwen3_vl.Qwen3VLConfig'>
    # https://github.com/QwenLM/Qwen3-VL/issues/43
    AutoModelForCausalLM.register(
        config_class=Qwen3VLConfig, model_class=Qwen3VLForConditionalGeneration
    )

except ImportError:
    print(
        "[WARN] Qwen3VLForConditionalGeneration not found in transformers. "
        "Make sure you have the latest version."
    )
    pass


# Configure logging
def setup_logging() -> logging.Logger:
    """Set up logging configuration for the inference script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


logger = setup_logging()


@dataclass
class ScriptArguments:
    """Custom arguments for the inference script."""

    dataset_id_or_path: str
    """Path to dataset file (.jsonl) or HuggingFace dataset identifier."""

    dataset_splits: str = "train"
    """Dataset splits to use for inference."""

    max_seq_length: int = 2048
    """Maximum sequence length for tokenization."""

    tokenizer_name_or_path: Optional[str] = None
    """Path to tokenizer or HuggingFace tokenizer identifier. If None, uses model tokenizer."""

    processor_name_or_path: Optional[str] = None
    """Path to processor or HuggingFace processor identifier. If None, uses model processor."""

    modality_type: Optional[str] = "text"
    """Type of modality to use during inference "video", "image", "audio" or "text" """

    eval_max_samples: int = 1000
    """Maximum number of samples to use for evaluation (for efficiency)."""

    eval_max_new_tokens: int = 2048
    """Maximum number of new tokens to generate during evaluation."""

    eval_batch_size: int = 4
    """Batch size for inference. Higher values = faster inference but more memory."""

    use_liger: bool = False
    """Whether to use LigerKernel over AutoClass for loading model."""

    mxfp4: bool = False
    """Whether to use MXFP4 quantization instead of standard 4-bit quantization."""

    spectrum_config_path: Optional[str] = None
    """Path to YAML config file specifying which parameters to unfreeze for Spectrum training."""

    eval_mlflow_tracking_server_arn: str = os.getenv(
        "MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"
    )
    """MLflow tracking server ARN for evaluation metadata."""

    eval_mlflow_experiment_name: str = os.getenv(
        "MLFLOW_EXPERIMENT_NAME", "Default"
    )
    """MLflow experiment name for evaluation metadata."""

    eval_metrics: List[str] = field(
        default_factory=lambda: [
            "bert",
            "bleu",
            "rouge1",
            "rouge2",
            "rougeL",
            "rougeLsum",
            "bleu"
        ]
    )
    """List of evaluation metrics to compute."""

    eval_model_judge: str = None
    """Model judge identifier for evaluation."""

    eval_model_judge_parameters: Dict[str, Any] = field(
        default_factory=lambda: {"temperature": 0.1}
    )
    """Parameters for the model judge."""


def load_eval_dataset(script_args: ScriptArguments) -> Dataset:
    """
    Load evaluation dataset (90/10 split if local .jsonl).

    Args:
        script_args: Script arguments containing dataset configuration

    Returns:
        Evaluation dataset

    Raises:
        ValueError: If dataset loading fails or required attributes are missing
    """
    dataset_path = script_args.dataset_id_or_path

    try:
        if dataset_path.endswith(".jsonl"):
            # Load local JSONL file
            logger.info(f"Loading local JSONL dataset: {dataset_path}")
            full_dataset = load_dataset("json", data_files=dataset_path, split="train")
            total_samples = len(full_dataset)
            split_idx = int(0.9 * total_samples)
            logger.warning(
                f"Using 90/10 split (train={split_idx}, eval={total_samples - split_idx})"
            )
            # Use evaluation split (last 10%)
            return full_dataset.select(range(split_idx, total_samples))
        else:
            # Load HuggingFace dataset
            logger.info(f"Loading HuggingFace dataset: {dataset_path}")

            # Check if we have the required split attributes
            if not hasattr(script_args, "dataset_test_split"):
                raise ValueError(
                    "dataset_test_split not found in script_args for HuggingFace dataset"
                )

            test_split = getattr(script_args, "dataset_test_split")
            config = getattr(script_args, "config", None)

            if config is not None:
                return load_dataset(dataset_path, config, split=test_split)
            else:
                return load_dataset(dataset_path, split=test_split)

    except Exception as e:
        logger.error(f"Failed to load evaluation dataset: {e}")
        raise


def load_tokenizer_or_processor(script_args: ScriptArguments, model_args: ModelConfig):
    """
    Load tokenizer (default) or processor (for multimodal).

    Args:
        script_args: Script arguments containing tokenizer/processor configuration
        model_args: Model arguments containing model configuration

    Returns:
        Configured tokenizer or processor
    """
    name = (
        script_args.tokenizer_name_or_path
        or script_args.processor_name_or_path
        or model_args.model_name_or_path
    )

    if script_args.processor_name_or_path:
        logger.info(f"Loading processor from {name}")
        return AutoProcessor.from_pretrained(
            name,
            revision=model_args.model_revision,
            trust_remote_code=model_args.trust_remote_code,
        )

    logger.info(f"Loading tokenizer from {name}")
    tokenizer = AutoTokenizer.from_pretrained(
        name,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")

    return tokenizer


def prepare_model_for_vllm(model_args: ModelConfig, tokenizer, is_peft: bool) -> str:
    """
    Prepare model for vLLM inference.

    If PEFT -> merge adapters from .../peft_adapter, save full model + tokenizer
    to the parent directory (i.e., without the /peft_adapter suffix) and return that path.
    Else -> return model path under SM_MODEL_DIR (Spectrum/Full).

    Args:
        model_args: Model configuration
        tokenizer: Tokenizer instance
        is_peft: Whether this is a PEFT model

    Returns:
        Path to model directory for vLLM
    """
    base_model_name = model_args.model_name_or_path
    sm_model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    if is_peft:
        base_dir = os.path.join(sm_model_dir, base_model_name)
        peft_dir = os.path.join(base_dir, "peft_adapter") + "/"

        logger.info(f"Preparing PEFT model for vLLM.")
        logger.info(f"Base model directory (full model target): {base_dir}")
        logger.info(f"PEFT adapter directory: {peft_dir}")

        if not os.path.isdir(peft_dir):
            raise FileNotFoundError(
                f"Expected PEFT adapter directory not found at {peft_dir}"
            )

        logger.info(f"Merging PEFT adapters from: {peft_dir}")
        model = AutoPeftModelForCausalLM.from_pretrained(peft_dir)
        merged = model.merge_and_unload()

        # Save merged model + tokenizer to the parent directory (without /peft_adapter)
        # This ensures vLLM loads from .../<model_name>/ and not from .../peft_adapter
        if not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)

        logger.info(f"Saving merged PEFT model to: {base_dir}")
        merged.save_pretrained(base_dir)
        tokenizer.save_pretrained(base_dir)

        logger.info(f"Merged PEFT model + tokenizer saved to: {base_dir}")
        return base_dir

    # Non-PEFT: use the standard model directory under SM_MODEL_DIR
    model_dir = os.path.join(sm_model_dir, base_model_name)
    logger.info(f"Using non-PEFT model directory for vLLM: {model_dir}")
    return model_dir


def run_inference_with_vllm(
    eval_ds: Dataset,
    llm: LLM,
    tokenizer,
    script_args: ScriptArguments,
    out_file: str,
) -> None:
    """
    Run inference with vLLM in batches and save results as JSONL.

    Args:
        eval_ds: Evaluation dataset
        llm: vLLM instance
        tokenizer: Tokenizer instance
        script_args: Script arguments
        out_file: Output file path
    """
    results: List[Dict[str, Any]] = []

    max_items = min(len(eval_ds), script_args.eval_max_samples)
    batch_size = max(1, script_args.eval_batch_size)

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=script_args.eval_max_new_tokens,
    )

    # Collect valid indices that actually have "messages"
    valid_indices: List[int] = []
    for i in range(max_items):
        sample = eval_ds[i]
        messages = sample.get("messages", [])
        if messages:
            valid_indices.append(i)

    if not valid_indices:
        logger.warning("No valid samples with 'messages' found in eval dataset.")
        # Still write an empty file for downstream consumers
        with open(out_file, "w", encoding="utf-8") as f:
            pass
        return

    logger.info(
        f"Running batched inference on {len(valid_indices)} samples "
        f"(batch_size={batch_size}, max_items={max_items})"
    )

    # Batched loop
    for start_idx in tqdm(
        range(0, len(valid_indices), batch_size), desc="Inference (batched)"
    ):
        batch_indices = valid_indices[start_idx : start_idx + batch_size]

        prompts: List[str] = []
        batch_original_messages: List[List[Dict[str, Any]]] = []

        # Build prompts for this batch
        for idx in batch_indices:
            sample = eval_ds[idx]
            messages: List[Dict[str, Any]] = sample.get("messages", [])
            if not messages:
                continue

            # Strip last assistant so model responds to user message
            last_asst = max(
                [j for j, m in enumerate(messages) if m.get("role") == "assistant"],
                default=-1,
            )
            messages_for_gen = [m for j, m in enumerate(messages) if j != last_asst]

            prompt = tokenizer.apply_chat_template(
                messages_for_gen,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt)
            batch_original_messages.append(messages)

        if not prompts:
            continue

        # Run vLLM on the whole batch of prompts
        outputs = llm.generate(prompts, sampling_params)

        # Map outputs back to their corresponding messages
        for messages, out in zip(batch_original_messages, outputs):
            if not out.outputs:
                response_text = ""
            else:
                response_text = out.outputs[0].text.strip()

            ground_truth: List[Dict[str, Any]] = []
            messages_with_pred: List[Dict[str, Any]] = []
            assistant_seen = 0

            for msg in messages:
                if msg.get("role") == "assistant":
                    # Record ground-truth assistant content (including thinking if present)
                    ground_truth.append(
                        {
                            "role": "assistant",
                            "content": msg.get("content"),
                            "thinking": msg.get("thinking", None),
                        }
                    )

                    # First assistant turn is replaced by model prediction
                    if assistant_seen == 0:
                        messages_with_pred.append(
                            {
                                "role": "assistant",
                                "content": response_text,
                                "thinking": None,
                            }
                        )
                    else:
                        # Later assistant messages are kept as-is
                        messages_with_pred.append(msg)

                    assistant_seen += 1
                else:
                    # Non-assistant messages are copied as-is
                    messages_with_pred.append(msg)

            results.append(
                {
                    "messages": messages_with_pred,
                    "ground_truth": ground_truth,
                }
            )

    # Write all results to JSONL
    with open(out_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"Saved batched inference results: {out_file}")


def write_eval_config(
    output_dir: str,
    base_out: str,
    target_out: str,
    script_args: ScriptArguments,
    run_name: str,
) -> None:
    """
    Write evaluation metadata YAML file for downstream scoring.

    Args:
        output_dir: Output directory path
        base_out: Base model predictions file path
        target_out: Target model predictions file path
        script_args: Script arguments
        run_name: MLflow run name
    """
    eval_config = {
        "source_model_predictions_path": base_out,
        "target_model_predictions_path": target_out,
        "mlflow_tracking_server_arn": script_args.eval_mlflow_tracking_server_arn,
        "mlflow_experiment_name": script_args.eval_mlflow_experiment_name,
        "mlflow_run_name": run_name,
        "metrics": script_args.eval_metrics,
        "model_judge": script_args.eval_model_judge,
        "model_judge_parameters": script_args.eval_model_judge_parameters,
    }

    yaml_path = os.path.join(output_dir, "eval_config.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(eval_config, f, sort_keys=False)
    logger.info(f"Saved evaluation config: {yaml_path}")


def inference_function(
    model_args: ModelConfig, script_args: ScriptArguments, training_args: SFTConfig
) -> None:
    """
    Main inference function that orchestrates the entire inference process.

    Args:
        model_args: Model configuration from TRL parser
        script_args: Custom script arguments
        training_args: Training configuration from TRL parser
    """
    logger.info("=" * 50)
    logger.info("Starting Model Inference")
    logger.info("=" * 50)

    logger.info(f"\n\n🌀🌀🌀 MODALITY: {script_args.modality_type} 🌀🌀🌀")

    # Log all parameters
    logger.info(f"Model parameters: {model_args}")
    logger.info(f"Script parameters: {script_args}")
    logger.info(f"Training parameters: {training_args}")

    # Load evaluation dataset
    eval_ds = load_eval_dataset(script_args)
    ds_name = os.path.basename(script_args.dataset_id_or_path).replace(".jsonl", "")
    tokenizer = load_tokenizer_or_processor(script_args, model_args)

    model_base = os.path.basename(model_args.model_name_or_path)
    os.makedirs(training_args.output_dir, exist_ok=True)

    # ---- Target (Fine-tuned) model ----
    logger.info("Preparing fine-tuned model for inference...")
    if getattr(model_args, "use_peft", False):
        ft_path = prepare_model_for_vllm(model_args, tokenizer, is_peft=True)
    else:
        ft_path = prepare_model_for_vllm(model_args, tokenizer, is_peft=False)

    target_out = os.path.join(
        training_args.output_dir, f"{model_base}--{ds_name}__target.jsonl"
    )

    if os.path.exists(ft_path):
        logger.info("Loading fine-tuned model with vLLM...")
        llm_kwargs = {
            "model": ft_path,
            "tokenizer": ft_path,
            "trust_remote_code": True,
            "tensor_parallel_size": torch.cuda.device_count(),
        }
        # Note: MXFP4 quantization for target model is commented out in original
        # if script_args.mxfp4:
        #     logger.info("Using MXFP4 quantization for target model")
        #     llm_kwargs["quantization"] = "mxfp4"

        llm_target = LLM(**llm_kwargs)
        run_inference_with_vllm(eval_ds, llm_target, tokenizer, script_args, target_out)
        del llm_target
    else:
        logger.warning(f"Target model not found at {ft_path}, skipping.")

    # ---- Base model ----
    logger.info("Preparing base model for inference...")
    base_out = os.path.join(
        training_args.output_dir, f"{model_base}--{ds_name}__base.jsonl"
    )

    logger.info("Loading base model with vLLM...")
    llm_kwargs = {
        "model": model_args.model_name_or_path,
        "tokenizer": model_args.model_name_or_path,
        "trust_remote_code": True,
        "tensor_parallel_size": torch.cuda.device_count(),
    }
    if script_args.mxfp4:
        logger.info("Using MXFP4 quantization for base model")
        llm_kwargs["quantization"] = "mxfp4"

    llm_base = LLM(**llm_kwargs)
    run_inference_with_vllm(eval_ds, llm_base, tokenizer, script_args, base_out)
    del llm_base

    # ---- Write evaluation config ----
    logger.info("Writing evaluation configuration...")
    write_eval_config(
        training_args.output_dir,
        base_out,
        target_out,
        script_args,
        training_args.run_name,
    )

    logger.info("Inference completed successfully")
    logger.info("=" * 50)


def main() -> None:
    """
    Main entry point for the inference script.

    Parses arguments using TRL parser and runs the inference function.
    """
    try:
        # Parse arguments using TRL parser (preserving core functionality)
        parser = TrlParser((ModelConfig, ScriptArguments, SFTConfig))
        model_args, script_args, training_args = parser.parse_args_and_config()

        # Set seed for reproducibility
        if hasattr(training_args, "seed"):
            set_seed(training_args.seed)
            logger.info(f"Set random seed to {training_args.seed}")

        # Run the main inference loop
        inference_function(model_args, script_args, training_args)

    except Exception as e:
        logger.error(f"Inference failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
