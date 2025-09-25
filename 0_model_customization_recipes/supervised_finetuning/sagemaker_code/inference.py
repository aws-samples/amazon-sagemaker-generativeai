#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone inference script for Base and Fine-tuned (PEFT / Spectrum / Full) models.

Features:
- Uses TRL TrlParser to parse the same recipe YAML used for training.
- 90/10 split for local .jsonl datasets (like training), or HuggingFace splits if provided.
- Runs inference using vLLM (each model loaded once).
- Supports MXFP4 quantization if enabled in ScriptArguments.
- Saves results in JSONL format under output_dir:
    <output_dir>/<model_basename>--<dataset_name>__base.jsonl
    <output_dir>/<model_basename>--<dataset_name>__target.jsonl
- At the end, writes an evaluation config YAML with paths, metrics, and model judge info.
"""

import os
import json
import yaml
import shutil
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import torch
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoProcessor, set_seed
from peft import AutoPeftModelForCausalLM
from trl import TrlParser, ModelConfig, SFTConfig
from vllm import LLM, SamplingParams


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
def setup_logging() -> logging.Logger:
    """Configure logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("inference")


logger = setup_logging()


# ---------------------------------------------------------------------
# Script Arguments
# ---------------------------------------------------------------------
@dataclass
class ScriptArguments:
    # Dataset
    dataset_id_or_path: str
    dataset_splits: str = "train"
    max_seq_length: int = 2048

    # Tokenizer/Processor
    tokenizer_name_or_path: Optional[str] = None
    processor_name_or_path: Optional[str] = None

    # Inference
    modality_type: Optional[str] = "text"
    eval_max_samples: int = 1000
    eval_max_new_tokens: int = 2048
    use_liger: bool = False
    mxfp4: bool = False  # if True, enable vLLM MXFP4 quantization
    spectrum_config_path: Optional[str] = None

    # Evaluation metadata
    eval_mlflow_tracking_server_arn: str = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    eval_mlflow_experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "default")
    eval_metrics: List[str] = field(
        default_factory=lambda: ["bert", "rouge2", "toxicity", "bleu", "answer_similarity"]
    )
    eval_model_judge: str = "openai:/gpt-4o"
    eval_model_judge_parameters: Dict[str, Any] = field(default_factory=lambda: {"temperature": 0.1})


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
def load_eval_dataset(script_args: ScriptArguments) -> Dataset:
    """Load evaluation dataset (90/10 split if local .jsonl)."""
    src = script_args.dataset_id_or_path
    if src.endswith(".jsonl"):
        logger.info(f"Loading local JSONL dataset: {src}")
        full = load_dataset("json", data_files=src, split="train")
        n = len(full)
        split_idx = int(0.9 * n)
        logger.warning(f"Using 90/10 split (train={split_idx}, eval={n - split_idx})")
        return full.select(range(990, 1000)) #full.select(range(split_idx, n))

    if hasattr(script_args, "dataset_test_split"):
        test_split = getattr(script_args, "dataset_test_split")
        cfg = getattr(script_args, "config", None)
        return load_dataset(src, cfg, split=test_split) if cfg else load_dataset(src, split=test_split)

    raise ValueError("HF dataset requires `dataset_test_split` in YAML or local .jsonl input.")


# ---------------------------------------------------------------------
# Tokenizer / Processor
# ---------------------------------------------------------------------
def load_tokenizer_or_processor(script_args: ScriptArguments, model_args: ModelConfig):
    """Load tokenizer (default) or processor (for multimodal)."""
    name = (
        script_args.tokenizer_name_or_path
        or script_args.processor_name_or_path
        or model_args.model_name_or_path
    )

    if script_args.processor_name_or_path:
        return AutoProcessor.from_pretrained(
            name,
            revision=model_args.model_revision,
            trust_remote_code=model_args.trust_remote_code,
        )

    tok = AutoTokenizer.from_pretrained(
        name,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# ---------------------------------------------------------------------
# PEFT Merge & Save
# ---------------------------------------------------------------------
def prepare_model_for_vllm(model_args: ModelConfig, tokenizer, is_peft: bool) -> str:
    """
    If PEFT -> merge adapters, save full model + tokenizer to /tmp/<model_name>/ and return path.
    Else -> return model path under SM_MODEL_DIR (Spectrum/Full).
    """
    base_model = model_args.model_name_or_path
    tmp_dir = f"/tmp/{os.path.basename(base_model)}"

    if is_peft:
        ft_path = os.path.join(
            os.environ.get("SM_MODEL_DIR", "/opt/ml/model"),
            base_model,
            "peft",
        )
        logger.info(f"Merging PEFT adapters from: {ft_path}")
        model = AutoPeftModelForCausalLM.from_pretrained(ft_path)
        merged = model.merge_and_unload()

        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        merged.save_pretrained(tmp_dir)
        tokenizer.save_pretrained(tmp_dir)

        logger.info(f"Merged PEFT model + tokenizer saved to: {tmp_dir}")
        return tmp_dir

    return os.path.join(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"), base_model)


# ---------------------------------------------------------------------
# vLLM Inference
# ---------------------------------------------------------------------
def run_inference_with_vllm(
    eval_ds: Dataset,
    llm: LLM,
    tokenizer,
    script_args: ScriptArguments,
    out_file: str,
):
    """Run inference with vLLM and save results as JSONL."""
    results = []
    max_items = min(len(eval_ds), script_args.eval_max_samples)
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=script_args.eval_max_new_tokens,
    )

    for i in tqdm(range(max_items), desc="Inference"):
        sample = eval_ds[i]
        messages: List[Dict[str, str]] = sample.get("messages", [])
        if not messages:
            continue

        # Strip last assistant so model responds to user message
        last_asst = max(
            [i for i, m in enumerate(messages) if m["role"] == "assistant"],
            default=-1,
        )
        messages_for_gen = [m for i, m in enumerate(messages) if i != last_asst]

        prompt = tokenizer.apply_chat_template(
            messages_for_gen,
            tokenize=False,
            add_generation_prompt=True,
        )
        outputs = llm.generate([prompt], sampling_params)
        response = outputs[0].outputs[0].text.strip()

        # Build results
        ground_truth, messages_with_pred = [], []
        assistant_seen = 0
        for msg in messages:
            if msg["role"] == "assistant":
                ground_truth.append(
                    {
                        "role": "assistant",
                        "content": msg["content"],
                        "thinking": msg.get("thinking", None),
                    }
                )
                if assistant_seen == 0:
                    messages_with_pred.append(
                        {
                            "role": "assistant",
                            "content": response,
                            "thinking": None,
                        }
                    )
                else:
                    messages_with_pred.append(msg)
                assistant_seen += 1
            else:
                messages_with_pred.append(msg)

        results.append({"messages": messages_with_pred, "ground_truth": ground_truth})

    with open(out_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Saved inference results: {out_file}")


# ---------------------------------------------------------------------
# Write Evaluation Config
# ---------------------------------------------------------------------
def write_eval_config(
    output_dir: str,
    base_out: str,
    target_out: str,
    script_args: ScriptArguments,
    run_name: str, 
):
    """Write evaluation metadata YAML file for downstream scoring."""
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


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = TrlParser((ModelConfig, ScriptArguments, SFTConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()

    if hasattr(training_args, "seed"):
        set_seed(training_args.seed)
        logger.info(f"Seed set to {training_args.seed}")

    eval_ds = load_eval_dataset(script_args)
    ds_name = os.path.basename(script_args.dataset_id_or_path).replace(".jsonl", "")
    tokenizer = load_tokenizer_or_processor(script_args, model_args)

    model_base = os.path.basename(model_args.model_name_or_path)
    os.makedirs(training_args.output_dir, exist_ok=True)

    # ---- Target (Fine-tuned) model ----
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
        # if script_args.mxfp4:
        #     logger.info("Using MXFP4 quantization for target model")
        #     llm_kwargs["quantization"] = "mxfp4"
        #     llm_kwargs["quantization"] = "mxfp4"

        llm_target = LLM(**llm_kwargs)
        run_inference_with_vllm(eval_ds, llm_target, tokenizer, script_args, target_out)
        del llm_target
    else:
        logger.warning(f"Target model not found at {ft_path}, skipping.")

    # ---- Base model ----
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

    # ---- Write eval config ----
    write_eval_config(training_args.output_dir, base_out, target_out, script_args, training_args.run_name)


if __name__ == "__main__":
    main()
