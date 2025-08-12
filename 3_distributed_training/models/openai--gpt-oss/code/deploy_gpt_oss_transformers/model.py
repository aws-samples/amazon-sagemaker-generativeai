import os
import json
import traceback
from typing import Any, Dict, List, Optional

import torch
from djl_python import Input, Output
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# Globals (one per DJL worker)
MODEL = None
TOKENIZER = None
DEVICE = None


def _as_list(x):
    if x is None:
        return None
    return x if isinstance(x, list) else [x]


def _model_dir_from_env_or_props(properties: Optional[Dict[str, Any]] = None) -> str:
    """Resolve the on-disk model path inside the container."""
    sm_model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    # Allow overrides via properties/env, else default to ./openai/gpt-oss-20b
    subdir = (properties or {}).get("model_subdir") or os.environ.get(
        "MODEL_SUBDIR", "openai/gpt-oss-20b"
    )
    return os.path.join(sm_model_dir, subdir)


def load_model(properties: Optional[Dict[str, Any]] = None):
    """Load tokenizer + model in 4-bit NF4 with bf16 compute."""
    global MODEL, TOKENIZER, DEVICE
    if MODEL is not None and TOKENIZER is not None:
        return

    local_model_dir = _model_dir_from_env_or_props(properties)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32

    # BitsAndBytes 4-bit config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch_dtype,
    )

    # Tokenizer
    TOKENIZER = AutoTokenizer.from_pretrained(local_model_dir, use_fast=True)
    if TOKENIZER.pad_token_id is None and TOKENIZER.eos_token_id is not None:
        TOKENIZER.pad_token_id = TOKENIZER.eos_token_id

    # Model
    MODEL = AutoModelForCausalLM.from_pretrained(
        local_model_dir,
        device_map="auto",               # shard across visible GPUs
        quantization_config=bnb_config,  # bitsandbytes backend
        torch_dtype=torch_dtype,
        attn_implementation="eager",     # safest in container
        use_cache=True,
    )
    MODEL.eval()


def initialize(properties: dict):
    """DJL calls this once at startup (init=true)."""
    load_model(properties)


def _format_inputs(payload: Dict[str, Any]) -> torch.Tensor:
    """Support either 'messages' (chat) or 'prompt'/'inputs' (plain)."""
    global TOKENIZER, MODEL

    if "messages" in payload:
        input_ids = TOKENIZER.apply_chat_template(
            payload["messages"],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(MODEL.device)
        return input_ids

    text = payload.get("prompt", payload.get("inputs"))
    if text is None:
        raise ValueError("Missing 'prompt'/'inputs' or 'messages' in payload.")

    enc = TOKENIZER(text, return_tensors="pt", padding=False, truncation=False)
    return enc["input_ids"].to(MODEL.device)


def _generate(payload: Dict[str, Any]) -> str:
    """Run generation and apply optional stop sequences."""
    global MODEL, TOKENIZER

    params = (payload.get("parameters") or {})
    generation_kwargs = {
        "max_new_tokens": params.get("max_new_tokens", 512),
        "temperature": params.get("temperature", 0.7),
        "top_p": params.get("top_p", 0.95),
        "do_sample": params.get("do_sample", True),
        "repetition_penalty": params.get("repetition_penalty", 1.0),
        "eos_token_id": params.get("eos_token_id", None),
        "pad_token_id": params.get("pad_token_id", TOKENIZER.pad_token_id),
    }

    stop_sequences = _as_list(params.get("stop"))

    input_ids = _format_inputs(payload)

    with torch.inference_mode():
        output_ids = MODEL.generate(input_ids, **generation_kwargs)

    text = TOKENIZER.decode(output_ids[0], skip_special_tokens=True)

    # if stop_sequences:
    #     cut = len(text)
    #     for s in stop_sequences:
    #         idx = text.find(s)
    #         if idx != -1:
    #             cut = min(cut, idx)
    #     text = text[:cut]

    return text


def handle(inputs: Input) -> Output:
    """DJL entry point. Accepts JSON / NDJSON and returns JSON."""
    global MODEL, TOKENIZER
    out = Output()

    try:
        # Init ping or accidental empty body: ensure model is loaded and ACK.
        if inputs is None or inputs.is_empty():
            if MODEL is None or TOKENIZER is None:
                load_model()
            out.add_as_json({"status": "ok"})
            return out

        # Eager load if not preloaded
        if MODEL is None or TOKENIZER is None:
            load_model()

        ctype = inputs.get_property("Content-Type") or inputs.get_property("content-type")
        if ctype == "application/jsonlines":
            # NDJSON batch
            results: List[Dict[str, str]] = []
            for line in inputs.get_content().splitlines():
                if not line.strip():
                    continue
                payload = json.loads(line)
                results.append({"generated_text": _generate(payload)})
            out.add_as_json(results)
            return out

        payload = inputs.get_as_json()

        if isinstance(payload, list):
            results = [{"generated_text": _generate(p)} for p in payload]
            out.add_as_json(results)
            return out

        text = _generate(payload)
        out.add_as_json({"generated_text": text})
        return out

    except Exception as e:
        # Return full traceback for easier debugging in CloudWatch
        tb = traceback.format_exc()
        return out.error(f"{str(e)}\n{tb}")
