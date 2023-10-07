from djl_python import Input, Output
import os
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from typing import Any, Dict, Tuple
import deepspeed
import warnings
import tarfile

predictor = None
model_tar = "model.tar.gz"


def get_model(properties):
    
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    cwd = os.getcwd()

    print(os.listdir(cwd))
    
    print(f"Loading model from {cwd}")
    model = AutoModelForCausalLM.from_pretrained(
        cwd, low_cpu_mem_usage=True,trust_remote_code=True,
torch_dtype=torch.bfloat16
    )
    model = deepspeed.init_inference(model, mp_size=properties["tensor_parallel_degree"])
    
    print(f"Loading tokenizer from {cwd}")
    tokenizer = AutoTokenizer.from_pretrained(cwd)
    generator = pipeline(
        task="text-generation", model=model, tokenizer=tokenizer, device=local_rank
    )
    return generator


def handle(inputs: Input) -> None:
    global predictor
    if not predictor:
        predictor = get_model(inputs.get_properties())

    if inputs.is_empty():
        # Model server makes an empty call to warmup the model on startup
        return None
    data = inputs.get_as_json()
    text = data["text"]
    generation_kwargs = data["properties"]
    outputs = predictor(text, **generation_kwargs)
    result = {"outputs": outputs}
    return Output().add(result)
