import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("git+https://github.com/huggingface/transformers.git@main")

import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import transformers
from datasets import load_dataset


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def main():
    model = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-6.7b", 
        # device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-6.7b", use_fast=False)

    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    print_trainable_parameters(model)

    data = load_dataset("Abirate/english_quotes")
    data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=data["train"],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            max_steps=200,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
            output_dir="outputs",
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()

if __name__ == "__main__":
    main()

