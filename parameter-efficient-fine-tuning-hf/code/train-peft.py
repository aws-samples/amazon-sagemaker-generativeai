import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("git+https://github.com/huggingface/transformers.git@main")
install("git+https://github.com/huggingface/peft.git")

import argparse
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft import prepare_model_for_int8_training
import transformers
from datasets import load_dataset
from peft import LoraConfig, get_peft_model


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

def train(args):
    model = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-6.7b", 
        load_in_8bit=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-6.7b", use_fast=False)

    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    data = load_dataset("Abirate/english_quotes")
    data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=data["train"],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            learning_rate=args.lr,
            fp16=args.fp16,
            logging_steps=args.logging_steps,
            output_dir=args.model_dir,
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False  # silence the warnings.
    trainer.train()
    
    # save the model 
    model.save_pretrained(args.model_dir) 
    
def main(): 
    parser = argparse.ArgumentParser()

    # parsing the hyperparameters
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--per_device_train_batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--max_steps', type=int, default=200)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--fp16', type=bool, default=True)
    parser.add_argument('--logging_steps', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--test_batch_size', type=int, default=8)

    # PyTorch container environment variables for data, model, and output directories
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument("--save-model", action="store_true", default=True, help="For Saving the current Model")
    args, _ = parser.parse_known_args()
    
    # call the train function to start training the model.
    train(args)


if __name__ == "__main__":
    main()

