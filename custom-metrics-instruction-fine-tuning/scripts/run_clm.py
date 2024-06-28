import os
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    default_data_collator,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from datasets import load_from_disk
import torch
from sagemaker.session import Session
from sagemaker.experiments.run import load_run
import boto3

import bitsandbytes as bnb
from huggingface_hub import login, HfFolder
import evaluate
import numpy as np

LOG_DIR = "/opt/ml/output/tensorboard"

def parse_arge():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    # add model id and dataset path argument
    parser.add_argument(
        "--model_id",
        type=str,
        help="Model id to use for training.",
    )
    parser.add_argument(
        "--train_dataset_path", type=str, default="lm_dataset", help="Path to dataset."
    )
    parser.add_argument(
        "--eval_dataset_path", type=str, default="lm_dataset", help="Path to dataset."
    )    
    parser.add_argument(
        "--hf_token", type=str, default=HfFolder.get_token(), help="huggingface token."
    )
    # add training hyperparameters for epochs, batch size, learning rate, and seed
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size to use for training.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size to use for evaluation.",
    )
    parser.add_argument(
        "--lr", type=float, default=5e-5, help="Learning rate to use for training."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed to use for training."
    )
    parser.add_argument(
        "--gradient_checkpointing",
        type=bool,
        default=True,
        help="Path to deepspeed config file.",
    )
    parser.add_argument(
        "--bf16",
        type=bool,
        default=True if torch.cuda.get_device_capability()[0] == 8 else False,
        help="Whether to use bf16.",
    )
    parser.add_argument(
        "--merge_weights",
        type=bool,
        default=True,
        help="Whether to merge LoRA weights with base model.",
    )
    args, _ = parser.parse_known_args()

    if args.hf_token:
        print(f"Logging into the Hugging Face Hub with token {args.hf_token[:10]}...")
        login(token=args.hf_token)

    return args


# COPIED FROM https://github.com/artidoro/qlora/blob/main/qlora.py
def print_trainable_parameters(model, use_4bit=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    print(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )


# COPIED FROM https://github.com/artidoro/qlora/blob/main/qlora.py
def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def create_peft_model(model, gradient_checkpointing=True, bf16=True):
    from peft import (
        get_peft_model,
        LoraConfig,
        TaskType,
        prepare_model_for_kbit_training,
    )
    from peft.tuners.lora import LoraLayer

    # prepare int-4 model for training
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=gradient_checkpointing
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # get lora target modules
    modules = find_all_linear_names(model)
    print(f"Found {len(modules)} modules to quantize: {modules}")

    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=modules,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, peft_config)

    # pre-process the model by upcasting the layer norms in float 32 for
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if bf16:
                module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.float32)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                if bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

    model.print_trainable_parameters()
    return model




# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
    
#     # decode to text
    
#     if isinstance(logits, tuple):
#         logits = logits[0]

#     tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')

#     print('before decoding ...')
#     print(logits)
    
#     decoded_preds = tokenizer.batch_decode(logits, skip_special_tokens=True)   
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
#     correct_pred = 0
#     for pred, label in zip(decoded_preds, decoded_labels):
#         print(f'prediction: {pred}')
#         print(f'label: {label}')
#         if pred == label:
#             correct_pred += 1
#     result = {}
#     result['accuracy'] = correct_pred/len(decoded_preds)
    
#     return result #metric.compute(predictions=decoded_preds, references=decoded_labels)

# def preprocess_logits_for_metrics(logits, labels):
#     """
#     Original Trainer may have a memory leak. 
#     This is a workaround to avoid storing too many tensors that are not needed.
#     """
#     pred_ids = torch.argmax(logits[0], dim=-1)
#     return pred_ids, labels

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)

metric = evaluate.load("accuracy")

def process_tokens(text):
    text_str = ','.join(map("'{0}'".format, text))
    answer = []
    for item in text_str.split('\'##\',\'#\',\'Answer\','):
        if '\'###\'' in item:
            answer.append(item.split('\'###\'')[0])
        elif '\'Inst\',\'ruction\'' in item:
            answer.append(item.split('\'Inst\',\'ruction\'')[0])
        else:
            answer.append(item)
    return answer

def custom_metric(preds: list, labels: list):
    correct_pred = len(set(preds).intersection(set(labels)))

    result = {}
    result['accuracy'] = correct_pred/len(labels)  
    
    return result

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    

    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics
    labels = labels.reshape(-1)
    preds = preds.reshape(-1)

    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]
    
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # process tokens
    processed_preds = process_tokens(decoded_preds)
    processed_labels = process_tokens(decoded_labels)
      
    
    return custom_metric(processed_preds, processed_labels) #metric.compute(predictions=preds, references=labels)

def training_function(args):
    # set seed
    set_seed(args.seed)

    train_dataset = load_from_disk(args.train_dataset_path)
    eval_dataset = load_from_disk(args.eval_dataset_path)
    
    # load model from the hub with a bnb config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        use_cache=False
        if args.gradient_checkpointing
        else True,  # this is needed for gradient checkpointing
        device_map="auto",
        quantization_config=bnb_config,
    )

    # create peft config
    model = create_peft_model(
        model, gradient_checkpointing=args.gradient_checkpointing, bf16=args.bf16
    )

    # Define training args
    output_dir = LOG_DIR #"./tmp/llama2"
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        bf16=args.bf16,  # Use BF16 if available
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        gradient_checkpointing=args.gradient_checkpointing,
        evaluation_strategy="steps",
        eval_steps=120,
        # logging strategies
        # logging_dir=f"{output_dir}/logs",
        logging_dir=LOG_DIR,
        report_to='tensorboard',
        logging_strategy="steps",
        logging_steps=20,
        save_strategy="steps", # "no"
        save_steps=100,
    )

    
    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics = preprocess_logits_for_metrics,
    )

    # Start training
    trainer.train()
        
    # evaluate model
    eval_result = trainer.evaluate(eval_dataset=eval_dataset)
    
    for key, value in sorted(eval_result.items()):
        print(f"{key} = {value}\n")

    sagemaker_save_dir="/opt/ml/model/"
    if args.merge_weights:
        # merge adapter weights with base model and save
        # save int 4 model
        trainer.model.save_pretrained(output_dir, safe_serialization=False)
        # clear memory
        del model
        del trainer
        torch.cuda.empty_cache()

        from peft import AutoPeftModelForCausalLM

        # load PEFT model in fp16
        model = AutoPeftModelForCausalLM.from_pretrained(
            output_dir,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )  
        # Merge LoRA and base model and save
        model = model.merge_and_unload()        
        model.save_pretrained(
            sagemaker_save_dir, safe_serialization=True, max_shard_size="2GB"
        )
    else:
        trainer.model.save_pretrained(
            sagemaker_save_dir, safe_serialization=True
        )

    # save tokenizer for easy inference
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.save_pretrained(sagemaker_save_dir)


def main():
    args = parse_arge()
    training_function(args)


if __name__ == "__main__":
    main()
