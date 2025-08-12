import unsloth
from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer
import argparse
import os
import re
import pandas as pd
import torch
from datasets import load_dataset
from peft import LoraConfig
from rouge import Rouge
import wandb
import sys



def extract_content(text):
    match = re.match(r'^answer: <(.*)>$', text, re.DOTALL)
    return match.group(1) if match else text

def format_reward_func(prompts, completions):
    pattern = r'^answer: <.*>$'
    matches = [re.match(pattern, content, re.DOTALL) for content in completions]
    return [1.0 if match else 0.0 for match in matches]

def len_reward_func(prompts, completions):
    return [len(c)/1024. for c in completions]

def answer_similarity(prompts, completions, df):
    processed_completions = [extract_content(comp) for comp in completions]
    ground_truth_completions = []
    for prompt in prompts:
        row = df.loc[df['prompt'] == prompt]
        if not row.empty:
            gt = row['completion'].iloc[0]
            ground_truth_completions.append(extract_content(gt))
        else:
            ground_truth_completions.append("")
    rouge = Rouge()
    scores = [s["rouge-l"]["f"] for s in rouge.get_scores(processed_completions, ground_truth_completions)]
    return scores

def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--model_output_dir", type=str, default="/opt/ml/model")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--hf_model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--hf_dataset", type=str, default="w601sxs/processed_simpleCoT_b1ade")
    args = parser.parse_args()

    wandb.login(key=os.environ["WANDB_API_KEY"])
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "grpo-training"),
        name=os.environ.get("WANDB_RUN_NAME", "grpo-run-01"),
    )
    print("🚀 W&B run URL:", wandb.run.get_url())
    sys.stdout.flush()


    max_seq_length = 4096
    lora_rank = 8

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.hf_model_name,
        max_seq_length = max_seq_length,
        load_in_4bit = True,
        fast_inference = False,
        max_lora_rank = lora_rank,
        gpu_memory_utilization = 0.7,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha = lora_rank,
        use_gradient_checkpointing = "unsloth",
        random_state = 340,
    )

    dataset = load_dataset(args.hf_dataset, split="train")
    df = dataset.to_pandas()

    def wrapped_answer_similarity(prompts, completions):
        return answer_similarity(prompts, completions, df)

    training_args = GRPOConfig(
        use_vllm = False,
        #vllm_device = "auto",
        #vllm_gpu_memory_utilization = 0.6,
        learning_rate = 5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        optim = "paged_adamw_8bit",
        logging_steps = 1,
        bf16 = is_bfloat16_supported(),
        fp16 = not is_bfloat16_supported(),
        per_device_train_batch_size = args.per_device_train_batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        num_generations = 4,
        max_prompt_length = 2048,
        max_completion_length = 256,
        num_train_epochs = args.num_train_epochs,
        save_steps = 250,
        max_grad_norm = 0.1,
        #report_to = "none",
        report_to = "wandb",
        output_dir = args.model_output_dir,
    )

    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [wrapped_answer_similarity, format_reward_func, len_reward_func],
        args = training_args,
        train_dataset = dataset,
    )

    trainer.train()
    trainer.save_model(args.model_output_dir)
    wandb.finish()


if __name__ == "__main__":
    main()