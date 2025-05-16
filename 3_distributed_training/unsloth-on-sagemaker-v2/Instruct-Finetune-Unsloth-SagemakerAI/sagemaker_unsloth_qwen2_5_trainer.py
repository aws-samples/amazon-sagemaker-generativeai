import unsloth
from unsloth import FastLanguageModel, is_bfloat16_supported
import argparse
import json
from datasets import load_dataset, Dataset
from transformers import TrainingArguments
from trl import SFTTrainer

def load_alpaca_dataset(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return Dataset.from_list(data)

def format_alpaca(example):
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""
    instruction = example["instruction"]
    input_text = example["input"]
    output_text = example["output"]
    return {
        "text": alpaca_prompt.format(instruction, input_text, output_text) + tokenizer.eos_token
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--model_output_dir", type=str, default="/opt/ml/model")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=60)
    args = parser.parse_args()

    global tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen2.5-7B-Instruct",
        max_seq_length = 4096,
        dtype = None,
        load_in_4bit = True,
    )

    dataset = load_alpaca_dataset(args.train_file)
    dataset = dataset.map(format_alpaca)

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = 4096,
        dataset_num_proc = 2,
        packing = False,
        args = TrainingArguments(
            output_dir = args.model_output_dir,
            per_device_train_batch_size = args.per_device_train_batch_size,
            gradient_accumulation_steps = args.gradient_accumulation_steps,
            max_steps = args.max_steps,
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            report_to = "none",
        ),
    )

    trainer.train()
    trainer.save_model(args.model_output_dir)

if __name__ == "__main__":
    main()
