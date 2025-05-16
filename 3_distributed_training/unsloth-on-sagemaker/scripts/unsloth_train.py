# import torch
# print('TORCH VERSION:', torch.__version__)
# print('TORCH VERSION:', torch.__version__)
# print('TORCH VERSION:', torch.__version__)

import urllib.request
import subprocess
import sys


def patch_torch_and_flash_attn():
    # download flash_attn which works with torch 2.7.0 and cuda 12.8
    url = 'https://github.com/Zarrac/my-pytorch-builds/releases/download/flash-attn-2.7.4.post1-cuda12.8/flash_attn-2.7.4.post1+pt270cu128cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'
    filename = url.split('/')[-1]
    urllib.request.urlretrieve(url, filename)
    # uninstall torch 2.6.0
    # subprocess.check_call([sys.executable, "-m", "pip", "uninstall", 'torch', 'torchvision', 'torchaudio', '-y'])
    # install torch 2.7.0+cu128
    # subprocess.check_call([sys.executable, "-m", "pip", "install", '-r', 'requirements.txt',])
    # install flash_attn 2.7.4.post1 with patch
    subprocess.check_call([sys.executable, "-m", "pip", "install", filename])


print('PATHCING FLASH_ATTN...')
print('Please be patient, this may take a few minutes...')
patch_torch_and_flash_attn()

import torch

print('TORCH VERSION:', torch.__version__)
print('TORCH VERSION:', torch.__version__)
print('TORCH VERSION:', torch.__version__)


# import urllib.request
# urllib.request.urlretrieve("http://www.example.com/songs/mp3.mp3", "mp3.mp3")
# url = 'https://github.com/Zarrac/my-pytorch-builds/releases/download/flash-attn-2.7.4.post1-cuda12.8/flash_attn-2.7.4.post1+pt270cu128cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'
# filename = url.split('/')[-1]

# ### flash_attn installation workaround
# import subprocess
# import sys

# def install(package):
#     print('INSTALLING FLASH_ATTN')
#     subprocess.check_call([sys.executable, "-m", "pip", "install", filename])
# install('flash_attn')
###


from unsloth import FastModel
import torch

fourbit_models = [
    # 4bit dynamic quants for superior accuracy and low memory use
    "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",
    # Other popular models!
    "unsloth/Llama-3.1-8B",
    "unsloth/Llama-3.2-3B",
    "unsloth/Llama-3.3-70B",
    "unsloth/mistral-7b-instruct-v0.3",
    "unsloth/Phi-4",
]  # More models at https://huggingface.co/unsloth

model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-3-4b-it",
    max_seq_length=2048,  # Choose any for long context!
    load_in_4bit=True,  # 4 bit quantization to reduce memory
    load_in_8bit=False,  # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning=False,  # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)

model.config.use_cache = False


model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,  # Turn off for just text!
    finetune_language_layers=True,  # Should leave on!
    finetune_attention_modules=True,  # Attention good for GRPO
    finetune_mlp_modules=True,  # SHould leave on always!
    r=8,  # Larger = higher accuracy, but might overfit
    lora_alpha=8,  # Recommended alpha == r at least
    lora_dropout=0,
    bias="none",
    random_state=3407,
)

from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template="gemma-3",
)

from datasets import load_dataset

dataset = load_dataset("mlabonne/FineTome-100k", split="train")


from unsloth.chat_templates import standardize_data_formats

dataset = standardize_data_formats(dataset)


def apply_chat_template(examples):
    texts = tokenizer.apply_chat_template(examples["conversations"])
    return {"text": texts}


pass
dataset = dataset.map(apply_chat_template, batched=True)


from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    eval_dataset=None,  # Can set up evaluation!
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,  # Use GA to mimic batch size!
        warmup_steps=5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps=30,
        learning_rate=2e-4,  # Reduce to 2e-5 for long training runs
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",  # Use this for WandB etc
    ),
)

# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(
    torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


trainer_stats = trainer.train()


# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(
    f"Peak reserved memory for training % of max memory = {lora_percentage} %."
)


model.save_pretrained_merged(
    "/opt/ml/model/",
    tokenizer,
    save_method="merged_16bit",
)
