#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 Amazon Web Services. All rights reserved.
"""Fine-tuning script for OpenVLA with LoRA on SageMaker."""
import argparse
import logging
import math
import os
import copy
import shutil
from pathlib import Path

import yaml
import datasets
from datasets import load_dataset
import numpy as np
import torch
import torch.utils.checkpoint
from torch.utils.data import DataLoader
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from peft import LoraConfig, get_peft_model
from peft.utils import get_peft_model_state_dict
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image

import sys
script_dir = Path(__file__).parent.absolute()
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from utils.openvla_utils import ActionTokenizer, PurePromptBuilder, VicunaV15ChatPromptBuilder

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, remaining_argv = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(parents=[config_parser])

    # Model arguments
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="openvla/openvla-7b")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)

    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--train_data_dir", type=str, default=None)
    parser.add_argument("--image_column", type=str, default="image")
    parser.add_argument("--instruction_column", type=str, default="instruction")
    parser.add_argument("--action_column", type=str, default="action")

    # Validation arguments
    parser.add_argument("--validation_prompt", type=str, default=None)
    parser.add_argument("--validation_image", type=str, default=None)
    parser.add_argument("--num_validation_samples", type=int, default=4)
    parser.add_argument("--validation_epochs", type=int, default=1)

    # Training arguments
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="openvla-finetuned-lora")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--center_crop", action="store_true", default=False)
    parser.add_argument("--random_flip", action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--scale_lr", action="store_true", default=False)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Logging and checkpointing
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--mixed_precision", type=str, choices=["no", "fp16", "bf16"], default=None)
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openvla-finetuning")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    # LoRA Configuration
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,v_proj,k_proj,out_proj")

    # Task-specific arguments
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--task_type", type=str, default="manipulation",
                        choices=["manipulation", "classification", "detection"])
    parser.add_argument("--image_interpolation_mode", type=str, default="lanczos")

    # Load YAML config if provided
    if config_args.config:
        with open(config_args.config, "r") as f:
            yaml_args = yaml.safe_load(f)
        valid_args = {a.dest for a in parser._actions}
        yaml_filtered = {k: v for k, v in yaml_args.items() if k in valid_args and v is not None}
        parser.set_defaults(**yaml_filtered)

    args = parser.parse_args(remaining_argv)

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.pretrained_model_name_or_path is None:
        raise ValueError("--pretrained_model_name_or_path must be specified")
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args


def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # --- Load Model & Processor ---
    logger.info(f"Loading OpenVLA model from {args.pretrained_model_name_or_path}")
    processor = AutoProcessor.from_pretrained(
        args.pretrained_model_name_or_path, revision=args.revision,
        cache_dir=args.cache_dir, trust_remote_code=True,
    )
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    model = AutoModelForVision2Seq.from_pretrained(
        args.pretrained_model_name_or_path, revision=args.revision,
        cache_dir=args.cache_dir, torch_dtype=torch.bfloat16,
        attn_implementation="eager", trust_remote_code=True,
    )
    model.requires_grad_(False)

    weight_dtype = torch.bfloat16
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    model.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # --- LoRA Adapter ---
    target_modules = [m.strip() for m in args.lora_target_modules.split(",")]
    lora_config = LoraConfig(
        r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian",
        target_modules=target_modules, task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # --- Optimizer ---
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps
            * args.train_batch_size * accelerator.num_processes
        )

    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    if args.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        trainable_params, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay, eps=args.adam_epsilon,
    )

    # --- Dataset ---
    if args.dataset_name is not None:
        dataset = load_dataset(args.dataset_name, args.dataset_config_name, cache_dir=args.cache_dir)
    else:
        from datasets import load_from_disk
        dataset = load_from_disk(args.train_data_dir)

    train_dataset = dataset["train"]
    column_names = train_dataset.column_names
    image_column = args.image_column if args.image_column in column_names else column_names[0]

    def preprocess_train(examples):
        processed = {"pixel_values": [], "instructions": [], "actions": []}
        for i in range(len(examples["observation/image"])):
            img_seq = examples["observation/image"][i]
            action_seq = examples["actions"][i]
            instruction = examples["language_instruction"][i]
            processed["pixel_values"].append(img_seq[0])
            processed["instructions"].append(instruction)
            processed["actions"].append(action_seq[0])
        return processed

    if args.max_train_samples is not None:
        train_dataset = train_dataset.shuffle(seed=args.seed).select(range(args.max_train_samples))
    train_dataset = train_dataset.with_transform(preprocess_train)

    val_dataset = None
    if "validation" in dataset:
        val_dataset = dataset["validation"].with_transform(preprocess_train)

    # --- Collate Function ---
    def collate_fn(examples):
        images = [ex["pixel_values"] for ex in examples]
        instructions = [ex["instructions"] for ex in examples]
        actions = [ex["actions"] for ex in examples]

        prompt_builder_fn = PurePromptBuilder if "v01" not in args.pretrained_model_name_or_path else VicunaV15ChatPromptBuilder

        input_ids_list, attention_mask_list, pixel_values_list, labels_list = [], [], [], []

        for img, instruction, action in zip(images, instructions, actions):
            prompt_builder = prompt_builder_fn("openvla")
            prompt_text = prompt_builder.build_prompt(instruction, "")

            input_ids = processor.tokenizer(
                prompt_text, truncation=True,
                max_length=processor.tokenizer.model_max_length,
                return_tensors="pt",
            )["input_ids"][0]

            action_tokens = torch.tensor(action_tokenizer.tokenize(action), dtype=torch.long)
            full_input_ids = torch.cat([input_ids, action_tokens], dim=0)
            labels = torch.cat([torch.full_like(input_ids, -100), action_tokens], dim=0)
            attention_mask = torch.ones_like(full_input_ids)
            pixel_values = processor.image_processor.apply_transform(img)

            input_ids_list.append(full_input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)
            pixel_values_list.append(pixel_values)

        max_len = max(len(ids) for ids in input_ids_list)
        pad_token_id = processor.tokenizer.pad_token_id

        padded_input_ids, padded_attention_mask, padded_labels = [], [], []
        for input_ids, attn_mask, labels in zip(input_ids_list, attention_mask_list, labels_list):
            pad_len = max_len - len(input_ids)
            padded_input_ids.append(torch.cat([input_ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)]))
            padded_attention_mask.append(torch.cat([attn_mask, torch.zeros(pad_len, dtype=torch.long)]))
            padded_labels.append(torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)]))

        return {
            "pixel_values": torch.stack(pixel_values_list),
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "labels": torch.stack(padded_labels),
        }

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.dataloader_num_workers,
    )

    # --- LR Scheduler ---
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    from transformers import get_scheduler as get_transformers_scheduler
    lr_scheduler = get_transformers_scheduler(
        args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        tracker_config = vars(args)
        init_kwargs = {}
        if args.wandb_run_name:
            init_kwargs["wandb"] = {"name": args.wandb_run_name}
        accelerator.init_trackers(
            project_name=args.wandb_project if args.report_to == "wandb" else "openvla-fine-tune",
            config=tracker_config, init_kwargs=init_kwargs,
        )

    # --- Training Loop ---
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        path = args.resume_from_checkpoint
        if path == "latest":
            dirs = sorted([d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")],
                          key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if dirs else None
        if path:
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(os.path.basename(path).split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(range(0, args.max_train_steps), initial=global_step,
                        desc="Steps", disable=not accelerator.is_local_main_process)

    for epoch in range(first_epoch, args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(
                        input_ids=batch["input_ids"].to(accelerator.device),
                        attention_mask=batch["attention_mask"].to(accelerator.device),
                        pixel_values=batch["pixel_values"].to(torch.bfloat16).to(accelerator.device),
                        labels=batch["labels"].to(accelerator.device),
                    )
                    loss = outputs.loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % 100 == 0 and torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        accelerator.log({
                            f"gpu_{i}_memory_allocated_gb": torch.cuda.memory_allocated(i) / (1024**3),
                            f"gpu_{i}_memory_reserved_gb": torch.cuda.memory_reserved(i) / (1024**3),
                        }, step=global_step)

                if (accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED) \
                        and global_step % args.checkpointing_steps == 0:
                    if args.checkpoints_total_limit is not None:
                        checkpoints = sorted(
                            [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")],
                            key=lambda x: int(x.split("-")[1]),
                        )
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            for ckpt in checkpoints[:len(checkpoints) - args.checkpoints_total_limit + 1]:
                                shutil.rmtree(os.path.join(args.output_dir, ckpt))
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved checkpoint to {save_path}")

            progress_bar.set_postfix({"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]})
            accelerator.log({
                "train/loss": loss.detach().item(),
                "train/learning_rate": lr_scheduler.get_last_lr()[0],
                "train/epoch": epoch, "train/global_step": global_step,
            }, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # --- Save Final Model ---
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Saving final model...")
        model_unwrapped = accelerator.unwrap_model(model)
        lora_state_dict = get_peft_model_state_dict(model_unwrapped)
        torch.save(lora_state_dict, os.path.join(args.output_dir, "pytorch_lora_weights.bin"))
        model_unwrapped.save_pretrained(args.output_dir)
        processor.save_pretrained(args.output_dir)
        logger.info(f"Model saved to {args.output_dir}")

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
