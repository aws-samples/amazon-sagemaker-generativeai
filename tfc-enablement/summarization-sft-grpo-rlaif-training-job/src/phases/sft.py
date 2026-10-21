"""
Phase 1 — SFT: teach the summary format/style by imitating reference TL;DRs.

Runs under torchrun (see src/sft.sh), so this module is DDP-aware:
  * each rank loads its own full model replica (data parallel)
  * only rank 0 merges the LoRA adapter + saves; other ranks wait at a barrier

Without torchrun it still works as a single process.
"""
from __future__ import annotations

import logging
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

import common

LOG = logging.getLogger("summ-job.sft")
ADAPTER_SUBDIR = "lora_adapter"


def run(cfg, paths) -> str:
    """Fine-tune and save the merged model to paths['sft_model']. Returns that path."""
    base_model = cfg["base_model"]
    out = paths["sft_model"]
    use_lora = bool(cfg["use_lora"])
    local_rank, rank, world_size = common.dist_info()
    is_main = rank == 0

    LOG.info("SFT base=%s lora=%s | rank %d/%d (local_rank=%d)",
             base_model, use_lora, rank, world_size, local_rank)
    if world_size > 1:
        LOG.info("DDP: %d processes, each holding a full model replica.", world_size)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    from datasets import load_dataset

    train_ds = load_dataset("json", data_files=os.path.join(paths["data"], "sft_train.jsonl"),
                            split="train")
    val_file = os.path.join(paths["data"], "sft_val.jsonl")
    val_ds = load_dataset("json", data_files=val_file, split="train") if os.path.exists(val_file) else None
    LOG.info("SFT train=%d val=%s", len(train_ds), len(val_ds) if val_ds else 0)

    device_map = common.cuda_device_map(local_rank)
    attn_impl = common.resolve_attn_impl(cfg["attn_impl"], on_cpu=device_map is None)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, attn_implementation=attn_impl,
        device_map=device_map, trust_remote_code=True,
    )
    model.config.use_cache = False

    peft_config = None
    if use_lora:
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=0.05,
            bias="none", task_type="CAUSAL_LM", target_modules="all-linear",
        )

    sft_config = common.make_trl_config(SFTConfig, dict(
        output_dir=os.path.join(paths["checkpoints"], "sft"),
        num_train_epochs=cfg["sft_epochs"],
        per_device_train_batch_size=cfg["sft_bs"],
        gradient_accumulation_steps=cfg["sft_grad_accum"],
        learning_rate=cfg["sft_lr"] if use_lora else 1e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        logging_steps=10,
        save_strategy="no",
        eval_strategy="no",
        max_length=cfg["seq_len"],
        completion_only_loss=True,   # mask the post; train only on the summary
        packing=False,
        report_to="none",
        seed=cfg["seed"],
    ))

    trainer = SFTTrainer(
        model=model, args=sft_config, train_dataset=train_ds,
        eval_dataset=val_ds, processing_class=tokenizer, peft_config=peft_config,
    )
    if is_main and use_lora:
        trainer.model.print_trainable_parameters()
    trainer.train()

    if not use_lora:
        trainer.save_model(out)
        if is_main:
            tokenizer.save_pretrained(out)
        common.barrier()
        LOG.info("SFT done -> %s", out)
        return out

    adapter_dir = os.path.join(paths["work"], ADAPTER_SUBDIR)
    if is_main:
        trainer.model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(out)
        LOG.info("saved LoRA adapter -> %s", adapter_dir)
    common.barrier()

    if is_main:
        _merge_adapter(base_model, adapter_dir, out, cfg["attn_impl"])
        tokenizer.save_pretrained(out)

    del trainer, model
    torch.cuda.empty_cache()
    common.barrier()
    LOG.info("SFT done -> %s", out)
    return out


def _merge_adapter(base_model, adapter_dir, out, attn_impl):
    """Merge the LoRA adapter into bf16 base weights on CPU (clean, servable)."""
    from peft import PeftModel

    cpu_attn = common.resolve_attn_impl(attn_impl, on_cpu=True)
    LOG.info("merging adapter into bf16 base on CPU (attn=%s) ...", cpu_attn)
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, attn_implementation=cpu_attn,
        device_map=None, low_cpu_mem_usage=True, trust_remote_code=True,
    )
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.save_pretrained(out, safe_serialization=True)
    LOG.info("merged model saved -> %s", out)
    del merged, base
