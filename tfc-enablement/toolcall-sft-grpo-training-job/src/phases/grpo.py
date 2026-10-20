"""
Phase 2 — GRPO (RLVR): teach correctness on top of the SFT model.

This is Reinforcement Learning from Verifiable Rewards: the reward is the
deterministic verifier in tool_reward.py (format + name + param-name + param-value),
NOT a learned reward model. TRL GRPOTrainer samples `num_generations` completions
per prompt, scores each with the verifier, and optimizes relative to the group mean.

DDP-aware (runs under torchrun via src/grpo.sh): each rank is a full replica and
generates for its shard of prompts. With LoRA the KL reference is free — TRL disables
the adapter to get reference logprobs rather than holding a second model copy.
"""
from __future__ import annotations

import logging
import os

import torch
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

import common
import tool_reward as R

LOG = logging.getLogger("toolcall-job.grpo")
ADAPTER_SUBDIR = "grpo_lora_adapter"


def _assert_generation_divisible(cfg, world_size):
    """TRL requires global_batch * grad_accum to be divisible by num_generations."""
    global_completions = cfg["grpo_bs"] * max(world_size, 1) * cfg["grpo_grad_accum"]
    ng = cfg["num_generations"]
    if global_completions % ng != 0:
        raise SystemExit(
            f"GRPO batch misconfigured: grpo_bs({cfg['grpo_bs']}) * world_size({world_size}) "
            f"* grad_accum({cfg['grpo_grad_accum']}) = {global_completions} is not divisible "
            f"by num_generations({ng}). Adjust --grpo-bs / --grpo-grad-accum / "
            f"--num-generations so it divides evenly."
        )
    LOG.info("GRPO batch OK: %d completions/step divisible by num_generations=%d",
             global_completions, ng)


def run(cfg, paths, sft_model_path) -> str:
    """GRPO-train on top of sft_model_path; save merged model to paths['grpo_model']."""
    out = paths["grpo_model"]
    use_lora = bool(cfg["use_lora"])
    local_rank, rank, world_size = common.dist_info()
    is_main = rank == 0

    _assert_generation_divisible(cfg, world_size)
    LOG.info("GRPO from %s | reward_weights=%s | rank %d/%d",
             sft_model_path, cfg["reward_weights"], rank, world_size)

    tokenizer = AutoTokenizer.from_pretrained(sft_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for generation

    from datasets import load_dataset

    grpo_ds = load_dataset("json", data_files=os.path.join(paths["data"], "grpo_train.jsonl"),
                           split="train")
    LOG.info("GRPO prompts: %d", len(grpo_ds))

    peft_config = None
    if use_lora:
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=0.05,
            bias="none", task_type="CAUSAL_LM", target_modules="all-linear",
        )

    grpo_config = common.make_trl_config(GRPOConfig, dict(
        output_dir=os.path.join(paths["checkpoints"], "grpo"),
        num_train_epochs=cfg["grpo_epochs"],
        per_device_train_batch_size=cfg["grpo_bs"],
        gradient_accumulation_steps=cfg["grpo_grad_accum"],
        num_generations=cfg["num_generations"],
        learning_rate=cfg["grpo_lr"],
        beta=cfg["grpo_beta"],
        temperature=cfg["grpo_temperature"],
        max_prompt_length=cfg["seq_len"] - cfg["max_completion_length"],
        max_completion_length=cfg["max_completion_length"],
        reward_weights=cfg["reward_weights"],  # weights the 4 reward funcs below
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        seed=cfg["seed"],
        use_vllm=cfg["use_vllm"],  # dropped by make_trl_config if the build lacks it
    ))

    trainer = GRPOTrainer(
        model=sft_model_path,
        args=grpo_config,
        train_dataset=grpo_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
        reward_funcs=R.GRPO_REWARD_FUNCS,  # [format, name, param_name, param_value]
    )
    trainer.train()

    if not use_lora:
        trainer.save_model(out)
        if is_main:
            tokenizer.save_pretrained(out)
        common.barrier()
        LOG.info("GRPO done -> %s", out)
        return out

    adapter_dir = os.path.join(paths["work"], ADAPTER_SUBDIR)
    if is_main:
        trainer.model.save_pretrained(adapter_dir)
        LOG.info("saved GRPO LoRA adapter -> %s", adapter_dir)
    common.barrier()

    if is_main:
        _merge_adapter(sft_model_path, adapter_dir, out, cfg["attn_impl"])
        tokenizer.save_pretrained(out)

    del trainer
    torch.cuda.empty_cache()
    common.barrier()
    LOG.info("GRPO done -> %s", out)
    return out


def _merge_adapter(sft_model_path, adapter_dir, out, attn_impl):
    """Merge the GRPO adapter into the (already SFT-merged) bf16 base on CPU."""
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    cpu_attn = common.resolve_attn_impl(attn_impl, on_cpu=True)
    LOG.info("merging GRPO adapter into bf16 SFT base on CPU (attn=%s) ...", cpu_attn)
    base = AutoModelForCausalLM.from_pretrained(
        sft_model_path, torch_dtype=torch.bfloat16, attn_implementation=cpu_attn,
        device_map=None, low_cpu_mem_usage=True, trust_remote_code=True,
    )
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.save_pretrained(out, safe_serialization=True)
    LOG.info("merged SFT+GRPO model saved -> %s", out)
    del merged, base
