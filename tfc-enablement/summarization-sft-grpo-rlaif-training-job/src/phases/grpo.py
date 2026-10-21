"""
Phase 2 — GRPO (RLAIF): optimize summary quality on top of the SFT model.

This is Reinforcement Learning from AI Feedback: the reward is an LLM-as-judge
rubric score (judge_reward.py), NOT a deterministic verifier and NOT a human label.
TRL GRPOTrainer samples `num_generations` summaries per prompt; each is scored by the
TRAINING judge; GRPO optimizes relative to the group mean.

DDP-aware (runs under torchrun via src/grpo.sh): each rank is a full replica and
generates for its shard of prompts. With LoRA the KL reference is free — TRL disables
the adapter to get reference logprobs rather than holding a second model copy.

Note on cost: every rollout completion is a judge call. Keep --n-grpo modest and use
--judge-max-concurrency to parallelize. The verdicts are cached per (post, summary) so
the four component reward funcs judge each completion once, not four times.
"""
from __future__ import annotations

import logging
import os

import torch
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

import common
import judge_reward as J

LOG = logging.getLogger("summ-job.grpo")
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


@torch.no_grad()
def _sft_baselines(cfg, paths, sft_model_path, prompts, is_main, world_size):
    """Greedy-decode the SFT model's own summary for each GRPO prompt (the comparison
    baseline for compare_against='sft'). Rank 0 generates and writes the file ATOMICALLY;
    every rank waits for it by POLLING the filesystem.

    We can't use dist.barrier() here: this runs BEFORE the GRPOTrainer builds the
    accelerate/TRL process group, so torch.distributed isn't initialized yet and the
    barrier would be a no-op — letting non-main ranks race ahead to a missing file.
    """
    import json
    import time

    import data_utils
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache = os.path.join(paths["data"], "grpo_baselines.jsonl")

    if is_main and not os.path.exists(cache):
        LOG.info("generating SFT baselines for %d GRPO prompts (rank 0 only) ...", len(prompts))
        tok = AutoTokenizer.from_pretrained(sft_model_path, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        attn = common.resolve_attn_impl(cfg["attn_impl"])
        # Pin to this process's own GPU (cuda:local_rank), not device_map="auto",
        # so we don't fan the 0.8B across every rank's GPU.
        local_rank = max(0, common.dist_info()[0])
        model = AutoModelForCausalLM.from_pretrained(
            sft_model_path, torch_dtype=torch.bfloat16, attn_implementation=attn,
            device_map={"": local_rank}, trust_remote_code=True)
        model.eval()
        bs, outs = cfg["eval_batch_size"], []
        for i in range(0, len(prompts), bs):
            batch = prompts[i:i + bs]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=cfg["seq_len"]).to(model.device)
            gen = model.generate(**enc, max_new_tokens=cfg["max_completion_length"],
                                 do_sample=False, pad_token_id=tok.pad_token_id)
            for j in range(len(batch)):
                new = gen[j][enc["input_ids"].shape[1]:]
                outs.append(data_utils.clean_summary(tok.decode(new, skip_special_tokens=True)))
        tmp = cache + ".tmp"
        with open(tmp, "w") as f:
            for s in outs:
                f.write(json.dumps({"baseline": s}, ensure_ascii=False) + "\n")
        os.replace(tmp, cache)  # atomic: other ranks only ever see the complete file
        del model
        torch.cuda.empty_cache()
        LOG.info("wrote SFT baselines -> %s", cache)

    # All ranks (including rank 0 if the cache already existed) wait for the file.
    deadline = time.time() + 2 * 3600
    while not os.path.exists(cache):
        if time.time() > deadline:
            raise SystemExit("timed out waiting for SFT baselines cache (rank %d)" % common.dist_info()[1])
        time.sleep(5)

    with open(cache) as f:
        rows = [json.loads(line)["baseline"] for line in f if line.strip()]
    if len(rows) != len(prompts):
        raise SystemExit(f"SFT-baseline count {len(rows)} != #prompts {len(prompts)}")
    return rows


def run(cfg, paths, sft_model_path) -> str:
    """GRPO-train on top of sft_model_path; save merged model to paths['grpo_model']."""
    out = paths["grpo_model"]
    use_lora = bool(cfg["use_lora"])
    local_rank, rank, world_size = common.dist_info()
    is_main = rank == 0

    _assert_generation_divisible(cfg, world_size)

    # Install the TRAINING judge used by the reward functions.
    judge = J.JudgeClient(
        backend=cfg["judge_backend"],
        model=cfg["train_judge_model"],
        region=cfg["judge_region"] or common.aws_region(),
        max_concurrency=cfg["judge_max_concurrency"],
    )
    J.set_default_client_reset_caches(judge)

    # Reward mode: 'rubric' = original absolute 4-axis; 'guarded' = holistic preference
    # vs reference with faithfulness/coverage floors and no conciseness term.
    if cfg["reward_mode"] == "guarded":
        gw = cfg["guard_weights"]
        J.set_guard_params(w_pref=gw[0], w_faith=gw[1], w_cov=gw[2],
                           faith_gate=cfg["faith_gate"], gate_mult=cfg["gate_mult"])
        reward_funcs = [J.reward_guarded]
        reward_weights = [1.0]
    else:
        reward_funcs = J.GRPO_REWARD_FUNCS
        reward_weights = cfg["reward_weights"]
    LOG.info("GRPO from %s | judge=%s/%s | reward_mode=%s weights=%s | rank %d/%d",
             sft_model_path, cfg["judge_backend"], cfg["train_judge_model"],
             cfg["reward_mode"], reward_weights, rank, world_size)

    tokenizer = AutoTokenizer.from_pretrained(sft_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for generation

    from datasets import load_dataset

    grpo_ds = load_dataset("json", data_files=os.path.join(paths["data"], "grpo_train.jsonl"),
                           split="train")
    LOG.info("GRPO prompts: %d", len(grpo_ds))

    # Build the 'baseline' column the guarded reward judges each rollout against.
    #   compare_against='reference' -> baseline = gold reference (free, no generation)
    #   compare_against='sft'       -> baseline = the SFT model's OWN greedy summary,
    #                                   so training optimizes the SAME pairwise-vs-SFT
    #                                   target the eval measures.
    if cfg["reward_mode"] == "guarded":
        if cfg["compare_against"] == "reference":
            grpo_ds = grpo_ds.add_column("baseline", list(grpo_ds["reference"]))
        else:
            baselines = _sft_baselines(cfg, paths, sft_model_path, list(grpo_ds["prompt"]),
                                       is_main, world_size)
            grpo_ds = grpo_ds.add_column("baseline", baselines)
        LOG.info("guarded baseline = %s", cfg["compare_against"])

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
        reward_weights=reward_weights,  # matches reward_funcs chosen by reward_mode
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        seed=cfg["seed"],
    ))

    trainer = GRPOTrainer(
        model=sft_model_path,
        args=grpo_config,
        train_dataset=grpo_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
        reward_funcs=reward_funcs,  # guarded: [reward_guarded]; rubric: 4 component funcs
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
