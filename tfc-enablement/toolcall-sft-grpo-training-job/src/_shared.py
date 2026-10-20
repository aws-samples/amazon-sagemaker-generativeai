"""Shared argument parsing + cfg builder for the per-step job entry scripts.

Each step (prepare_job, sft_job, grpo_job, evaluate_job) reuses these so the phase
`cfg` dict always has every key the phases read, regardless of which step runs.
"""
from __future__ import annotations

import argparse


def add_pipeline_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # --- model ---
    p.add_argument("--base-model", default="Qwen/Qwen3.5-4B",
                   help="base model to fine-tune (e.g. Qwen/Qwen3.5-4B or Qwen/Qwen3.5-9B)")
    p.add_argument("--attn-impl", default="flash_attention_2",
                   choices=["flash_attention_2", "eager", "sdpa"])
    p.add_argument("--enable-thinking", type=int, default=0, choices=[0, 1],
                   help="Qwen3.5 thinking mode; 0 for clean single-turn tool calls")

    # --- data ---
    p.add_argument("--xlam-dataset", default="Salesforce/xlam-function-calling-60k")
    p.add_argument("--n-train", type=int, default=8000, help="SFT training examples")
    p.add_argument("--n-grpo", type=int, default=3000, help="GRPO (RLVR) prompts")
    p.add_argument("--n-eval", type=int, default=500, help="held-out eval examples")
    p.add_argument("--n-irrelevance", type=int, default=200,
                   help="held-out 'no-tool-needed' EVAL examples (over-calling/abstention)")
    p.add_argument("--grpo-irrelevance-frac", type=float, default=0.25,
                   help="fraction of the GRPO training set that is 'no-tool-needed' "
                        "(gold=[]) so RL learns restraint; 0 disables")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)

    # --- LoRA (used by both SFT and GRPO) ---
    p.add_argument("--use-lora", type=int, default=1, choices=[0, 1])
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)

    # --- SFT ---
    p.add_argument("--sft-epochs", type=float, default=1.0)
    p.add_argument("--sft-lr", type=float, default=2e-4)
    p.add_argument("--sft-bs", type=int, default=2, help="per-device train batch size")
    p.add_argument("--sft-grad-accum", type=int, default=8)

    # --- GRPO / RLVR ---
    p.add_argument("--grpo-epochs", type=float, default=1.0)
    # 1e-6 left the greedy policy glued to SFT (KL ~0.0008). 5e-6 moves it enough to
    # commit learned behavior (e.g. abstention) into greedy decoding. Raise to 1e-5 if
    # end-of-run KL is still < ~0.005; back off if exact_match drops.
    p.add_argument("--grpo-lr", type=float, default=5e-6)
    p.add_argument("--grpo-bs", type=int, default=8,
                   help="per-device completions per step; global*accum must be divisible "
                        "by --num-generations")
    p.add_argument("--grpo-grad-accum", type=int, default=4)
    p.add_argument("--num-generations", type=int, default=8, help="GRPO group size")
    p.add_argument("--grpo-beta", type=float, default=0.01,
                   help="KL penalty to the SFT ref (lowered from 0.04 to allow more drift)")
    p.add_argument("--grpo-temperature", type=float, default=1.0)
    p.add_argument("--max-completion-length", type=int, default=256)
    # reward weights: [format, name, param_name, param_value]
    p.add_argument("--reward-weights", default="0.1,0.3,0.2,0.4",
                   help="comma-separated weights for [format,name,param_name,param_value]")
    p.add_argument("--use-vllm", type=int, default=0, choices=[0, 1],
                   help="offload GRPO rollouts to vLLM (needs vLLM in the image; much faster)")

    # --- eval ---
    p.add_argument("--eval-base", type=int, default=1, choices=[0, 1],
                   help="also evaluate the untuned base model for context")
    p.add_argument("--eval-max-new-tokens", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=16)
    return p


def parse_reward_weights(s: str) -> list:
    w = [float(x) for x in str(s).split(",")]
    if len(w) != 4:
        raise SystemExit(f"--reward-weights needs 4 values, got {s!r}")
    return w


def cfg_from_args(args) -> dict:
    return {
        "base_model": args.base_model,
        "attn_impl": args.attn_impl,
        "enable_thinking": bool(args.enable_thinking),
        "xlam_dataset": args.xlam_dataset,
        "n_train": args.n_train,
        "n_grpo": args.n_grpo,
        "n_eval": args.n_eval,
        "n_irrelevance": args.n_irrelevance,
        "grpo_irrelevance_frac": args.grpo_irrelevance_frac,
        "seq_len": args.seq_len,
        "seed": args.seed,
        "use_lora": bool(args.use_lora),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "sft_epochs": args.sft_epochs,
        "sft_lr": args.sft_lr,
        "sft_bs": args.sft_bs,
        "sft_grad_accum": args.sft_grad_accum,
        "grpo_epochs": args.grpo_epochs,
        "grpo_lr": args.grpo_lr,
        "grpo_bs": args.grpo_bs,
        "grpo_grad_accum": args.grpo_grad_accum,
        "num_generations": args.num_generations,
        "grpo_beta": args.grpo_beta,
        "grpo_temperature": args.grpo_temperature,
        "max_completion_length": args.max_completion_length,
        "reward_weights": parse_reward_weights(args.reward_weights),
        "use_vllm": bool(args.use_vllm),
        "eval_base": bool(args.eval_base),
        "eval_max_new_tokens": args.eval_max_new_tokens,
        "eval_batch_size": args.eval_batch_size,
    }
