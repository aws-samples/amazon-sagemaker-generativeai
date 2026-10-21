"""Shared argument parsing + cfg builder for the per-step job entry scripts.

Each step (prepare_job, sft_job, grpo_job, evaluate_job) reuses these so the phase
`cfg` dict always has every key the phases read, regardless of which step runs.

This is the RLAIF (RL from AI Feedback) twin of the tool-calling (RLVR) project:
the GRPO reward is an LLM-as-judge rubric score, not a deterministic verifier.
"""
from __future__ import annotations

import argparse


def add_pipeline_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # --- model ---
    p.add_argument("--base-model", default="Qwen/Qwen3.5-4B",
                   help="base model to fine-tune; a smaller base (e.g. Qwen/Qwen3.5-0.8B) "
                        "leaves more headroom for GRPO to show value")
    p.add_argument("--attn-impl", default="flash_attention_2",
                   choices=["flash_attention_2", "eager", "sdpa"])
    p.add_argument("--enable-thinking", type=int, default=0, choices=[0, 1],
                   help="Qwen3.5 thinking mode; 0 for clean short summaries")

    # --- data ---
    p.add_argument("--tldr-dataset", default="CarperAI/openai_summarize_tldr",
                   help="Reddit TL;DR summarization dataset (prompt + label/reference)")
    p.add_argument("--n-train", type=int, default=8000, help="SFT training examples")
    p.add_argument("--n-grpo", type=int, default=2000,
                   help="GRPO (RLAIF) prompts — kept modest; each rollout is judge-scored")
    p.add_argument("--n-eval", type=int, default=400, help="held-out eval examples")
    p.add_argument("--n-pairwise", type=int, default=200,
                   help="held-out examples used for LLM-judge pairwise win-rate")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)

    # --- LoRA (used by both SFT and GRPO) ---
    p.add_argument("--use-lora", type=int, default=1, choices=[0, 1])
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)

    # --- SFT ---
    p.add_argument("--sft-epochs", type=float, default=1.0)
    p.add_argument("--sft-lr", type=float, default=2e-4)
    p.add_argument("--sft-bs", type=int, default=4, help="per-device train batch size")
    p.add_argument("--sft-grad-accum", type=int, default=4)

    # --- GRPO / RLAIF ---
    p.add_argument("--grpo-epochs", type=float, default=1.0)
    # As in the RLVR twin: 1e-6 can leave the greedy policy glued to SFT (KL ~0).
    # 5e-6 moves it enough to commit the judge-preferred behavior into greedy decoding.
    p.add_argument("--grpo-lr", type=float, default=5e-6)
    p.add_argument("--grpo-bs", type=int, default=8,
                   help="per-device completions per step; global*accum must be divisible "
                        "by --num-generations")
    p.add_argument("--grpo-grad-accum", type=int, default=4)
    p.add_argument("--num-generations", type=int, default=8, help="GRPO group size")
    p.add_argument("--grpo-beta", type=float, default=0.01,
                   help="KL penalty to the SFT ref (low to allow drift toward judge preference)")
    p.add_argument("--grpo-temperature", type=float, default=1.0)
    p.add_argument("--max-completion-length", type=int, default=128,
                   help="summaries are short; cap tokens to discourage length hacking")

    # --- judge (RLAIF reward) ---
    p.add_argument("--judge-backend", default="bedrock",
                   choices=["bedrock", "local", "mock"],
                   help="where the AI judge runs: bedrock (API), local (HF model on GPU), "
                        "or mock (deterministic heuristic for smoke tests / CI)")
    p.add_argument("--train-judge-model", default="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                   help="judge model used as the GRPO TRAINING reward")
    # IMPORTANT: the eval judge should DIFFER from the train judge (ideally a different
    # family/size) so GRPO can't win by learning to please its own training judge.
    p.add_argument("--eval-judge-model", default="openai.gpt-oss-120b-1:0",
                   help="judge model used only for EVALUATION; must differ from "
                        "--train-judge-model to avoid judge self-collusion / reward hacking. "
                        "gpt-oss-120b is OpenAI family — independent of BOTH the Sonnet train "
                        "judge and the Qwen policy. It's a reasoning model (judge calls use a "
                        "larger token budget + low reasoning effort). Validate vs humans "
                        "(notebooks/06_judge_validation) before trusting the win-rate.")
    p.add_argument("--judge-region", default=None,
                   help="AWS region for the Bedrock judge (default: job region)")
    p.add_argument("--judge-max-concurrency", type=int, default=8,
                   help="parallel judge calls (Bedrock); lower if you hit throttling")
    # reward weights for --reward-mode rubric: [coverage, faithfulness, coherence, conciseness]
    p.add_argument("--reward-weights", default="0.30,0.40,0.15,0.15",
                   help="rubric-mode weights for [coverage,faithfulness,coherence,conciseness]")

    # --- reward mode -------------------------------------------------------
    p.add_argument("--reward-mode", default="guarded", choices=["rubric", "guarded"],
                   help="rubric = absolute 4-axis reward (original); "
                        "guarded = holistic preference-vs-reference with faithfulness/coverage "
                        "floors and NO conciseness term (improves pairwise without regressing)")
    # guarded blend: reward = w_pref*preference + w_faith*faithfulness + w_cov*coverage
    p.add_argument("--guard-weights", default="0.45,0.30,0.25",
                   help="guarded-mode weights for [preference, faithfulness, coverage]")
    p.add_argument("--faith-gate", type=float, default=0.40,
                   help="guarded-mode: if faithfulness < this, multiply reward by --gate-mult "
                        "(hard anti-hallucination gate)")
    p.add_argument("--gate-mult", type=float, default=0.30,
                   help="guarded-mode reward multiplier when the faithfulness gate trips")
    p.add_argument("--compare-against", default="sft", choices=["sft", "reference"],
                   help="guarded-mode: what each rollout's preference is judged against — "
                        "'sft' (the SFT model's own output, aligns with the pairwise-vs-SFT "
                        "eval) or 'reference' (the gold summary)")

    # --- eval ---
    p.add_argument("--eval-base", type=int, default=1, choices=[0, 1],
                   help="also evaluate the untuned base model for context")
    p.add_argument("--eval-max-new-tokens", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=16)
    return p


def parse_reward_weights(s: str) -> list:
    w = [float(x) for x in str(s).split(",")]
    if len(w) != 4:
        raise SystemExit(f"--reward-weights needs 4 values "
                         f"[coverage,faithfulness,coherence,conciseness], got {s!r}")
    return w


def cfg_from_args(args) -> dict:
    return {
        "base_model": args.base_model,
        "attn_impl": args.attn_impl,
        "enable_thinking": bool(args.enable_thinking),
        "tldr_dataset": args.tldr_dataset,
        "n_train": args.n_train,
        "n_grpo": args.n_grpo,
        "n_eval": args.n_eval,
        "n_pairwise": args.n_pairwise,
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
        "judge_backend": args.judge_backend,
        "train_judge_model": args.train_judge_model,
        "eval_judge_model": args.eval_judge_model,
        "judge_region": args.judge_region,
        "judge_max_concurrency": args.judge_max_concurrency,
        "reward_weights": parse_reward_weights(args.reward_weights),
        "reward_mode": args.reward_mode,
        "guard_weights": [float(x) for x in str(args.guard_weights).split(",")],
        "faith_gate": args.faith_gate,
        "gate_mult": args.gate_mult,
        "compare_against": args.compare_against,
        "eval_base": bool(args.eval_base),
        "eval_max_new_tokens": args.eval_max_new_tokens,
        "eval_batch_size": args.eval_batch_size,
    }
