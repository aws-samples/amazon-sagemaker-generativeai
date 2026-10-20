"""
Phase 0 — Prepare data.

Loads TL;DR, renders summarization examples with the base model's chat template,
and writes disjoint JSONL splits into paths['data']:

    sft_train.jsonl   prompt + completion            (completion-only-loss SFT)
    sft_val.jsonl     prompt + completion            (small holdout for SFT eval)
    grpo_train.jsonl  prompt + document              (RLAIF prompts; judge needs the post)
    eval.jsonl        prompt + document + reference  (held-out benchmark)

CPU-only (just tokenizer + formatting), so this can run on a cheap instance or locally.
"""
from __future__ import annotations

import logging
import os

from transformers import AutoTokenizer

import data_utils

LOG = logging.getLogger("summ-job.prepare")


def run(cfg, paths) -> str:
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eval_rows, grpo_rows, sft_rows = data_utils.load_and_split(
        cfg["tldr_dataset"], tokenizer,
        n_eval=cfg["n_eval"], n_grpo=cfg["n_grpo"], n_train=cfg["n_train"],
        seed=cfg["seed"], enable_thinking=cfg["enable_thinking"],
    )

    n_val = max(1, int(0.02 * len(sft_rows)))
    sft_val, sft_train = sft_rows[:n_val], sft_rows[n_val:]

    data_utils.write_jsonl(os.path.join(paths["data"], "sft_train.jsonl"), sft_train,
                           ["prompt", "completion"])
    data_utils.write_jsonl(os.path.join(paths["data"], "sft_val.jsonl"), sft_val,
                           ["prompt", "completion"])
    # 'reference' is carried on GRPO rows too: the GUARDED reward judges each rollout
    # against the reference. (SFT/eval splits are unchanged, so the existing SFT model
    # stays valid — only this file gains a column.)
    data_utils.write_jsonl(os.path.join(paths["data"], "grpo_train.jsonl"), grpo_rows,
                           ["prompt", "document", "reference"])
    data_utils.write_jsonl(os.path.join(paths["data"], "eval.jsonl"), eval_rows,
                           ["prompt", "document", "reference"])

    LOG.info("prepare done: sft_train=%d sft_val=%d grpo=%d eval=%d",
             len(sft_train), len(sft_val), len(grpo_rows), len(eval_rows))
    return paths["data"]
