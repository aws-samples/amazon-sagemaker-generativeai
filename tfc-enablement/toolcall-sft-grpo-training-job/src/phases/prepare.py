"""
Phase 0 — Prepare data.

Loads xLAM, renders single-turn tool-calling examples with the base model's chat
template (thinking disabled), and writes four disjoint JSONL splits into paths['data']:

    sft_train.jsonl   prompt + completion   (completion-only-loss SFT)
    sft_val.jsonl     prompt + completion   (small holdout for SFT eval)
    grpo_train.jsonl  prompt + gold         (RLVR prompts + gold answers)
    eval.jsonl        prompt + gold + query (held-out benchmark)

CPU-only (just tokenizer + formatting), so this can run on a cheap instance or locally.
"""
from __future__ import annotations

import logging

from transformers import AutoTokenizer

import data_utils

LOG = logging.getLogger("toolcall-job.prepare")


def run(cfg, paths) -> str:
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eval_rows, grpo_rows, sft_rows = data_utils.load_and_split(
        cfg["xlam_dataset"], tokenizer,
        n_eval=cfg["n_eval"], n_grpo=cfg["n_grpo"], n_train=cfg["n_train"],
        seed=cfg["seed"], enable_thinking=cfg["enable_thinking"],
    )

    n_val = max(1, int(0.02 * len(sft_rows)))
    sft_val, sft_train = sft_rows[:n_val], sft_rows[n_val:]

    import os

    data_utils.write_jsonl(os.path.join(paths["data"], "sft_train.jsonl"), sft_train,
                           ["prompt", "completion"])
    data_utils.write_jsonl(os.path.join(paths["data"], "sft_val.jsonl"), sft_val,
                           ["prompt", "completion"])
    data_utils.write_jsonl(os.path.join(paths["data"], "eval.jsonl"), eval_rows,
                           ["prompt", "gold", "query"])

    # Irrelevance ('no-tool-needed') examples. We build one set and split it DISJOINTLY:
    #   * eval slice  -> irrelevance.jsonl (held-out abstention metric)
    #   * train slice -> mixed into grpo_train.jsonl (gold='[]') so GRPO learns restraint,
    #                    which SFT (positive-only demos) structurally cannot.
    import random as _r

    n_eval_irr = cfg.get("n_irrelevance", 0)
    frac = cfg.get("grpo_irrelevance_frac", 0.0)
    n_grpo_irr = int(round(len(grpo_rows) * frac / (1 - frac))) if 0 < frac < 1 else 0

    grpo_mixed = [{"prompt": r["prompt"], "gold": r["gold"]} for r in grpo_rows]
    if n_eval_irr or n_grpo_irr:
        total_irr = data_utils.build_irrelevance(
            cfg["xlam_dataset"], tokenizer, n_eval_irr + n_grpo_irr, cfg["seed"], cfg["enable_thinking"])
        eval_irr = total_irr[:n_eval_irr]
        grpo_irr = total_irr[n_eval_irr:n_eval_irr + n_grpo_irr]
        if eval_irr:
            data_utils.write_jsonl(os.path.join(paths["data"], "irrelevance.jsonl"), eval_irr,
                                   ["prompt", "gold", "query"])
        grpo_mixed += [{"prompt": r["prompt"], "gold": "[]"} for r in grpo_irr]
        _r.Random(cfg["seed"]).shuffle(grpo_mixed)
        LOG.info("GRPO train = %d tool-needed + %d irrelevance (gold=[])",
                 len(grpo_rows), len(grpo_irr))

    data_utils.write_jsonl(os.path.join(paths["data"], "grpo_train.jsonl"), grpo_mixed,
                           ["prompt", "gold"])

    LOG.info("prepare done: sft_train=%d sft_val=%d grpo=%d eval=%d eval_irrelevance=%d",
             len(sft_train), len(sft_val), len(grpo_mixed), len(eval_rows), n_eval_irr)
    return paths["data"]
