"""
Reddit TL;DR (CarperAI/openai_summarize_tldr) -> rendered training/eval examples.

Deterministic, filter-consistent splits: every job that calls `load_and_split`
with the same seed sees the same ordered list of usable rows, so slicing yields
DISJOINT eval / GRPO / SFT sets without any job needing to share an index file.

Rendered forms:
  SFT   : {"prompt": <chat template up to assistant>, "completion": <reference> + EOS}
  GRPO  : {"prompt": <same prompt>, "document": <source post>}  (judge needs the post)
  eval  : {"prompt": <same prompt>, "document": <post>, "reference": <ref summary>}

The reference summary is the SFT imitation target. GRPO does NOT use it — the judge
scores the generated summary against the POST, which is what lets RL optimize a
quality (faithful coverage) that imitating one reference cannot guarantee.
"""
from __future__ import annotations

import json
import logging
import random

import judge_reward as J

LOG = logging.getLogger("summ-job.data")

# The CarperAI TL;DR 'prompt' field ends with this marker; strip it to recover the post.
_TLDR_MARKER = "TL;DR:"


def clean_post(prompt: str) -> str:
    """Recover the raw post text from the dataset's completion-style prompt."""
    p = prompt or ""
    idx = p.rfind(_TLDR_MARKER)
    if idx != -1:
        p = p[:idx]
    return p.strip()


def render_prompt(tokenizer, post, enable_thinking=False):
    messages = [
        {"role": "system", "content": J.SYSTEM_PROMPT_POLICY},
        {"role": "user", "content": f"{J.TASK_INSTRUCTION}\n\nPOST:\n{post}"},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )


def build_completion(reference, tokenizer):
    return (reference or "").strip() + (tokenizer.eos_token or "")


def _usable(row):
    """A row is usable if it has a non-trivial post and a non-trivial reference."""
    post = clean_post(row.get("prompt", ""))
    ref = (row.get("label") or "").strip()
    return len(post) > 50 and len(ref) > 5


def load_and_split(dataset_id, tokenizer, n_eval, n_grpo, n_train, seed, enable_thinking):
    """Load TL;DR, deterministically shuffle+filter, render, return (eval, grpo, sft).

    The three lists are disjoint. `sft` is further split into (train, val) by the caller.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split="train")
    LOG.info("loaded %s: %d rows", dataset_id, len(ds))

    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)

    need = n_eval + n_grpo + n_train
    rows, scanned = [], 0
    for i in idx:
        row = ds[i]
        scanned += 1
        if not _usable(row):
            continue
        post = clean_post(row["prompt"])
        ref = row["label"].strip()
        prompt = render_prompt(tokenizer, post, enable_thinking)
        rows.append({
            "prompt": prompt,
            "completion": build_completion(ref, tokenizer),
            "document": post,
            "reference": ref,
        })
        if len(rows) >= need:
            break
    LOG.info("kept %d usable rows (scanned %d)", len(rows), scanned)

    eval_rows = rows[:n_eval]
    grpo_rows = rows[n_eval:n_eval + n_grpo]
    sft_rows = rows[n_eval + n_grpo:]
    return eval_rows, grpo_rows, sft_rows


def clean_summary(text: str) -> str:
    """Strip generation-hygiene junk: thinking blocks, leaked chat turns, echoed
    instructions, and consecutive duplicate lines. Applied uniformly to every model's
    output at eval + baseline time so comparisons aren't dragged by broken decoding."""
    if not text:
        return ""
    import re as _re
    t = _re.sub(r"<think>.*?</think>", " ", text, flags=_re.DOTALL | _re.IGNORECASE)
    t = t.replace("<think>", " ").replace("</think>", " ")
    # cut at the first leaked chat-turn marker
    lo = t.lower()
    cuts = [lo.find(m) for m in ("\nuser", "\nassistant", "\nsystem",
                                 "write a concise tl;dr", "summarize the following")]
    cuts = [c for c in cuts if c != -1]
    if cuts:
        t = t[:min(cuts)]
    # drop consecutive duplicate lines (repetition degeneration)
    lines, out = [ln.strip() for ln in t.splitlines() if ln.strip()], []
    for ln in lines:
        if not out or out[-1] != ln:
            out.append(ln)
    return " ".join(out).strip()


def write_jsonl(path, records, keys):
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps({k: r[k] for k in keys}, ensure_ascii=False) + "\n")
    LOG.info("wrote %s (%d rows)", path, len(records))


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
