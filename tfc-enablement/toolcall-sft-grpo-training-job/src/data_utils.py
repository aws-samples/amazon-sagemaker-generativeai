"""
xLAM (single-turn function calling) -> rendered training/eval examples.

Deterministic, filter-consistent splits: every job that calls `usable_examples`
with the same seed sees the same ordered list of usable rows, so slicing yields
DISJOINT eval / GRPO / SFT sets without any job needing to share an index file.

Rendered forms:
  SFT   : {"prompt": <chat template up to assistant>, "completion": <tool call> + EOS}
  GRPO  : {"prompt": <same prompt>, "gold": <json answers>}
  eval  : {"prompt": <same prompt>, "gold": <json answers>, "query": <str>}
"""
from __future__ import annotations

import json
import logging
import random

import tool_reward as R

LOG = logging.getLogger("toolcall-job.data")


def render_prompt(tokenizer, query, tools, enable_thinking=False):
    messages = [
        {"role": "system", "content": R.SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True,
            tokenize=False, enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, tokenize=False,
        )


def build_completion(answers, tokenizer):
    parts = []
    for a in R.normalize_answers(answers):
        blob = json.dumps({"name": a["name"], "arguments": a["arguments"]}, ensure_ascii=False)
        parts.append(f"<tool_call>\n{blob}\n</tool_call>")
    return "\n".join(parts) + (tokenizer.eos_token or "")


def _usable(row):
    """A row is usable if it has at least one tool and one valid answer."""
    tools = R.coerce_tools(row.get("tools"))
    answers = R.normalize_answers(row.get("answers"))
    return bool(tools) and bool(answers)


def load_and_split(dataset_id, tokenizer, n_eval, n_grpo, n_train, seed, enable_thinking):
    """Load xLAM, deterministically shuffle+filter, render, return (eval, grpo, sft) lists.

    The three lists are disjoint. `sft` is further split into (train, val) by the caller.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split="train")
    LOG.info("loaded %s: %d rows", dataset_id, len(ds))

    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)

    need = n_eval + n_grpo + n_train
    rows = []
    scanned = 0
    for i in idx:
        row = ds[i]
        scanned += 1
        if not _usable(row):
            continue
        tools = R.coerce_tools(row["tools"])
        answers = R.normalize_answers(row["answers"])
        prompt = render_prompt(tokenizer, row["query"], tools, enable_thinking)
        rows.append({
            "prompt": prompt,
            "completion": build_completion(answers, tokenizer),
            "gold": json.dumps(answers, ensure_ascii=False),
            "query": row["query"],
        })
        if len(rows) >= need:
            break
    LOG.info("kept %d usable rows (scanned %d)", len(rows), scanned)

    eval_rows = rows[:n_eval]
    grpo_rows = rows[n_eval:n_eval + n_grpo]
    sft_rows = rows[n_eval + n_grpo:]
    return eval_rows, grpo_rows, sft_rows


def build_irrelevance(dataset_id, tokenizer, n, seed, enable_thinking):
    """Synthetic 'no applicable tool' set for measuring OVER-CALLING / abstention.

    Each example pairs a real query with tools borrowed from a DIFFERENT row, so none
    of the offered tools can answer it. Correct behavior = emit NO tool call. gold='[]'.
    We drop any pairing where a borrowed tool name coincides with the query's real
    function(s), to avoid accidentally-relevant tools.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split="train")
    idx = list(range(len(ds)))
    random.Random(seed + 1).shuffle(idx)  # offset so this differs from the train split

    usable = []
    for i in idx:
        row = ds[i]
        if _usable(row):
            usable.append(row)
        if len(usable) >= 2 * n + 50:
            break

    half = len(usable) // 2
    rows = []
    for a in range(min(n, half)):
        q_row, t_row = usable[a], usable[a + half]
        q_names = {c["name"] for c in R.normalize_answers(q_row["answers"])}
        t_tools = R.coerce_tools(t_row["tools"])
        t_names = {t["function"]["name"] for t in t_tools if isinstance(t, dict) and "function" in t}
        if not t_tools or (q_names & t_names):
            continue  # skip if borrowed tools are (accidentally) relevant
        prompt = render_prompt(tokenizer, q_row["query"], t_tools, enable_thinking)
        rows.append({"prompt": prompt, "gold": "[]", "query": q_row["query"]})
    LOG.info("built %d irrelevance (no-tool-needed) examples", len(rows))
    return rows


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
