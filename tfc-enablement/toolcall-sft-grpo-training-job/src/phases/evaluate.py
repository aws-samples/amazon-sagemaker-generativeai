"""
Phase 3 — Evaluate: SFT vs SFT+GRPO (and base for context).

Greedy-decodes each model on the held-out eval split and scores with the SAME
verifier used for the RLVR reward, so training and evaluation agree on "correct".
Writes metrics.json / metrics.csv / comparison.png into paths['results'].

Single-process (runs on one GPU); the eval set is small and this keeps it simple.
"""
from __future__ import annotations

import json
import logging
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import common
import data_utils
import tool_reward as R

LOG = logging.getLogger("toolcall-job.eval")


@torch.no_grad()
def _generate_all(model_path, prompts, seq_len, max_new_tokens, batch_size):
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    attn = common.resolve_attn_impl("flash_attention_2")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation=attn,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()

    outs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=seq_len).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        for j in range(len(batch)):
            new = gen[j][enc["input_ids"].shape[1]:]
            outs.append(tok.decode(new, skip_special_tokens=True))
    del model
    torch.cuda.empty_cache()
    return outs


def _score(name, completions, golds, weights):
    comps = [R.score_components(c, g) for c, g in zip(completions, golds)]
    exact = [R.is_exact_match(c, g) for c, g in zip(completions, golds)]
    reward = [R.total_reward(c, g, weights) for c, g in zip(completions, golds)]
    return {
        "model": name,
        "format": float(np.mean([x["format"] for x in comps])),
        "name_f1": float(np.mean([x["name"] for x in comps])),
        "param_name": float(np.mean([x["param_name"] for x in comps])),
        "param_value": float(np.mean([x["param_value"] for x in comps])),
        "mean_reward": float(np.mean(reward)),
        "exact_match": float(np.mean(exact)),
    }


def run(cfg, paths, model_specs) -> str:
    """model_specs: list of (label, path). Returns paths['results']."""
    eval_rows = data_utils.read_jsonl(os.path.join(paths["data"], "eval.jsonl"))
    prompts = [r["prompt"] for r in eval_rows]
    golds = [r["gold"] for r in eval_rows]
    weights = cfg["reward_weights"]

    # Optional over-calling / abstention slice: correct behavior = NO tool call.
    irr_file = os.path.join(paths["data"], "irrelevance.jsonl")
    irr_prompts = ([r["prompt"] for r in data_utils.read_jsonl(irr_file)]
                   if os.path.exists(irr_file) else [])
    LOG.info("evaluating %d examples (+%d irrelevance) across %d models",
             len(eval_rows), len(irr_prompts), len(model_specs))

    results, generations = [], {}
    for label, path in model_specs:
        LOG.info("generating: %s (%s)", label, path)
        gens = _generate_all(path, prompts, cfg["seq_len"],
                             cfg["eval_max_new_tokens"], cfg["eval_batch_size"])
        generations[label] = gens
        res = _score(label, gens, golds, weights)
        if irr_prompts:
            igens = _generate_all(path, irr_prompts, cfg["seq_len"],
                                  cfg["eval_max_new_tokens"], cfg["eval_batch_size"])
            # abstention = fraction where the model correctly made NO tool call
            res["abstention"] = float(np.mean(
                [1.0 if len(R.parse_tool_calls(g)) == 0 else 0.0 for g in igens]))
        results.append(res)
        LOG.info("  %s", res)

    # Persist metrics
    with open(os.path.join(paths["results"], "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    _write_csv(os.path.join(paths["results"], "metrics.csv"), results)
    _save_samples(os.path.join(paths["results"], "samples.json"), eval_rows, generations, n=20)
    _plot(os.path.join(paths["results"], "comparison.png"), results)

    LOG.info("eval done -> %s", paths["results"])
    for r in results:
        LOG.info("RESULT %s", r)
    return paths["results"]


def _write_csv(path, results):
    if not results:
        return
    cols = list(results[0].keys())
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in results:
            f.write(",".join(str(r[c]) for c in cols) + "\n")


def _save_samples(path, eval_rows, generations, n=20):
    rows = []
    for i in range(min(n, len(eval_rows))):
        rows.append({
            "query": eval_rows[i].get("query", ""),
            "gold": eval_rows[i]["gold"],
            "generations": {label: gens[i] for label, gens in generations.items()},
        })
    with open(path, "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def _plot(path, results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        LOG.warning("matplotlib unavailable (%s); skipping chart.", e)
        return

    metrics = ["format", "name_f1", "param_name", "param_value", "exact_match"]
    if results and all("abstention" in r for r in results):
        metrics.append("abstention")
    x = np.arange(len(metrics))
    w = 0.8 / max(len(results), 1)
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, row in enumerate(results):
        ax.bar(x + i * w, [row[m] for m in metrics], w, label=row["model"])
    ax.set_xticks(x + w * (len(results) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("score")
    ax.set_ylim(0, 1)
    ax.set_title("Single-turn tool calling: SFT vs SFT+GRPO (RLVR)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    LOG.info("saved chart -> %s", path)
