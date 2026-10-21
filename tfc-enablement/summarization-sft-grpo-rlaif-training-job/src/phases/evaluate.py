"""
Phase 3 — Evaluate: base vs SFT vs SFT+GRPO (RLAIF).

Summarization has no exact-match ground truth, so "how do we evaluate SFT vs
SFT+GRPO?" is the whole problem. We use four complementary views, in decreasing
order of trust:

  1) PAIRWISE WIN-RATE (headline). For held-out posts, an LLM judge compares two
     summaries and picks the better one. We report win/tie/loss for:
        SFT+GRPO  vs  SFT        (did RL actually improve on imitation?)
        SFT+GRPO  vs  reference  (did RL beat the human/reference summary?)
        SFT       vs  base       (did imitation help at all?)
     A/B order is RANDOMIZED per example to cancel the judge's position bias.

  2) RUBRIC SCORES (absolute). The judge rates each summary 1-5 on coverage,
     faithfulness, coherence, conciseness; we report the mean per model (normalized
     to [0,1]) and the weighted mean_reward. Faithfulness is the summarization analog
     of the tool-calling "abstention" axis — a quality imitation of one reference
     can't guarantee but a reward can target.

  3) LENGTH (bias control). Mean summary length per model. If SFT+GRPO "wins" only by
     getting longer, that is verbosity reward-hacking, not real improvement — length
     makes that visible.

  4) ROUGE vs reference (sanity only). Cheap, reproducible, but weakly correlated with
     quality; reported so the audience sees why it is NOT the headline.

*** EVAL INTEGRITY ***
The EVAL judge (cfg['eval_judge_model']) must differ from the TRAIN judge
(cfg['train_judge_model']). Otherwise GRPO can lift the eval by learning to please its
own training judge. We assert this and warn loudly if they match. For real sign-off,
also validate the eval judge against a small human-labeled set with ORDINAL agreement
(see notebooks/06_judge_validation.ipynb) — nominal kappa understates an ordinal judge.

Single-process (one GPU); the eval set is small and this keeps it simple.
"""
from __future__ import annotations

import json
import logging
import os
import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import common
import data_utils
import judge_reward as J

LOG = logging.getLogger("summ-job.eval")
_COMPONENTS = ("coverage", "faithfulness", "coherence", "conciseness")


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
            # hygiene: strip <think> blocks, leaked chat turns, repetition — uniformly
            # across base / SFT / SFT+GRPO so no model is dragged by broken decoding.
            outs.append(data_utils.clean_summary(tok.decode(new, skip_special_tokens=True)))
    del model
    torch.cuda.empty_cache()
    return outs


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _rouge_l(pred, ref):
    """Length-normalized ROUGE-L F1 via LCS (no external dep)."""
    a, b = pred.lower().split(), ref.lower().split()
    if not a or not b:
        return 0.0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[len(a)][len(b)]
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(a), lcs / len(b)
    return 2 * prec * rec / (prec + rec)


def _rubric_scores(judge, posts, summaries, weights):
    comps = judge.score_batch(posts, summaries)
    out = {c: float(np.mean([x[c] for x in comps])) for c in _COMPONENTS}
    out["mean_reward"] = float(np.mean([J.total_reward(x, weights) for x in comps]))
    return out


def _win_rate(judge, posts, left, right):
    """Judge left vs right per post with randomized A/B order. Returns win/tie/loss
    fractions FOR LEFT, plus raw counts."""
    wins = ties = losses = 0
    for post, l, r in zip(posts, left, right):
        left_is_a = random.random() < 0.5
        a, b = (l, r) if left_is_a else (r, l)
        verdict = judge.compare(post, a, b)
        if verdict == "tie":
            ties += 1
        else:
            a_won = verdict == "A"
            left_won = (a_won and left_is_a) or ((not a_won) and (not left_is_a))
            wins += 1 if left_won else 0
            losses += 0 if left_won else 1
    n = max(1, wins + ties + losses)
    return {"win": wins / n, "tie": ties / n, "loss": losses / n,
            "wins": wins, "ties": ties, "losses": losses, "n": n}


def run(cfg, paths, model_specs) -> str:
    """model_specs: list of (label, path). Returns paths['results']."""
    eval_rows = data_utils.read_jsonl(os.path.join(paths["data"], "eval.jsonl"))
    prompts = [r["prompt"] for r in eval_rows]
    posts = [r["document"] for r in eval_rows]
    refs = [r["reference"] for r in eval_rows]
    weights = cfg["reward_weights"]
    rng = random.Random(cfg["seed"])
    random.seed(cfg["seed"])  # for the A/B order coin flips

    # --- eval judge, kept distinct from the training judge -----------------
    if cfg["eval_judge_model"] == cfg["train_judge_model"] and cfg["judge_backend"] != "mock":
        LOG.warning("!! EVAL judge == TRAIN judge (%s). GRPO can game its own judge; "
                    "the win-rate may be inflated. Use a different --eval-judge-model.",
                    cfg["eval_judge_model"])
    eval_judge = J.JudgeClient(
        backend=cfg["judge_backend"], model=cfg["eval_judge_model"],
        region=cfg["judge_region"] or common.aws_region(),
        max_concurrency=cfg["judge_max_concurrency"],
    )

    LOG.info("evaluating %d examples across %d models (eval judge=%s)",
             len(eval_rows), len(model_specs), cfg["eval_judge_model"])

    # --- generate + absolute metrics per model -----------------------------
    gens = {}
    results = []
    for label, path in model_specs:
        LOG.info("generating: %s (%s)", label, path)
        summaries = _generate_all(path, prompts, cfg["seq_len"],
                                  cfg["eval_max_new_tokens"], cfg["eval_batch_size"])
        gens[label] = summaries
        row = {"model": label}
        row.update(_rubric_scores(eval_judge, posts, summaries, weights))
        row["rouge_l"] = float(np.mean([_rouge_l(s, r) for s, r in zip(summaries, refs)]))
        row["len_words"] = float(np.mean([len(s.split()) for s in summaries]))
        results.append(row)
        LOG.info("  %s", row)

    # --- pairwise win-rates (headline) on a capped slice -------------------
    k = min(cfg["n_pairwise"], len(eval_rows))
    idx = list(range(len(eval_rows)))
    rng.shuffle(idx)
    idx = idx[:k]
    p_posts = [posts[i] for i in idx]
    labels = {lbl for lbl, _ in model_specs}

    pairwise = {}
    if "SFT+GRPO" in labels and "SFT" in labels:
        pairwise["SFT+GRPO_vs_SFT"] = _win_rate(
            eval_judge, p_posts, [gens["SFT+GRPO"][i] for i in idx], [gens["SFT"][i] for i in idx])
    if "SFT+GRPO" in labels:
        pairwise["SFT+GRPO_vs_reference"] = _win_rate(
            eval_judge, p_posts, [gens["SFT+GRPO"][i] for i in idx], [refs[i] for i in idx])
    if "SFT" in labels and "base" in labels:
        pairwise["SFT_vs_base"] = _win_rate(
            eval_judge, p_posts, [gens["SFT"][i] for i in idx], [gens["base"][i] for i in idx])
    for name, wr in pairwise.items():
        LOG.info("pairwise %s: win=%.3f tie=%.3f loss=%.3f (n=%d)",
                 name, wr["win"], wr["tie"], wr["loss"], wr["n"])

    # --- no-regression guardrail: ship SFT+GRPO only if it beats SFT head-to-head
    #     AND doesn't regress faithfulness / coverage / coherence / ROUGE-L vs SFT ----
    guardrail = _guardrail(results, pairwise, eps=0.01)
    if guardrail:
        LOG.info("GUARDRAIL: beats_sft_pairwise=%s  no_regression=%s  -> SHIP: %s",
                 guardrail["beats_sft_pairwise"], guardrail["no_regression"],
                 guardrail["recommended_model"])
        LOG.info("  deltas vs SFT (want >= -%.2f): %s", guardrail["epsilon"],
                 guardrail["deltas_vs_sft"])

    # --- persist -----------------------------------------------------------
    metrics = {"per_model": results, "pairwise": pairwise, "guardrail": guardrail,
               "train_judge": cfg["train_judge_model"], "eval_judge": cfg["eval_judge_model"],
               "reward_weights": weights, "n_eval": len(eval_rows), "n_pairwise": k}
    with open(os.path.join(paths["results"], "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    _write_csv(os.path.join(paths["results"], "metrics.csv"), results)
    _save_samples(os.path.join(paths["results"], "samples.json"), eval_rows, gens, n=50)
    _plot(os.path.join(paths["results"], "comparison.png"), results, pairwise)

    LOG.info("eval done -> %s", paths["results"])
    for r in results:
        LOG.info("RESULT %s", r)
    return paths["results"]


def _guardrail(results, pairwise, eps=0.01):
    """Decide whether SFT+GRPO is safe to ship: it must beat SFT head-to-head AND not
    regress faithfulness / coverage / coherence / ROUGE-L vs SFT. Otherwise recommend SFT.
    This enforces 'improve pairwise WITHOUT making other aspects worse'."""
    by = {r["model"]: r for r in results}
    if "SFT+GRPO" not in by or "SFT" not in by:
        return {}
    g, s = by["SFT+GRPO"], by["SFT"]
    wr = pairwise.get("SFT+GRPO_vs_SFT", {})
    beats = wr.get("win", 0.0) > wr.get("loss", 0.0)
    guarded_axes = ("faithfulness", "coverage", "coherence", "rouge_l")
    deltas = {ax: round(g.get(ax, 0.0) - s.get(ax, 0.0), 4) for ax in guarded_axes}
    no_regression = all(d >= -eps for d in deltas.values())
    passed = beats and no_regression
    return {
        "beats_sft_pairwise": beats,
        "win_minus_loss": round(wr.get("win", 0.0) - wr.get("loss", 0.0), 3),
        "deltas_vs_sft": deltas,
        "no_regression": no_regression,
        "passed": passed,
        "recommended_model": "SFT+GRPO" if passed else "SFT",
        "epsilon": eps,
    }


def _write_csv(path, results):
    if not results:
        return
    cols = list(results[0].keys())
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in results:
            f.write(",".join(str(r[c]) for c in cols) + "\n")


def _save_samples(path, eval_rows, gens, n=20):
    rows = []
    for i in range(min(n, len(eval_rows))):
        rows.append({
            "post": eval_rows[i]["document"],   # FULL post — truncating would make labelers (and the judge) misjudge faithfulness vs the cut-off tail
            "reference": eval_rows[i]["reference"],
            "generations": {label: g[i] for label, g in gens.items()},
        })
    with open(path, "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def _plot(path, results, pairwise):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        LOG.warning("matplotlib unavailable (%s); skipping chart.", e)
        return

    metrics = list(_COMPONENTS) + ["mean_reward", "rouge_l"]
    x = np.arange(len(metrics))
    w = 0.8 / max(len(results), 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    for i, row in enumerate(results):
        ax1.bar(x + i * w, [row[m] for m in metrics], w, label=row["model"])
    ax1.set_xticks(x + w * (len(results) - 1) / 2)
    ax1.set_xticklabels(metrics, rotation=20)
    ax1.set_ylabel("score")
    ax1.set_ylim(0, 1)
    ax1.set_title("Summarization (RLAIF): absolute rubric + ROUGE")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    if pairwise:
        names = list(pairwise.keys())
        wins = [pairwise[n]["win"] for n in names]
        ties = [pairwise[n]["tie"] for n in names]
        losses = [pairwise[n]["loss"] for n in names]
        yy = np.arange(len(names))
        ax2.barh(yy, wins, label="win", color="#2a9d8f")
        ax2.barh(yy, ties, left=wins, label="tie", color="#e9c46a")
        ax2.barh(yy, losses, left=[w_ + t for w_, t in zip(wins, ties)], label="loss", color="#e76f51")
        ax2.set_yticks(yy)
        ax2.set_yticklabels(names)
        ax2.set_xlim(0, 1)
        ax2.set_xlabel("fraction")
        ax2.set_title("LLM-judge pairwise win-rate (order-randomized)")
        ax2.legend(loc="lower right")
    else:
        ax2.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    LOG.info("saved chart -> %s", path)
