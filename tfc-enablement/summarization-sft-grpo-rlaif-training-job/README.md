# judge-the-summary

**SFT → GRPO (RLAIF) for TL;DR summarization — and how to evaluate SFT vs SFT+GRPO
when there is no exact-match ground truth.**

The RLAIF twin of the tool-calling (RLVR) project. There, the reward was a
deterministic verifier and "correct" was checkable. Here the task is open-ended —
a good summary has no single right answer — so the reward is an **LLM-as-judge**
(RL from AI Feedback) and the hard part moves to **evaluation**.

Base model: **Qwen/Qwen3.5-4B** (use **0.8B** for more GRPO headroom) ·
Data: **CarperAI/openai_summarize_tldr** (Reddit TL;DR) ·
Trainer: **TRL** (`SFTTrainer`, `GRPOTrainer`) · Parallelism: **torchrun DDP** ·
Judge: **Amazon Bedrock** (configurable) · Adapters: **LoRA** · Infra: **SageMaker**.

> Scope: single-document abstractive summarization. The two stages do different jobs:
> **SFT teaches the format/style by imitating reference summaries; GRPO/RLAIF pushes
> quality past what imitating one reference can reach.**

---

## RLVR vs RLAIF — what changed from the tool-calling project

| | tool-calling (RLVR) | summarization (RLAIF) — this repo |
|---|---|---|
| reward | deterministic verifier (`tool_reward.py`) | LLM-as-judge rubric (`judge_reward.py`) |
| "correct" | exact-match to gold call | judge's rubric score / pairwise preference |
| ground truth | yes (gold call) | no — only a reference summary (one of many) |
| headline eval | exact-match + abstention | **pairwise win-rate** + rubric + faithfulness |
| main risk | over-/under-calling | **reward hacking the judge** (verbosity, self-preference) |

Everything else — the per-step DDP jobs, S3 artifact flow, LoRA merge, the
one-job `entrypoint.py`, the notebook pipeline — is the same shape.

## The reward (RLAIF)

`src/judge_reward.py` — a judge model scores a `(post, summary)` pair on a rubric,
four components each normalized to `[0, 1]`, combined with weights
`[coverage, faithfulness, coherence, conciseness] = [0.30, 0.40, 0.15, 0.15]`:

| component | meaning |
|-----------|---------|
| `coverage` | captures the main point(s) of the post |
| `faithfulness` | everything stated is supported by the post (no hallucination) — weighted highest |
| `coherence` | fluent, well organized |
| `conciseness` | appropriately short for a TL;DR, no padding |

The judge runs on one of three backends (`--judge-backend`): `bedrock` (default),
`local` (an HF model on the box), or `mock` (a deterministic heuristic for offline
smoke tests / CI — never for real numbers). The four component reward functions
share a per-batch cache so each rollout is judged **once**, not four times.

## How we evaluate SFT vs SFT+GRPO (the crux)

There is no exact match, so evaluation uses four views, in decreasing order of trust
(`src/phases/evaluate.py`):

1. **Pairwise win-rate — the headline.** For held-out posts, the eval judge compares
   two summaries and picks the better one, with **A/B order randomized** to cancel
   position bias. We report win/tie/loss for:
   - `SFT+GRPO vs SFT` — did RL actually improve on imitation?
   - `SFT+GRPO vs reference` — did RL beat the human reference summary? (the classic
     "learning to summarize from feedback" result)
   - `SFT vs base` — did imitation help at all?
2. **Absolute rubric means** — coverage / faithfulness / coherence / conciseness and
   the weighted `mean_reward`. **Faithfulness is the summarization analog of the
   tool-calling "abstention" axis**: a safety-relevant quality that imitating one
   reference does not guarantee but a reward can target.
3. **Length (`len_words`) — a bias control.** If SFT+GRPO "wins" only by getting
   longer, that's verbosity reward-hacking, not improvement. Length makes it visible.
4. **ROUGE vs reference — sanity only.** Cheap and reproducible but weakly correlated
   with quality; reported so you can see why it is *not* the headline (RLAIF often
   lifts win-rate while ROUGE is flat).

### Two eval-integrity rules that make or break the result

- **The eval judge must differ from the training judge.** If they are the same model,
  GRPO can raise the score by learning to please its own judge — reward hacking that
  looks like real improvement. Defaults: train = Claude **Sonnet 4.5**
  (`us.anthropic.claude-sonnet-4-5-20250929-v1:0`), eval = **OpenAI gpt-oss-120b**
  (`openai.gpt-oss-120b-1:0`) — a **different-family** held-out judge, independent of both
  the Sonnet-trained reward and the Qwen policy, so a win can't come from pleasing a
  same-provider grader. (gpt-oss is a reasoning model; the judge calls use a larger token
  budget + low reasoning effort and read only the text block.) The code asserts the two
  judges differ and warns loudly if not. Validate the eval judge against humans
  (`notebooks/06_judge_validation`) before trusting the win-rate.
- **Validate the judge against humans before trusting it.** Hand-label ~30–50 pairs and
  measure **ordinal** agreement (Krippendorff's alpha ordinal / quadratic-weighted
  kappa) — never nominal kappa, which treats a 4-vs-5 miss as a total miss and makes a
  good judge look broken. See `notebooks/06_judge_validation.ipynb`.

## What "value of GRPO" looks like here

Same honest framing as the RLVR project. SFT imitates reference summaries — it learns
the format and a reference-like style, but it inherits whatever the references do
(including occasional unfaithfulness) and can't exceed them. GRPO/RLAIF optimizes the
judge's rubric over the model's *own* summaries, so it can:

- raise **faithfulness** above the reference-imitation ceiling, and
- win **head-to-head** against both the SFT model and the reference,

**without** getting longer (watch `len_words`). If win-rate rises only with length, or
only against your own training judge, that's a hack, not a result — which is exactly why
the eval is built the way it is.

## Repo layout

```
launch_job.py            one-job runner (prepare→SFT→GRPO→eval) via SageMaker ModelTrainer
src/
  judge_reward.py        the RLAIF reward: LLM-judge rubric + pairwise + backends (train==eval forbidden)
  data_utils.py          TL;DR load, disjoint splits, chat rendering
  common.py              logging, DDP/attn helpers, HF token, S3 sync, TRL-config compat
  _shared.py             pipeline args + cfg builder
  phases/                prepare.py, sft.py, grpo.py, evaluate.py (the actual logic)
  *.sh / *_job.py        per-step torchrun DDP entrypoints
notebooks/
  00_setup.ipynb         auto-detects role/region/bucket, prompts for HF token, writes config
  01_prepare.ipynb       builds + inspects data locally, uploads to S3
  02_sft.ipynb           launch SFT job (DDP)
  03_grpo.ipynb          launch GRPO/RLAIF job (DDP) — needs Bedrock access for the judge
  04_evaluate.ipynb      launch eval job, download metrics + win-rate chart
  05_rouge.ipynb         optional: ROUGE/BERTScore cross-check (why automatic metrics aren't enough)
  06_judge_validation.ipynb  optional: validate the eval judge vs humans (ordinal agreement)
tests/
  test_judge_reward.py   reward/parse/aggregation unit tests (CPU, no model, no Bedrock) — 10 tests
```

## Reproduce

```bash
pip install -r requirements-launch.txt      # client side
python tests/test_judge_reward.py            # sanity-check the reward (CPU, offline)
```

Then run the notebooks in order: `00_setup` → `01_prepare` → `02_sft` → `03_grpo` →
`04_evaluate` (optionally `05_rouge`, and `06_judge_validation` before you trust the
numbers). Role/region/bucket are auto-detected; you enter your HF token once in
`00_setup`. Data prep runs in-notebook; SFT/GRPO/eval run as SageMaker training jobs.

For a quick offline smoke test with no Bedrock and no GPU judge, pass
`--judge-backend mock` (deterministic heuristic reward) — useful to validate the
plumbing, not for real numbers.

## Requirements / gotchas

- **Bedrock access.** With `--judge-backend bedrock`, the SageMaker job role needs
  `bedrock:InvokeModel` on both judge models, in a region where they're enabled. Bedrock
  model IDs go stale — if an invoke 404s as "legacy," list active IDs and update
  `--train-judge-model` / `--eval-judge-model`.
- **Judge cost/latency.** Every GRPO rollout is a judge call (`num_generations` ×
  prompts × epochs). Keep `--n-grpo` modest, raise `--judge-max-concurrency`, and prefer
  checkpoint-by-eval over long runs.
- **GRPO needs reward variance.** As in the RLVR twin: if the judge gives every summary
  in a group the same score (`frac_reward_zero_std` high, entropy collapsed), there's no
  gradient — raise temperature / `num_generations`. And if end-of-run `KL ≈ 0`, the
  greedy policy (what eval uses) hasn't moved — raise `--grpo-lr` / lower `--grpo-beta`.

## Takeaway

With a verifiable task you check the answer; with an open-ended one you *judge* it — and
the judge becomes part of your threat model. The engineering that makes RLAIF honest is
mostly in the evaluation: a judge held out from training, order-randomized pairwise
comparisons, a length control, and judge-vs-human validation on an ordinal scale.
