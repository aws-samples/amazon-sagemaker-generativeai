# know-when-to-call

**SFT → GRPO (RLVR) for single-turn tool calling — and what GRPO adds that SFT can't.**

Fine-tune a small model to call functions, then show *where reinforcement learning earns
its keep*: GRPO with a verifiable reward recovers a safety-relevant behavior — **knowing
when *not* to call a tool** — that supervised fine-tuning structurally destroys, **without
sacrificing accuracy**.

Base model: **Qwen/Qwen3.5-0.8B** · Data: **Salesforce/xlam-function-calling-60k** ·
Trainer: **TRL** (`SFTTrainer`, `GRPOTrainer`) · Parallelism: **torchrun DDP** ·
Infra: **SageMaker training jobs** · Adapters: **LoRA**.

> Scope: **single-turn** tool calling (one user query → one assistant turn, one or more
> parallel calls). Multi-turn orchestration is out of scope.

---

## TL;DR result

Three checkpoints — **base → SFT → SFT+GRPO** — on a 500-example held-out set (+200
held-out "no-tool-needed" prompts for the abstention metric):

| model | format | name F1 | param-name | param-value | mean reward | **exact match** | **abstention** |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| base       | 0.900 | 0.880 | 0.774 | 0.685 | 0.783 | 0.494 | 0.620 |
| SFT        | 0.996 | 0.975 | 0.952 | 0.849 | 0.922 | **0.670** | **0.050** |
| SFT+GRPO   | 0.992 | 0.974 | 0.957 | 0.866 | 0.929 | **0.692** | **0.905** |

**The story in two axes:**

- **Accuracy** — SFT is the big win: exact-match 0.494 → 0.670. GRPO holds and slightly
  improves it (→ 0.692).
- **Restraint (abstention)** — SFT *regresses catastrophically*: 0.620 → **0.050**. Training
  only on "a-tool-is-needed" examples teaches the model to **always call something**.
  **GRPO recovers it to 0.905** — above base — *while accuracy goes up, not down*.

That last row is the point of the project: **GRPO fixed a failure SFT cannot, at no
accuracy cost.** Supervised imitation can only reproduce behaviors present in its data;
RL with a verifiable reward can optimize a behavior (saying "no tool applies") that the
demonstrations never contain.

> Single seed, in-distribution (xLAM) + a synthetic irrelevance slice. The abstention
> delta (+0.855) is far beyond any plausible noise; the accuracy deltas are within noise
> and should be read as "held," not "improved." Multi-seed error bars are the natural next step.

---

## Why this is the honest "value of GRPO"

A common disappointment: on a capable base, **SFT already saturates single-turn tool
calling, and GRPO just ties it.** We saw exactly that on Qwen3.5-4B (GRPO ≈ SFT, no
headroom). The value of GRPO only becomes visible when:

1. **There's headroom** — we used the smaller **0.8B** base, which SFT does *not* saturate.
2. **You measure the right thing** — standard metrics (format/name/param/exact-match) all
   *improved* under SFT and completely hid the over-calling regression. The **abstention**
   metric is what exposed it.
3. **GRPO is given a behavior imitation can't teach** — "correctly make no call," which
   requires negative (no-tool-needed) examples and a reward that credits restraint.

## The verifiable reward (RLVR)

Deterministic verifier in [`src/tool_reward.py`](src/tool_reward.py) — no reward model, no
human labels. Per sample, four components in `[0,1]`, combined with weights
`[format, name, param_name, param_value] = [0.10, 0.30, 0.20, 0.40]`:

| component | meaning |
|-----------|---------|
| `format` | a parseable tool call was emitted (both Hermes-JSON and Qwen-XML accepted) |
| `name` | F1 of predicted vs gold function-name multiset |
| `param_name` | mean F1 of argument keys over name-matched calls |
| `param_value` | mean fraction of gold argument values reproduced correctly |

**Abstention case** (gold = `[]`, no tool should be called): a correct no-call scores **1.0**,
an over-call scores **0.1**. This is the signal that lets GRPO learn restraint.

The same verifier scores GRPO training *and* evaluation, so "correct" means one thing
everywhere.

---

## What it took to make GRPO actually work

This was not plug-and-play. The debugging arc is the real lesson:

1. **GRPO ties SFT on a saturated base.** On Qwen3.5-4B, SFT hit ~0.95 and GRPO had no
   reward variance to learn from. → Switched to the **0.8B** base for headroom.
2. **The regression was invisible until measured.** Added a held-out **irrelevance /
   abstention** slice; it revealed SFT's over-calling (abstention 0.62 → 0.05).
3. **GRPO needs reward *variance*.** First GRPO runs showed `entropy ≈ 0.05` and
   `frac_reward_zero_std ≈ 0.5` — half the prompt-groups produced identical rewards → zero
   gradient. → Raised sampling **temperature** and **mixed ~25% irrelevance (`gold=[]`)
   into GRPO training** (disjoint from the eval slice).
4. **GRPO needs enough step size.** Even with signal, `lr=1e-6` left `KL ≈ 0.0008` — the
   *greedy* policy (what eval uses) stayed ≈ SFT, so the eval didn't move even though
   training reward climbed. → **`grpo_lr 1e-6 → 5e-6`, `beta 0.04 → 0.01`**, targeting
   end-of-run `KL ≈ 0.02–0.1`. That's when abstention finally moved (0.075 → 0.905).

**Rules of thumb that fell out of this:**
- Watch `frac_reward_zero_std` (want it low) and `entropy` (want it non-collapsed) — they
  tell you if GRPO has anything to learn from.
- Watch `KL` — eval uses greedy decoding, so if KL ≈ 0 the greedy policy hasn't moved and
  your eval won't change regardless of training reward.
- Guardrail: if `exact_match` drops as `abstention` rises, the model is over-cautious —
  lower the LR or the irrelevance fraction. (Here exact-match went *up*, so no over-correction.)

---

## Final configuration

| | SFT | GRPO (RLVR) |
|---|---|---|
| base | Qwen3.5-0.8B | SFT-merged checkpoint |
| adapter | LoRA (r=16, α=32, all-linear) | LoRA (r=16, α=32, all-linear) |
| lr | 2e-4 | **5e-6** |
| epochs | 1 | 1 |
| extra | completion-only loss | `num_generations=8`, `beta=0.01`, raised temperature, **25% irrelevance mix**, reward weights `[0.1,0.3,0.2,0.4]` |

Data splits are deterministic and **disjoint** (SFT / GRPO / eval / irrelevance), driven by a
single seed — no shared index file needed.

---

## Repo layout

```
launch_job.py            one-job runner (prepare→SFT→GRPO→eval) via SageMaker ModelTrainer
src/
  tool_reward.py         the verifier: parsing + RLVR reward funcs + exact-match (train == eval)
  data_utils.py          xLAM load, disjoint splits, Qwen rendering, irrelevance builder
  common.py              logging, DDP/attn helpers, HF token, S3 sync, TRL-config compat
  _shared.py             pipeline args + cfg builder
  phases/                prepare.py, sft.py, grpo.py, evaluate.py (the actual logic)
  *.sh / *_job.py        per-step torchrun DDP entrypoints
notebooks/
  00_setup.ipynb         auto-detects role/region/bucket, prompts for HF token, writes config
  01_prepare.ipynb       builds + inspects data locally (SFT/GRPO/eval/irrelevance), uploads to S3
  02_sft.ipynb           launch SFT job (DDP)
  03_grpo.ipynb          launch GRPO/RLVR job (DDP)
  04_evaluate.ipynb      launch eval job, download metrics + comparison chart
  05_mmlu.ipynb          optional: MMLU regression check (LightEval) — UNDER CONSTRUCTION (not working yet)
  06_bfcl.ipynb          optional: BFCL external benchmark — UNDER CONSTRUCTION (not working yet)
tests/
  test_tool_reward.py    verifier unit tests (CPU, no model) — 12 tests
```

## Reproduce

```bash
pip install -r requirements-launch.txt      # client side
python tests/test_tool_reward.py            # sanity-check the verifier (CPU)
```

Then run the notebooks in order: `00_setup` → `01_prepare` → `02_sft` → `03_grpo` →
`04_evaluate`. (The optional `05_mmlu` and `06_bfcl` benchmark notebooks are **under
construction — not working yet**; skip them.) Everything (role, region, S3 bucket) is auto-detected;
you enter your Hugging Face token once in `00_setup`. Data prep runs in-notebook (no job);
SFT/GRPO/eval run as SageMaker training jobs.

Hardware: SFT fits a single 24 GB GPU; GRPO is comfortable on one 24–48 GB GPU for the
0.8B. See `_shared.py` for all knobs.

## Metrics glossary

- **format** — did the model emit a parseable tool call?
- **name F1** — right function(s) selected (F1 over the call multiset).
- **param-name** — right argument *keys* (F1), over name-matched calls.
- **param-value** — fraction of gold argument *values* reproduced correctly.
- **mean reward** — weighted blend of the above; the quantity GRPO optimizes.
- **exact match** — order-insensitive exact equality to the gold call(s) (headline accuracy).
- **abstention** — on no-tool-needed prompts, fraction where the model correctly makes **no** call.

## Limitations

- Single seed; in-distribution evaluation (xLAM) plus a synthetic irrelevance slice. Treat
  the abstention recovery as the robust result and the accuracy deltas as "held."
- Small base (0.8B) chosen specifically to expose headroom; on larger/stronger bases SFT
  tends to saturate and GRPO's accuracy gains shrink.
- The irrelevance slice is synthetic (real queries + mismatched tools), not a curated
  benchmark.
- **The external-benchmark notebooks are UNDER CONSTRUCTION and do not work yet** —
  `05_mmlu.ipynb` (MMLU regression check) and `06_bfcl.ipynb` (BFCL). They're scaffolds
  that still need environment-specific tuning; don't rely on their output.

## Takeaway

SFT teaches the model *how to call*; GRPO/RLVR teaches it *when not to*. On single-turn tool
calling, that distinction is where reinforcement learning adds value imitation cannot —
measurable, reproducible, and at no cost to accuracy.
