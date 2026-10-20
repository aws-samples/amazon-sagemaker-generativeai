"""
The reward: an LLM-as-judge (RLAIF) scorer for TL;DR summarization.

This is the RLAIF counterpart to the tool-calling project's deterministic verifier.
There is no exact-match ground truth for a summary, so the reward is a *judgment*:
a judge model scores a (post, summary) pair on a rubric. The SAME scorer is used by
GRPO training (phases/grpo.py) and by evaluation (phases/evaluate.py), so "good"
means the same thing in both places — EXCEPT that evaluation deliberately uses a
DIFFERENT judge model than training (see the note below).

Rubric — four components, each normalized to [0, 1]:
    coverage       - captures the key points of the post
    faithfulness   - everything stated is supported by the post (no hallucination)
    coherence      - fluent, well-organized, readable
    conciseness    - appropriately short for a TL;DR (no padding)

GRPO sums these with the configured reward weights. Splitting into four functions
lets TRL log each component separately during training.

*** EVAL-INTEGRITY RULE ***
The judge that produces the TRAINING reward and the judge that SCORES the eval must
be different models (ideally different families/sizes). If they are the same, GRPO
can raise the eval simply by learning to please its own judge — reward hacking that
looks like real improvement. We also (a) randomize A/B order in pairwise eval to
cancel position bias and (b) report summary length to expose verbosity bias.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor

LOG = logging.getLogger("summ-job.judge")

SYSTEM_PROMPT = (
    "You are a careful evaluator of TL;DR summaries of online forum posts. "
    "You reward summaries that are faithful to the post and penalize anything "
    "not supported by it."
)

# System prompt the POLICY model (the one being fine-tuned) runs under. Distinct
# from SYSTEM_PROMPT above, which configures the JUDGE.
SYSTEM_PROMPT_POLICY = (
    "You are a helpful assistant that writes concise, faithful TL;DR summaries of "
    "forum posts."
)

# Instruction the policy model is trained/evaluated under (kept here so training,
# eval, and the judge all agree on the task definition).
TASK_INSTRUCTION = (
    "Write a concise TL;DR summary (1-3 sentences) of the following forum post. "
    "Only use information that appears in the post."
)

_RUBRIC_USER = """Rate the SUMMARY of the POST below on four axes, each an integer 1-5:
- coverage: does it capture the main point(s) of the post?
- faithfulness: is everything in it supported by the post (no invented facts)?
- coherence: is it fluent and well organized?
- conciseness: is it appropriately short, with no padding?

POST:
{post}

SUMMARY:
{summary}

Respond with ONLY a JSON object, no prose, exactly:
{{"coverage": <1-5>, "faithfulness": <1-5>, "coherence": <1-5>, "conciseness": <1-5>}}"""

_PAIRWISE_USER = """Below is a POST and two candidate TL;DR summaries, A and B.
Pick the better summary overall, weighing faithfulness (no invented facts) most,
then coverage, coherence, and conciseness.

POST:
{post}

SUMMARY A:
{a}

SUMMARY B:
{b}

Respond with ONLY one token: "A", "B", or "tie"."""

# Combined call used by the GUARDED training reward: rate the candidate on the
# four axes AND judge it against the reference, in ONE request (saves a judge call).
_FULL_USER = """You are given a POST, a CANDIDATE TL;DR summary, and a REFERENCE TL;DR summary.

First rate the CANDIDATE on four axes, each an integer 1-5:
- coverage: does it capture the main point(s) of the post?
- faithfulness: is everything in it supported by the post (no invented facts)?
- coherence: is it fluent and well organized?
- conciseness: is it appropriately short, with no padding?

Then decide whether the CANDIDATE is better than, equal to, or worse than the
REFERENCE overall (weigh faithfulness most, then coverage, coherence, conciseness).

POST:
{post}

CANDIDATE:
{summary}

REFERENCE:
{reference}

Respond with ONLY a JSON object, no prose, exactly:
{{"coverage": <1-5>, "faithfulness": <1-5>, "coherence": <1-5>, "conciseness": <1-5>, "preference": "better"|"equal"|"worse"}}"""

_COMPONENTS = ("coverage", "faithfulness", "coherence", "conciseness")
_PREF_MAP = {"better": 1.0, "equal": 0.5, "worse": 0.0}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
_JSON_OBJ_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _norm_1to5(x) -> float:
    """Map a 1-5 rating to [0,1]; be forgiving about strings/out-of-range."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    v = max(1.0, min(5.0, v))
    return (v - 1.0) / 4.0


def parse_rubric(text: str) -> dict:
    """Extract the four-axis scores from a judge response, normalized to [0,1].

    Missing/garbled axes default to 0.0 so a judge that fails to answer can't
    silently inflate the reward.
    """
    obj = None
    if text:
        m = _JSON_OBJ_RE.search(text)
        if m:
            try:
                obj = json.loads(m.group(0))
            except (json.JSONDecodeError, ValueError):
                obj = None
    if not isinstance(obj, dict):
        return {c: 0.0 for c in _COMPONENTS}
    return {c: _norm_1to5(obj.get(c)) for c in _COMPONENTS}


def parse_pairwise(text: str) -> str:
    """Return 'A', 'B', or 'tie' from a judge verdict (robust to extra text)."""
    if not text:
        return "tie"
    t = text.strip().lower()
    # Look at the first non-space token-ish char first.
    head = t[:5]
    if head.startswith("a"):
        return "A"
    if head.startswith("b"):
        return "B"
    if "tie" in t:
        return "tie"
    if re.search(r"\ba\b", t):
        return "A"
    if re.search(r"\bb\b", t):
        return "B"
    return "tie"


def parse_full(text: str) -> dict:
    """Parse the combined call: 4 rubric axes in [0,1] + 'preference' in [0,1].

    preference: 1.0 = candidate better than reference, 0.5 = equal, 0.0 = worse.
    A judge that fails to answer defaults axes to 0.0 and preference to 0.0
    (worse) — so a non-answer can't inflate the guarded reward.
    """
    out = parse_rubric(text)  # 4 axes, defaults 0.0
    pref = 0.0
    if text:
        m = re.search(r'"?preference"?\s*:\s*"?(better|equal|worse)"?', text, re.I)
        if m:
            pref = _PREF_MAP[m.group(1).lower()]
    out["preference"] = pref
    return out


# ---------------------------------------------------------------------------
# Judge client (bedrock / local / mock)
# ---------------------------------------------------------------------------
class JudgeClient:
    """Wraps a judge model behind a uniform .score()/.compare() API.

    backend:
      - "bedrock": call a Bedrock model via boto3 bedrock-runtime (default).
      - "local":   a locally loaded HF chat model (shares the box; heavier).
      - "mock":    deterministic heuristic — NO model. For CI/tests/offline smoke;
                   never use for real numbers.
    """

    def __init__(self, backend="bedrock", model=None, region=None,
                 max_concurrency=8, max_chars=6000):
        self.backend = backend
        self.model = model
        self.region = region
        self.max_concurrency = max(1, int(max_concurrency))
        self.max_chars = max_chars
        self._lock = threading.Lock()
        self._client = None
        self._local = None  # (tokenizer, model) for backend == "local"

    # --- backend plumbing ---------------------------------------------------
    def _bedrock(self):
        if self._client is None:
            import boto3

            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    def _invoke(self, user: str, max_tokens: int = 64, temperature: float = 0.0) -> str:
        """Return raw judge text for one (system, user) exchange.

        Handles OpenAI gpt-oss (reasoning) models on Bedrock Converse: they emit a
        reasoningContent block before the answer, so we (a) raise the token budget so
        the actual answer isn't truncated by hidden reasoning, (b) ask for low reasoning
        effort (with a fallback if the field is unsupported), (c) skip temperature (some
        reasoning models reject temperature=0), and (d) extract ONLY the text block(s),
        ignoring reasoningContent.
        """
        if self.backend == "mock":
            return ""  # handled upstream; mock never calls _invoke
        if self.backend == "local":
            return self._invoke_local(user, max_tokens)
        if self.backend != "bedrock":
            raise ValueError(f"unknown judge backend: {self.backend}")

        is_openai = str(self.model or "").startswith("openai.")
        mt = max(max_tokens, 512) if is_openai else max_tokens      # reasoning needs headroom
        inf = {"maxTokens": mt} if is_openai else {"maxTokens": mt, "temperature": temperature}
        # For gpt-oss try low reasoning effort first, then fall back without the field.
        attempts = [{"reasoning_effort": "low"}, None] if is_openai else [None]

        last = None
        for extra in attempts:
            try:
                kwargs = dict(
                    modelId=self.model,
                    system=[{"text": SYSTEM_PROMPT}],
                    messages=[{"role": "user", "content": [{"text": user}]}],
                    inferenceConfig=inf,
                )
                if extra:
                    kwargs["additionalModelRequestFields"] = extra
                resp = self._bedrock().converse(**kwargs)
                blocks = resp.get("output", {}).get("message", {}).get("content", [])
                # Join only the text blocks; drop reasoningContent / other block types.
                text = "".join(b.get("text", "") for b in blocks if isinstance(b, dict))
                if text.strip():
                    return text
            except Exception as e:  # noqa: BLE001
                last = e
        if last is not None:
            LOG.warning("bedrock judge call failed (%s); returning empty verdict.", last)
        return ""

    def _invoke_local(self, user: str, max_tokens: int) -> str:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self._local is None:
            tok = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
            mdl = AutoModelForCausalLM.from_pretrained(
                self.model, torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True,
            )
            mdl.eval()
            self._local = (tok, mdl)
        tok, mdl = self._local
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user}]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        enc = tok(prompt, return_tensors="pt").to(mdl.device)
        with torch.no_grad():
            out = mdl.generate(**enc, max_new_tokens=max_tokens, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

    # --- public API ---------------------------------------------------------
    def _clip(self, s: str) -> str:
        s = s or ""
        return s if len(s) <= self.max_chars else s[: self.max_chars]

    def score(self, post: str, summary: str) -> dict:
        """Rubric score for one (post, summary) -> dict of 4 comps in [0,1]."""
        if self.backend == "mock":
            return _mock_rubric(post, summary)
        if not (summary and summary.strip()):
            return {c: 0.0 for c in _COMPONENTS}
        user = _RUBRIC_USER.format(post=self._clip(post), summary=self._clip(summary))
        return parse_rubric(self._invoke(user, max_tokens=64))

    def score_batch(self, posts, summaries) -> list:
        """Threaded rubric scoring for a batch (keeps GRPO rollouts from stalling)."""
        pairs = list(zip(posts, summaries))
        if self.backend == "mock" or self.max_concurrency == 1:
            return [self.score(p, s) for p, s in pairs]
        out = [None] * len(pairs)
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as ex:
            futs = {ex.submit(self.score, p, s): i for i, (p, s) in enumerate(pairs)}
            for fut in futs:
                out[futs[fut]] = fut.result()
        return out

    def compare(self, post: str, a: str, b: str) -> str:
        """Pairwise verdict for A vs B -> 'A' | 'B' | 'tie' (order as given)."""
        if self.backend == "mock":
            return _mock_compare(post, a, b)
        user = _PAIRWISE_USER.format(post=self._clip(post), a=self._clip(a), b=self._clip(b))
        return parse_pairwise(self._invoke(user, max_tokens=4))

    def score_full(self, post: str, summary: str, reference: str) -> dict:
        """Combined rubric + preference-vs-reference for the GUARDED training reward.

        Returns the 4 rubric axes AND 'preference' (candidate vs reference), all in
        [0,1], from a single judge call.
        """
        if self.backend == "mock":
            return _mock_full(post, summary, reference)
        if not (summary and summary.strip()):
            return {c: 0.0 for c in _COMPONENTS} | {"preference": 0.0}
        user = _FULL_USER.format(post=self._clip(post), summary=self._clip(summary),
                                 reference=self._clip(reference))
        return parse_full(self._invoke(user, max_tokens=96))

    def score_full_batch(self, posts, summaries, references) -> list:
        """Threaded combined scoring for a batch of (post, summary, reference)."""
        triples = list(zip(posts, summaries, references))
        if self.backend == "mock" or self.max_concurrency == 1:
            return [self.score_full(p, s, r) for p, s, r in triples]
        out = [None] * len(triples)
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as ex:
            futs = {ex.submit(self.score_full, p, s, r): i
                    for i, (p, s, r) in enumerate(triples)}
            for fut in futs:
                out[futs[fut]] = fut.result()
        return out


# ---------------------------------------------------------------------------
# Mock judge — deterministic heuristic (NO model). Keeps the pipeline runnable
# and the unit tests offline. It rewards summaries that reuse the post's words
# (proxy for faithfulness/coverage) and are short (conciseness).
# ---------------------------------------------------------------------------
def _tokens(s: str):
    return re.findall(r"[a-z0-9]+", (s or "").lower())


def _mock_rubric(post: str, summary: str) -> dict:
    ps, ss = set(_tokens(post)), _tokens(summary)
    if not ss:
        return {c: 0.0 for c in _COMPONENTS}
    overlap = sum(1 for w in ss if w in ps) / len(ss)       # faithfulness proxy
    coverage = min(1.0, len(set(ss) & ps) / max(1, len(ps) ** 0.5 * 2))
    n = len(ss)
    conciseness = 1.0 if 8 <= n <= 60 else max(0.0, 1.0 - abs(n - 34) / 100)
    coherence = 0.6 + 0.4 * overlap
    return {
        "coverage": round(min(1.0, coverage), 4),
        "faithfulness": round(overlap, 4),
        "coherence": round(min(1.0, coherence), 4),
        "conciseness": round(conciseness, 4),
    }


def _mock_compare(post: str, a: str, b: str) -> str:
    wa = sum(_mock_rubric(post, a).values())
    wb = sum(_mock_rubric(post, b).values())
    if abs(wa - wb) < 1e-6:
        return "tie"
    return "A" if wa > wb else "B"


def _mock_full(post: str, summary: str, reference: str) -> dict:
    out = _mock_rubric(post, summary)
    verdict = _mock_compare(post, summary, reference)  # candidate=A, reference=B
    out["preference"] = {"A": 1.0, "tie": 0.5, "B": 0.0}[verdict]
    return out


# ---------------------------------------------------------------------------
# Aggregation + headline metric
# ---------------------------------------------------------------------------
def total_reward(components: dict, weights) -> float:
    return float(
        weights[0] * components["coverage"]
        + weights[1] * components["faithfulness"]
        + weights[2] * components["coherence"]
        + weights[3] * components["conciseness"]
    )


# ---------------------------------------------------------------------------
# TRL GRPO reward functions (RLAIF).
# TRL calls each with (completions=[str], **dataset_columns); our GRPO dataset
# carries a "document" column (the source post). To avoid judging each completion
# four times (once per component func), the first func to see a batch scores it and
# caches the four-axis result; the other three read the cache.
# ---------------------------------------------------------------------------
_DEFAULT_CLIENT = None
_CACHE_LOCK = threading.Lock()
_SCORE_CACHE: dict = {}


def set_default_client(client: JudgeClient) -> None:
    """Install the judge used by the GRPO reward functions (call once at train start)."""
    global _DEFAULT_CLIENT, _SCORE_CACHE
    _DEFAULT_CLIENT = client
    _SCORE_CACHE = {}


def default_client() -> JudgeClient:
    """Lazily build the training judge from env if set_default_client wasn't called."""
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        _DEFAULT_CLIENT = JudgeClient(
            backend=os.environ.get("JUDGE_BACKEND", "bedrock"),
            model=os.environ.get("TRAIN_JUDGE_MODEL"),
            region=os.environ.get("JUDGE_REGION") or None,
            max_concurrency=int(os.environ.get("JUDGE_MAX_CONCURRENCY", "8")),
        )
    return _DEFAULT_CLIENT


def _batch_components(completions, documents) -> list:
    """Score (document, completion) pairs with caching, so a batch is judged once."""
    client = default_client()
    results, to_fetch = [None] * len(completions), []
    for i, (c, d) in enumerate(zip(completions, documents)):
        key = (hash(d), hash(c))
        cached = _SCORE_CACHE.get(key)
        if cached is not None:
            results[i] = cached
        else:
            to_fetch.append((i, key, d, c))
    if to_fetch:
        scored = client.score_batch([d for _, _, d, _ in to_fetch],
                                    [c for _, _, _, c in to_fetch])
        with _CACHE_LOCK:
            for (i, key, _, _), comp in zip(to_fetch, scored):
                results[i] = comp
                _SCORE_CACHE[key] = comp
    return results


def _component_reward(name, completions, documents):
    comps = _batch_components(completions, documents)
    return [c[name] for c in comps]


def reward_coverage(completions, document, **kwargs):
    return _component_reward("coverage", completions, document)


def reward_faithfulness(completions, document, **kwargs):
    return _component_reward("faithfulness", completions, document)


def reward_coherence(completions, document, **kwargs):
    return _component_reward("coherence", completions, document)


def reward_conciseness(completions, document, **kwargs):
    return _component_reward("conciseness", completions, document)


# Order must match the reward weights [coverage, faithfulness, coherence, conciseness].
GRPO_REWARD_FUNCS = [reward_coverage, reward_faithfulness, reward_coherence, reward_conciseness]


# ---------------------------------------------------------------------------
# GUARDED reward (RLAIF, multi-objective).
#
# Optimizes holistic PREFERENCE (candidate vs the reference summary) but with
# FLOORS on faithfulness and coverage, and deliberately NO conciseness term — so
# GRPO can't win the pairwise objective by shortening, dropping content, or
# hallucinating. This is the "improve pairwise without regressing other axes" reward.
#
#   reward = w_pref·preference + w_faith·faithfulness + w_cov·coverage
#   if faithfulness < faith_gate:  reward *= gate_mult   (hard anti-hallucination gate)
#
# Note: NO conciseness in the reward (it only drove harmful shortening); it stays a
# *reported* eval metric so you can confirm it didn't crater either.
# ---------------------------------------------------------------------------
_GUARD = {"w_pref": 0.45, "w_faith": 0.30, "w_cov": 0.25,
          "faith_gate": 0.40, "gate_mult": 0.30}
_FULL_CACHE: dict = {}


def set_guard_params(w_pref=None, w_faith=None, w_cov=None, faith_gate=None, gate_mult=None):
    """Override the guarded-reward blend (called by phases/grpo.py from cfg)."""
    for k, v in dict(w_pref=w_pref, w_faith=w_faith, w_cov=w_cov,
                     faith_gate=faith_gate, gate_mult=gate_mult).items():
        if v is not None:
            _GUARD[k] = float(v)


def guarded_scalar(full: dict) -> float:
    r = (_GUARD["w_pref"] * full.get("preference", 0.0)
         + _GUARD["w_faith"] * full["faithfulness"]
         + _GUARD["w_cov"] * full["coverage"])
    if full["faithfulness"] < _GUARD["faith_gate"]:
        r *= _GUARD["gate_mult"]
    return float(r)


def _batch_full(completions, documents, baselines) -> list:
    """Combined (rubric + preference-vs-baseline) scoring with caching, judged once.

    `baselines` is the summary each candidate is compared against — either the gold
    reference or the SFT model's own output, depending on --compare-against.
    """
    client = default_client()
    results, to_fetch = [None] * len(completions), []
    for i, (c, d, b) in enumerate(zip(completions, documents, baselines)):
        key = (hash(d), hash(c), hash(b))
        cached = _FULL_CACHE.get(key)
        if cached is not None:
            results[i] = cached
        else:
            to_fetch.append((i, key, d, c, b))
    if to_fetch:
        scored = client.score_full_batch([d for _, _, d, _, _ in to_fetch],
                                         [c for _, _, _, c, _ in to_fetch],
                                         [b for _, _, _, _, b in to_fetch])
        with _CACHE_LOCK:
            for (i, key, _, _, _), full in zip(to_fetch, scored):
                results[i] = full
                _FULL_CACHE[key] = full
    return results


def reward_guarded(completions, document, baseline, **kwargs):
    """TRL reward func for --reward-mode guarded. Needs 'document' + 'baseline' columns.

    'baseline' is what each rollout is judged against (SFT output or reference) — aligning
    the training signal with the eval's pairwise-vs-SFT objective when compare-against=sft.
    """
    fulls = _batch_full(completions, document, baseline)
    return [guarded_scalar(f) for f in fulls]


def set_default_client_reset_caches(client: JudgeClient) -> None:
    """set_default_client + clear the guarded cache too (full train-start reset)."""
    global _FULL_CACHE
    set_default_client(client)
    _FULL_CACHE = {}
