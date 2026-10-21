"""
The verifier: deterministic parsing + reward (RLVR) for single-turn tool calling.

Used by BOTH the GRPO trainer (phases/grpo.py) and the evaluation harness
(phases/evaluate.py), so "correct" means exactly the same thing during training
and during scoring.

Reward decomposition (ToolRL-style), each component in [0, 1]:
    format       - a parseable <tool_call>{...}</tool_call> was emitted
    name         - F1 of predicted vs gold function-name multiset
    param_name   - mean F1 of argument keys over name-matched calls
    param_value  - mean fraction of gold argument values reproduced correctly

GRPO sums these with the configured reward weights. Splitting into four functions
lets TRL log each component separately during training.
"""
from __future__ import annotations

import json
import re
from collections import Counter

# Two tool-call formats seen in the wild, both parsed:
#   Hermes/JSON (what SFT teaches):
#     <tool_call>\n{"name": ..., "arguments": {...}}\n</tool_call>
#   Qwen XML (what the untuned base often emits):
#     <tool_call> <function=NAME> <parameter=KEY> VALUE </parameter> ... </function> </tool_call>
TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
FUNCTION_RE = re.compile(r"<function\s*=\s*([^>\s]+)\s*>(.*?)</function>", re.DOTALL)
PARAMETER_RE = re.compile(r"<parameter\s*=\s*([^>\s]+)\s*>(.*?)</parameter>", re.DOTALL)

SYSTEM_PROMPT = "You are a helpful assistant."


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _loads(x):
    if isinstance(x, (list, dict)):
        return x
    if not isinstance(x, str):
        return None
    try:
        return json.loads(x)
    except (json.JSONDecodeError, ValueError):
        return None


def _coerce_call(obj):
    if not isinstance(obj, dict) or "name" not in obj:
        return None
    args = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(args, str):
        args = _loads(args)
    if not isinstance(args, dict):
        args = {}
    return {"name": str(obj["name"]), "arguments": args}


def _parse_xml_functions(text):
    """Parse Qwen XML calls: <function=NAME><parameter=KEY> VALUE </parameter>...</function>."""
    calls = []
    for fm in FUNCTION_RE.finditer(text):
        name = fm.group(1).strip()
        args = {pm.group(1).strip(): pm.group(2).strip()
                for pm in PARAMETER_RE.finditer(fm.group(2))}
        calls.append({"name": name, "arguments": args})
    return calls


def _parse_block(inner):
    """Parse the contents of one <tool_call>...</tool_call> block (JSON or XML)."""
    inner = inner.strip()
    # Hermes/JSON first.
    obj = _loads(inner)
    if obj is None:
        m = JSON_OBJ_RE.search(inner)
        if m:
            obj = _loads(m.group(0))
    call = _coerce_call(obj) if isinstance(obj, dict) else None
    if call is not None:
        return [call]
    # Qwen XML fallback.
    return _parse_xml_functions(inner)


def parse_tool_calls(text):
    """Extract tool calls from a generation, accepting both Hermes-JSON and Qwen-XML."""
    if not text:
        return []
    out = []
    # 1) Anything inside <tool_call> ... </tool_call> blocks.
    for inner in TOOL_CALL_BLOCK_RE.findall(text):
        out.extend(_parse_block(inner))
    if out:
        return out
    # 2) No wrapper (or nothing parsed): bare <function=...> XML calls.
    out = _parse_xml_functions(text)
    if out:
        return out
    # 3) Last resort: a bare JSON object with name/arguments.
    m = JSON_OBJ_RE.search(text)
    if m:
        call = _coerce_call(_loads(m.group(0)))
        if call is not None:
            out.append(call)
    return out


def normalize_answers(answers):
    data = _loads(answers) if isinstance(answers, str) else answers
    if isinstance(data, dict):
        data = [data]
    out = []
    for a in (data or []):
        call = _coerce_call(a)
        if call is not None:
            out.append(call)
    return out


def coerce_tools(tools):
    """Coerce xLAM tool schemas into OpenAI-style {'type':'function','function':{...}}."""
    data = _loads(tools) if isinstance(tools, str) else tools
    out = []
    for t in (data or []):
        if isinstance(t, dict) and t.get("type") == "function" and "function" in t:
            out.append(t)
        elif isinstance(t, dict):
            out.append({"type": "function", "function": t})
    return out


# ---------------------------------------------------------------------------
# Value comparison
# ---------------------------------------------------------------------------
def _norm_val(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        try:
            return float(s)
        except ValueError:
            return s.lower()
    if isinstance(v, (list, dict)):
        return json.dumps(v, sort_keys=True)
    return v


def _values_equal(a, b):
    na, nb = _norm_val(a), _norm_val(b)
    if isinstance(na, float) and isinstance(nb, float):
        return abs(na - nb) < 1e-6
    return na == nb


def _multiset_f1(pred, gold):
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    cp, cg = Counter(pred), Counter(gold)
    inter = sum((cp & cg).values())
    if inter == 0:
        return 0.0
    prec = inter / sum(cp.values())
    rec = inter / sum(cg.values())
    return 2 * prec * rec / (prec + rec)


def _pair_by_name(pred, gold):
    gold_left = list(gold)
    pairs = []
    for p in pred:
        idx = next((i for i, g in enumerate(gold_left) if g["name"] == p["name"]), None)
        if idx is not None:
            pairs.append((p, gold_left.pop(idx)))
    return pairs


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------
def score_components(pred_text, gold_answers):
    gold = normalize_answers(gold_answers)
    pred = parse_tool_calls(pred_text)

    # Abstention case: gold is empty, i.e. NO tool should be called (irrelevance).
    # Correct = emit no call (full credit); over-calling = emitted a call (format only).
    if not gold:
        if not pred:
            return {"format": 1.0, "name": 1.0, "param_name": 1.0, "param_value": 1.0}
        return {"format": 1.0, "name": 0.0, "param_name": 0.0, "param_value": 0.0}

    fmt = 1.0 if pred else 0.0
    name = _multiset_f1([p["name"] for p in pred], [g["name"] for g in gold])

    pairs = _pair_by_name(pred, gold)
    if not pairs:
        both_empty = (not gold and not pred)
        return {"format": fmt, "name": name,
                "param_name": 1.0 if both_empty else 0.0,
                "param_value": 1.0 if both_empty else 0.0}

    pn_scores, pv_scores = [], []
    for p, g in pairs:
        pk, gk = set(p["arguments"].keys()), set(g["arguments"].keys())
        if not pk and not gk:
            pn_scores.append(1.0)
        elif not pk or not gk:
            pn_scores.append(0.0)
        else:
            inter = len(pk & gk)
            if inter == 0:
                pn_scores.append(0.0)
            else:
                prec, rec = inter / len(pk), inter / len(gk)
                pn_scores.append(2 * prec * rec / (prec + rec))
        if not gk:
            pv_scores.append(1.0 if not pk else 0.0)
        else:
            correct = sum(1 for k in gk
                          if k in p["arguments"] and _values_equal(p["arguments"][k], g["arguments"][k]))
            pv_scores.append(correct / len(gk))

    return {"format": fmt, "name": name,
            "param_name": sum(pn_scores) / len(pn_scores),
            "param_value": sum(pv_scores) / len(pv_scores)}


def total_reward(pred_text, gold_answers, weights):
    c = score_components(pred_text, gold_answers)
    return (weights[0] * c["format"] + weights[1] * c["name"]
            + weights[2] * c["param_name"] + weights[3] * c["param_value"])


# ---------------------------------------------------------------------------
# Exact match (headline eval metric)
# ---------------------------------------------------------------------------
def _args_exact(a, b):
    if set(a.keys()) != set(b.keys()):
        return False
    return all(_values_equal(a[k], b[k]) for k in b)


def is_exact_match(pred_text, gold_answers):
    gold = normalize_answers(gold_answers)
    pred = parse_tool_calls(pred_text)
    if len(pred) != len(gold):
        return False
    gold_left = list(gold)
    for p in pred:
        found = None
        for i, g in enumerate(gold_left):
            if g["name"] == p["name"] and _args_exact(p["arguments"], g["arguments"]):
                found = i
                break
        if found is None:
            return False
        gold_left.pop(found)
    return not gold_left


# ---------------------------------------------------------------------------
# TRL GRPO reward functions.
# TRL calls each with (prompts=..., completions=[str], **dataset_columns); our
# GRPO dataset carries a "gold" column (json string of the answers).
# ---------------------------------------------------------------------------
def reward_format(completions, gold, **kwargs):
    return [score_components(c, g)["format"] for c, g in zip(completions, gold)]


def reward_name(completions, gold, **kwargs):
    return [score_components(c, g)["name"] for c, g in zip(completions, gold)]


def reward_param_name(completions, gold, **kwargs):
    return [score_components(c, g)["param_name"] for c, g in zip(completions, gold)]


def reward_param_value(completions, gold, **kwargs):
    return [score_components(c, g)["param_value"] for c, g in zip(completions, gold)]


GRPO_REWARD_FUNCS = [reward_format, reward_name, reward_param_name, reward_param_value]
