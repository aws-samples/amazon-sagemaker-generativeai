"""
Unit tests for the verifier (src/tool_reward.py) — the RLVR reward + parsing used by
both GRPO training and evaluation. Runs on CPU, no model needed:

    python -m pytest tests/ -q      (or)    python tests/test_tool_reward.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import tool_reward as R  # noqa: E402

W = [0.10, 0.30, 0.20, 0.40]
GOLD = '[{"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}}]'

PERFECT = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}}\n</tool_call>'
WRONG_VAL = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "London", "unit": "celsius"}}\n</tool_call>'
WRONG_NAME = '<tool_call>\n{"name": "get_time", "arguments": {"city": "Paris"}}\n</tool_call>'
MISSING = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
GARBAGE = "The weather in Paris is nice today."


def test_perfect_match_scores_one():
    assert R.is_exact_match(PERFECT, GOLD) is True
    assert abs(R.total_reward(PERFECT, GOLD, W) - 1.0) < 1e-9
    assert R.score_components(PERFECT, GOLD)["param_value"] == 1.0


def test_wrong_value_partial_credit():
    c = R.score_components(WRONG_VAL, GOLD)
    assert c["format"] == 1.0 and c["name"] == 1.0
    assert c["param_value"] == 0.5          # 1 of 2 gold values correct
    assert R.is_exact_match(WRONG_VAL, GOLD) is False


def test_wrong_name_low_reward():
    c = R.score_components(WRONG_NAME, GOLD)
    assert c["name"] == 0.0 and c["param_name"] == 0.0 and c["param_value"] == 0.0
    assert abs(R.total_reward(WRONG_NAME, GOLD, W) - 0.10) < 1e-9  # format only


def test_missing_param_partial():
    c = R.score_components(MISSING, GOLD)
    assert c["name"] == 1.0
    assert 0.0 < c["param_name"] < 1.0
    assert c["param_value"] == 0.5
    assert R.is_exact_match(MISSING, GOLD) is False


def test_garbage_scores_zero_format():
    c = R.score_components(GARBAGE, GOLD)
    assert c["format"] == 0.0 and c["name"] == 0.0
    assert R.total_reward(GARBAGE, GOLD, W) == 0.0


def test_numeric_string_equivalence():
    gold = '[{"name": "add", "arguments": {"a": 2, "b": 3}}]'
    pred = '<tool_call>\n{"name": "add", "arguments": {"a": "2", "b": 3}}\n</tool_call>'
    assert R.is_exact_match(pred, gold) is True


def test_multi_call_order_insensitive():
    gold = ('[{"name": "f", "arguments": {"x": 1}}, '
            '{"name": "g", "arguments": {"y": 2}}]')
    pred = ('<tool_call>\n{"name": "g", "arguments": {"y": 2}}\n</tool_call>\n'
            '<tool_call>\n{"name": "f", "arguments": {"x": 1}}\n</tool_call>')
    assert R.is_exact_match(pred, gold) is True


def test_reward_functions_batch_api():
    comps = [PERFECT, WRONG_VAL, GARBAGE]
    golds = [GOLD, GOLD, GOLD]
    assert R.reward_format(comps, golds) == [1.0, 1.0, 0.0]
    assert R.reward_name(comps, golds) == [1.0, 1.0, 0.0]
    assert R.reward_param_value(comps, golds) == [1.0, 0.5, 0.0]


def test_abstention_correct_full_credit():
    # gold == [] (no tool needed); model correctly makes NO call -> perfect score.
    for text in ("I can't help with that using the available tools.", "", "just prose"):
        c = R.score_components(text, "[]")
        assert c == {"format": 1.0, "name": 1.0, "param_name": 1.0, "param_value": 1.0}
        assert abs(R.total_reward(text, "[]", W) - 1.0) < 1e-9
        assert R.is_exact_match(text, "[]") is True


def test_abstention_overcall_penalized():
    # gold == []; model over-calls -> only the format component, low reward.
    pred = '<tool_call>\n{"name": "foo", "arguments": {"x": 1}}\n</tool_call>'
    c = R.score_components(pred, "[]")
    assert c["format"] == 1.0 and c["name"] == 0.0 and c["param_value"] == 0.0
    assert abs(R.total_reward(pred, "[]", W) - 0.1) < 1e-9
    assert R.is_exact_match(pred, "[]") is False


def test_qwen_xml_format_parsed():
    # The untuned base emits Qwen XML instead of Hermes JSON — must still score.
    xml = ('<tool_call> <function=get_weather> <parameter=city> Paris </parameter> '
           '<parameter=unit> celsius </parameter> </function> </tool_call>')
    calls = R.parse_tool_calls(xml)
    assert calls == [{"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}}]
    assert R.is_exact_match(xml, GOLD) is True
    assert abs(R.total_reward(xml, GOLD, W) - 1.0) < 1e-9


def test_base_example_from_eval():
    # Exact base output that previously scored 0.0 across the board.
    gold = '[{"name": "autocomplete_places", "arguments": {"q": "San", "country": "US,CA", "limit": 5}}]'
    base = ('<tool_call> <function=autocomplete_places> <parameter=q> San </parameter> '
            '<parameter=country> US,CA </parameter> <parameter=type> CITY </parameter> '
            '<parameter=limit> 5 </parameter> </function> </tool_call>')
    c = R.score_components(base, gold)
    assert c["format"] == 1.0           # parsed now, not 0
    assert c["name"] == 1.0             # right function
    assert c["param_value"] == 1.0      # q, country, limit all correct
    assert c["param_name"] < 1.0        # base added an extra `type` param
    assert R.is_exact_match(base, gold) is False  # extra param => not exact


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print(f"\nAll {len(fns)} tests passed.")
