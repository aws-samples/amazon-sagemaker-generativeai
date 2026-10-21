"""
Unit tests for the RLAIF reward (src/judge_reward.py) — the judge-output parsing,
rubric aggregation, the deterministic mock judge, and the GRPO reward-func batch
API. Runs on CPU with NO model and NO Bedrock call:

    python -m pytest tests/ -q      (or)    python tests/test_judge_reward.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import judge_reward as J  # noqa: E402

W = [0.30, 0.40, 0.15, 0.15]


# --- rubric JSON parsing ----------------------------------------------------
def test_parse_rubric_normalizes_1to5():
    c = J.parse_rubric('{"coverage": 5, "faithfulness": 1, "coherence": 3, "conciseness": 5}')
    assert c["coverage"] == 1.0        # 5 -> 1.0
    assert c["faithfulness"] == 0.0    # 1 -> 0.0
    assert abs(c["coherence"] - 0.5) < 1e-9   # 3 -> 0.5
    assert c["conciseness"] == 1.0


def test_parse_rubric_handles_surrounding_text():
    c = J.parse_rubric('Sure! {"coverage":4,"faithfulness":4,"coherence":4,"conciseness":4} done')
    assert all(abs(v - 0.75) < 1e-9 for v in c.values())


def test_parse_rubric_garbage_defaults_zero():
    # A judge that fails to answer must NOT silently inflate the reward.
    c = J.parse_rubric("I cannot do that")
    assert c == {"coverage": 0.0, "faithfulness": 0.0, "coherence": 0.0, "conciseness": 0.0}


def test_parse_rubric_out_of_range_clamped():
    c = J.parse_rubric('{"coverage": 9, "faithfulness": -2, "coherence": 3, "conciseness": 3}')
    assert c["coverage"] == 1.0 and c["faithfulness"] == 0.0


# --- pairwise verdict parsing ----------------------------------------------
def test_parse_pairwise_variants():
    assert J.parse_pairwise("A") == "A"
    assert J.parse_pairwise("B is better") == "B"
    assert J.parse_pairwise("tie") == "tie"
    assert J.parse_pairwise("The answer is A.") == "A"
    assert J.parse_pairwise("") == "tie"


# --- weighted total ---------------------------------------------------------
def test_total_reward_weighting():
    comps = {"coverage": 1.0, "faithfulness": 0.0, "coherence": 1.0, "conciseness": 0.0}
    # 0.30*1 + 0.40*0 + 0.15*1 + 0.15*0 = 0.45
    assert abs(J.total_reward(comps, W) - 0.45) < 1e-9


# --- mock judge is deterministic + sane ------------------------------------
def test_mock_rubric_rewards_faithful_overlap():
    post = "the cat sat on the mat near the warm fire in the old house"
    faithful = "the cat sat on the mat"
    hallucinated = "aliens invaded Mars yesterday"
    cf = J._mock_rubric(post, faithful)
    ch = J._mock_rubric(post, hallucinated)
    assert cf["faithfulness"] > ch["faithfulness"]


def test_mock_empty_summary_scores_zero():
    c = J._mock_rubric("some post text here", "")
    assert all(v == 0.0 for v in c.values())


# --- JudgeClient(mock) end to end ------------------------------------------
def test_mock_client_score_and_compare():
    client = J.JudgeClient(backend="mock")
    post = "the quick brown fox jumps over the lazy dog in the park"
    good = "the quick brown fox jumps over the lazy dog"
    bad = "unrelated nonsense about quantum tunnelling"
    cg = client.score(post, good)
    assert 0.0 <= cg["faithfulness"] <= 1.0
    assert client.compare(post, good, bad) == "A"   # good (A) should win
    assert client.compare(post, bad, good) == "B"   # order flips -> B wins


# --- GRPO reward-func batch API (with an injected fake judge) --------------
class _FakeJudge:
    """Returns a fixed rubric per summary string, and counts calls to prove caching."""
    def __init__(self):
        self.calls = 0

    def score_batch(self, posts, summaries):
        self.calls += len(summaries)
        out = []
        for s in summaries:
            v = 1.0 if "good" in s else 0.0
            out.append({"coverage": v, "faithfulness": v, "coherence": v, "conciseness": v})
        return out


def test_reward_funcs_batch_and_cache():
    fake = _FakeJudge()
    J.set_default_client(fake)
    comps = ["a good summary", "a bad summary", "good again"]
    docs = ["post1", "post2", "post3"]

    cov = J.reward_coverage(comps, docs)
    faith = J.reward_faithfulness(comps, docs)
    assert cov == [1.0, 0.0, 1.0]
    assert faith == [1.0, 0.0, 1.0]
    # Four component funcs over the same batch must judge each item ONCE, not 4x:
    # only the first func should have triggered scoring (3 items), the rest hit cache.
    J.reward_coherence(comps, docs)
    J.reward_conciseness(comps, docs)
    assert fake.calls == 3, f"expected 3 judge calls (cached), got {fake.calls}"


# --- combined parse (rubric + preference) ----------------------------------
def test_parse_full_preference():
    txt = '{"coverage":4,"faithfulness":5,"coherence":4,"conciseness":3,"preference":"better"}'
    f = J.parse_full(txt)
    assert f["faithfulness"] == 1.0 and f["preference"] == 1.0
    assert J.parse_full('{"coverage":3,"faithfulness":3,"coherence":3,"conciseness":3,"preference":"worse"}')["preference"] == 0.0
    assert J.parse_full('{"coverage":3,"faithfulness":3,"coherence":3,"conciseness":3,"preference":"equal"}')["preference"] == 0.5
    # non-answer -> preference 0.0 (can't inflate)
    assert J.parse_full("sorry")["preference"] == 0.0


def test_guarded_scalar_blend_and_gate():
    J.set_guard_params(w_pref=0.45, w_faith=0.30, w_cov=0.25, faith_gate=0.40, gate_mult=0.30)
    # all high -> ~ full blend
    hi = {"preference": 1.0, "faithfulness": 1.0, "coverage": 1.0, "coherence": 1.0, "conciseness": 1.0}
    assert abs(J.guarded_scalar(hi) - 1.0) < 1e-9
    # low faithfulness trips the gate even if preference is high (anti-hallucination)
    hack = {"preference": 1.0, "faithfulness": 0.1, "coverage": 1.0, "coherence": 1.0, "conciseness": 1.0}
    base = 0.45 * 1.0 + 0.30 * 0.1 + 0.25 * 1.0
    assert abs(J.guarded_scalar(hack) - base * 0.30) < 1e-9
    # no conciseness term: conciseness can't change the reward
    a = dict(hi); b = dict(hi); b["conciseness"] = 0.0
    assert J.guarded_scalar(a) == J.guarded_scalar(b)


class _FakeFullJudge:
    def __init__(self):
        self.calls = 0

    def score_full_batch(self, posts, summaries, references):
        self.calls += len(summaries)
        out = []
        for s in summaries:
            # "good" summaries win vs reference and are faithful; "bad" ones hallucinate
            if "good" in s:
                out.append({"coverage": 1.0, "faithfulness": 1.0, "coherence": 1.0,
                            "conciseness": 1.0, "preference": 1.0})
            else:
                out.append({"coverage": 0.3, "faithfulness": 0.1, "coherence": 0.5,
                            "conciseness": 1.0, "preference": 1.0})  # claims "better" but unfaithful
        return out


def test_reward_guarded_rewards_good_gates_bad():
    fake = _FakeFullJudge()
    J.set_default_client_reset_caches(fake)
    comps = ["a good faithful summary", "a bad hallucinated summary"]
    docs = ["post1", "post2"]
    baselines = ["sft_out1", "sft_out2"]   # judged against the SFT baseline, not the reference
    r = J.reward_guarded(comps, docs, baselines)
    assert r[0] > r[1], "faithful 'good' summary must out-reward the unfaithful one"
    assert r[0] > 0.9          # good ~ full blend
    assert r[1] < 0.3          # bad gated down by the faithfulness floor
    # cache: a second call over the same batch triggers no new judge calls
    J.reward_guarded(comps, docs, baselines)
    assert fake.calls == 2


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print(f"\nAll {len(fns)} tests passed.")
