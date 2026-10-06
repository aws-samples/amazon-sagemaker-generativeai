"""Score ShoppingBench trajectories with the benchmark's own rules and print one JSON line.

Run from the ShoppingBench repo root with the harness venv (it imports ShoppingBench's scoring code):
    sbenv/bin/python score_rollouts.py --task product --rollouts rollout.jsonl --problems data/synthesize_product_test.jsonl

A problem counts as a success (toward ASR) when the recommended product is the gold product, or it passes every check
in the problem's specification: title similarity >= 0.5 (Qwen3-Embedding-0.6B), each price bound, each required
service, and every required SKU option / attribute (best-matching variant). Shop problems also require all products
from one shop; voucher problems require the total after the voucher to fit the budget. These rules live in
ShoppingBench's src/agent/rewards/orm.py ("outcome reward model") and are applied through run_evaluate.py's functions.

Why not call run_evaluate.py directly: it stops reading after 501 trajectories and only prints a text summary.
Trajectories whose query is not in --problems are skipped, so one rollout file can be scored on a subset.
Output: {"n": problems scored, "asr": share of successes, "exact_id": exact gold-product share, "recommend": share that
ended with a recommendation at all}.
"""
import argparse, contextlib, io, json, sys
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True, choices=["product", "shop", "voucher"])
ap.add_argument("--rollouts", required=True)
ap.add_argument("--problems", required=True)
args = ap.parse_args()

sys.path.insert(0, "src/agent")
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import run_evaluate as RE          # loads the catalog index and the title-similarity embedding model

def solved(task, s):                    # identical to run_evaluate.evaluate()'s "success rate"
    return s["rule"] >= 1 and (task == "product" or (task == "shop" and s["shop"] >= 1)
                               or (task == "voucher" and s["budget"] >= 1))

cfg = {"task": args.task, "synthesize_file": args.problems}
rewards, vouchers = RE.load_synthesize_rewards(cfg), RE.load_synthesize_vouchers(cfg)
res = {}
for line in open(args.rollouts):
    if not line.strip():
        continue
    traj = json.loads(line); q = traj[0]["extra_info"]["query"]
    if q not in rewards:
        continue
    s = defaultdict(float)
    with contextlib.redirect_stdout(io.StringIO()):
        {"product": lambda: RE.eval_product(s, traj, rewards[q]),
         "shop": lambda: RE.eval_shop(s, traj, rewards[q]),
         "voucher": lambda: RE.eval_voucher(s, traj, rewards[q], vouchers.get(q))}[args.task]()
    res[q] = {"solved": solved(args.task, s), "exact_id": s["gt"] >= 1, "recommended": s["product"] > 0}
n = max(len(res), 1)
print(json.dumps({"n": len(res), "asr": round(sum(r["solved"] for r in res.values()) / n, 3),
                  "exact_id": round(sum(r["exact_id"] for r in res.values()) / n, 3),
                  "recommend": round(sum(r["recommended"] for r in res.values()) / n, 3)}))
