"""Write the held-out ("clean") subset of each ShoppingBench test file: problems whose query never appears in a
training corpus. Usage:
    python make_clean_split.py --shopbench <repo> --corpus oro-ai/sn15-shoppingbench-sft-15k --tasks product shop
Writes <repo>/data/synthesize_<task>_test_clean.jsonl and prints the counts."""
import argparse, json, re
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files

ap = argparse.ArgumentParser()
ap.add_argument("--shopbench", required=True)
ap.add_argument("--corpus", required=True, help="Hugging Face dataset repo with the training trajectories")
ap.add_argument("--tasks", nargs="+", default=["product"])
args = ap.parse_args()

norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()        # ignore case / punctuation differences
f = next(x for x in list_repo_files(args.corpus, repo_type="dataset")
         if x.endswith(".jsonl") and "metadata" not in x and "eval" not in x)
train_q = set()
for line in open(hf_hub_download(args.corpus, f, repo_type="dataset")):
    r = json.loads(line)
    train_q.add(norm(next(m["content"] for m in r["messages"] if m["role"] == "user")))
data = Path(args.shopbench) / "data"
for t in args.tasks:
    rows = [json.loads(l) for l in open(data / f"synthesize_{t}_test.jsonl") if l.strip()]
    clean = [r for r in rows if norm(r["query"]) not in train_q]
    with open(data / f"synthesize_{t}_test_clean.jsonl", "w") as out:
        for r in clean: out.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{t}: {len(clean)} / {len(rows)} test problems are absent from {args.corpus}")
