#!/usr/bin/env python3
"""Inspect the teacher trajectory set before you encode it for SeqKD.

Prints the columns, the nested shape of `messages`, the first N rows, and -- the part that
actually matters -- how many unique tasks the file really contains and how many of the
benchmark's held-out queries leak into it.

  # straight from the Hub (default: the train split of the ShoppingBench teacher set)
  python inspect_teacher_dataset.py

  # a local file, more rows, plus the benchmark-contamination check
  python inspect_teacher_dataset.py path/to/oro_sft_15k_v0.jsonl --rows 10 --contamination

Requires `huggingface_hub` only when reading from the Hub.
"""
import argparse
import json
import urllib.request
from collections import Counter

REPO = "oro-ai/sn15-shoppingbench-sft-15k"
# The benchmark's held-out product-test split -- what ASR is scored on.
SB_TEST = ("https://raw.githubusercontent.com/yjwjy/ShoppingBench/main/"
           "data/synthesize_product_test.jsonl")


def typename(v):
    if isinstance(v, list):
        inner = {typename(x) for x in v[:5]}
        return "list[%s]" % ("|".join(sorted(inner)) if inner else "")
    if isinstance(v, dict):
        return "dict"
    return type(v).__name__


def preview(v, width):
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    s = " ".join(s.split())
    return s if len(s) <= width else s[:width] + " ..."


def resolve(path, split):
    """Return a local path: either the one given, or the Hub file for `split`."""
    if path:
        return path
    from huggingface_hub import hf_hub_download, list_repo_files
    jsonls = [f for f in list_repo_files(REPO, repo_type="dataset") if f.endswith(".jsonl")]
    jsonls = [f for f in jsonls if "metadata" not in f.lower()]
    # The set ships a train file and an eval file; pick by whether "eval" is in the name.
    want_eval = split == "eval"
    hits = [f for f in jsonls if ("eval" in f.lower()) == want_eval]
    if not hits:
        raise SystemExit("no %s split among %s" % (split, jsonls))
    print("# downloading %s :: %s" % (REPO, hits[0]))
    return hf_hub_download(REPO, hits[0], repo_type="dataset")


def rows_of(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def first_user(traj):
    """A trajectory's task == its first user message."""
    for m in traj.get("messages", []):
        if m.get("role") == "user":
            return m.get("content")
    return None


def show_columns(path, scan):
    cols, types, n = Counter(), {}, 0
    for row in rows_of(path):
        n += 1
        for k, v in row.items():
            cols[k] += 1
            types.setdefault(k, Counter())[typename(v)] += 1
        if n >= scan:
            break

    first = next(rows_of(path))
    print("=" * 100)
    print("COLUMNS  (scanned %d rows of %s)" % (n, path))
    print("=" * 100)
    print("%-22s %-11s %-14s %s" % ("column", "present", "type(s)", "example"))
    print("-" * 100)
    for k, c in cols.most_common():
        t = "|".join(t for t, _ in types[k].most_common(3))
        print("%-22s %-11s %-14s %s"
              % (k, "%d/%d" % (c, n), t, preview(first.get(k), 48)))

    if "messages" in cols:
        roles, mkeys = Counter(), Counter()
        for m in first.get("messages", []):
            roles[m.get("role")] += 1
            mkeys.update(m.keys())
        print("\nmessages[] roles in row 0 : %s" % dict(roles))
        print("messages[] keys  in row 0 : %s" % dict(mkeys))
    return first


def show_rows(path, limit, width):
    print("\n" + "=" * 100)
    print("FIRST %d ROWS" % limit)
    print("=" * 100)
    for i, row in enumerate(rows_of(path)):
        if i >= limit:
            break
        print("\n--- row %d --- (%s)" % (i, ", ".join(row.keys())))
        for k, v in row.items():
            if k == "messages":
                print("  messages: %d turns" % len(v))
                for j, m in enumerate(v):
                    extra = ""
                    if m.get("tool_calls"):
                        names = [tc.get("function", tc).get("name") for tc in m["tool_calls"]]
                        extra = "  tool_calls=%s" % names
                    print("    [%02d] %-10s %s%s"
                          % (j, m.get("role"), preview(m.get("content") or "", width), extra))
            elif k == "tools" and isinstance(v, list):
                print("  tools: %s" % [t.get("function", t).get("name") for t in v])
            else:
                print("  %s: %s" % (k, preview(v, width)))


def show_task_structure(path):
    """How many DISTINCT tasks are in here? The answer is much smaller than the row count:
    the set is many miners' attempts at the same queries, so near-duplicate trajectories
    sit next to each other. A row-level train/val split therefore leaks."""
    per_query = Counter()
    n = 0
    for row in rows_of(path):
        n += 1
        q = first_user(row)
        if q is not None:
            per_query[q] += 1
    print("\n" + "=" * 100)
    print("TASK STRUCTURE")
    print("=" * 100)
    print("%d trajectories -> %d unique queries (%.1f trajectories per query)"
          % (n, len(per_query), n / max(len(per_query), 1)))
    dup = sum(1 for c in per_query.values() if c > 1)
    print("%d/%d queries appear more than once" % (dup, len(per_query)))
    print("Split train/val BY QUERY, not by row -- a row-level shuffle puts near-duplicate")
    print("attempts at the same task on both sides and makes val loss optimistic.")
    return set(per_query)


def show_contamination(train_queries):
    """Overlap against the benchmark's held-out split. A train/val split inside this file
    does NOT protect against this: it holds back trajectories, while the leak is that the
    benchmark's queries are in the training half at all."""
    raw = urllib.request.urlopen(SB_TEST).read().decode()
    eval_rows = [json.loads(l) for l in raw.splitlines() if l.strip()]
    eq = {r["query"] for r in eval_rows}
    overlap = eq & train_queries
    print("\n" + "=" * 100)
    print("BENCHMARK CONTAMINATION  (vs synthesize_product_test.jsonl)")
    print("=" * 100)
    print("%d benchmark queries; %d appear VERBATIM in this file (%.1f%%)"
          % (len(eq), len(overlap), 100 * len(overlap) / len(eq)))
    print("clean (usable for an honest ASR): %d cases" % (len(eq) - len(overlap)))
    if overlap:
        print("\nTrain on this file unfiltered and ASR measures recall, not skill.")
        print("Drop every trajectory whose first user message is in the benchmark, and score")
        print("only on the %d clean cases." % (len(eq) - len(overlap)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", help="local .jsonl; omit to download from the Hub")
    ap.add_argument("--split", choices=["train", "eval"], default="train",
                    help="which Hub split to fetch when `path` is omitted")
    ap.add_argument("--rows", type=int, default=10, help="how many rows to print")
    ap.add_argument("--scan", type=int, default=2000, help="rows to scan for the schema")
    ap.add_argument("--width", type=int, default=110, help="truncate previews to N chars")
    ap.add_argument("--contamination", action="store_true",
                    help="also check overlap against the benchmark's held-out split (needs network)")
    args = ap.parse_args()

    path = resolve(args.path, args.split)
    show_columns(path, args.scan)
    show_rows(path, args.rows, args.width)
    queries = show_task_structure(path)
    if args.contamination:
        show_contamination(queries)


if __name__ == "__main__":
    main()
