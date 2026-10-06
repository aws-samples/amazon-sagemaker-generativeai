"""IFEval (instruction following) against an OpenAI-compatible URL with lm-evaluation-harness, plus a validity check.

    python run_ifeval.py --base-url http://127.0.0.1:8201/v1 --out out/ifeval_base --lm-eval <venv>/bin/lm_eval

IFEval grades the whole response string, so the model must answer without a reasoning block: point --base-url at a
relay that forces {"chat_template_kwargs": {"enable_thinking": false}}. The check counts predictions that still contain
"</think>"; any leak makes the score invalid. Prints {"prompt_strict", "inst_strict", "reasoning_leaks"}.
"""
import argparse, json, subprocess
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--lm-eval", default="lm_eval", help="lm_eval executable (its own venv: lm_eval[api,ifeval])")
ap.add_argument("--concurrency", type=int, default=8)
ap.add_argument("--max-tokens", type=int, default=1280)
args = ap.parse_args()

out = Path(args.out)
subprocess.run([args.lm_eval, "--model", "local-chat-completions",
                "--model_args", f"base_url={args.base_url}/chat/completions,model=eval,num_concurrent={args.concurrency}",
                "--apply_chat_template", "--tasks", "ifeval", "--gen_kwargs", f"max_tokens={args.max_tokens}",
                "--output_path", str(out), "--log_samples"], check=True)
leaks = sum(open(s).read().count("</think>") for s in out.rglob("samples_ifeval*.jsonl"))
res = json.load(open(sorted(out.rglob("results*.json"))[-1]))["results"]["ifeval"]
print(json.dumps({"prompt_strict": round(res["prompt_level_strict_acc,none"], 3),
                  "inst_strict": round(res["inst_level_strict_acc,none"], 3), "reasoning_leaks": leaks}))
