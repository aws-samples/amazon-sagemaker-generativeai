"""Drive the ShoppingBench harness from a notebook: search server, agent rollouts, scoring, health checks.

ShoppingBench (https://github.com/yjwjy/ShoppingBench) pieces used here, all unmodified:
  src/search_engine/server.py   product search over the local Lucene index, on 127.0.0.1:5631 (port fixed in the tools)
  src/agent/run_rollout.py      the agent loop: prompt -> model reply -> parse <tool_call> -> run tool -> repeat
  src/agent/run_evaluate.py     the scoring functions (applied by score_rollouts.py in this folder)
Everything runs in the harness venv (sbenv) created by setup_shopbench.sh.
"""
import json, os, signal, subprocess, time
from pathlib import Path
import requests

SCRIPTS = Path(__file__).resolve().parent
SEARCH_URL = "http://127.0.0.1:5631"


def has_reasoning(text):
    """True if a reply carries actual reasoning. With thinking on, Qwen3.5's template opens <think> in the prompt, so
    the reply is "reasoning ... </think> answer". A fine-tuned model may also emit an empty "<think></think>" with
    thinking off, which does not count."""
    return "</think>" in text and bool(text.split("</think>")[0].replace("<think>", "").strip())


def nlines(p):
    return sum(1 for _ in open(p)) if Path(p).exists() else 0


def problem_queries(problems):
    return {json.loads(l)["query"] for l in open(problems) if l.strip()}


def trajectories(rollout, queries=None):
    """Trajectories in a rollout file, optionally only those for the given queries (a file can hold more, e.g. from
    an earlier run on another problem set)."""
    if not Path(rollout).exists():
        return []
    out = [json.loads(l) for l in open(rollout) if l.strip()]
    return out if queries is None else [t for t in out if t and t[0]["extra_info"]["query"] in queries]


class Harness:
    def __init__(self, work):
        """work: the folder setup_shopbench.sh populated (ShoppingBench/, sbenv/, jdk/)."""
        self.work = Path(work).expanduser().resolve()
        self.sb, self.py = self.work / "ShoppingBench", self.work / "sbenv" / "bin" / "python"
        self.logs = self.work / "logs"; self.logs.mkdir(parents=True, exist_ok=True)
        jdk = self.work / "jdk"
        java_home = jdk / "Contents" / "Home" if (jdk / "Contents" / "Home").exists() else jdk   # macOS vs Linux
        self.env = {**os.environ, "JAVA_HOME": str(java_home), "PATH": f"{java_home}/bin:" + os.environ["PATH"]}
        # pyserini 1.0.0 builds an OpenAI client at import time, and current openai releases raise without a key.
        # The harness passes its own key on every model call, so a placeholder is harmless.
        self.env.setdefault("OPENAI_API_KEY", "unused")
        self.procs = {}

    # ---------- background processes ----------
    def start(self, name, args, env=None):
        self.stop(name)
        log = open(self.logs / f"{name}.log", "w")
        self.procs[name] = subprocess.Popen(args, cwd=self.sb, stdout=log, stderr=subprocess.STDOUT,
                                            env={**self.env, **(env or {})}, start_new_session=True)
        return self.procs[name]

    def running(self, name):
        p = self.procs.get(name)
        return p is not None and p.poll() is None

    def stop(self, name):
        p = self.procs.pop(name, None)
        if p and p.poll() is None:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(60)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)

    def stop_all(self):
        for name in list(self.procs):
            self.stop(name)

    # ---------- search server ----------
    @staticmethod
    def search_ok():
        try:
            return "product_id" in requests.get(f"{SEARCH_URL}/find_product?q=shoes&page=1", timeout=60).text
        except Exception:
            return False

    def start_search(self):
        """Start the search server unless one already answers. Loading the index takes a few minutes. If rollouts
        log many 'Read timed out' errors, the server is overloaded or stuck: call start_search(restart=True)."""
        self.start("search", [str(self.py), "src/search_engine/server.py"], env={"HOST": "127.0.0.1", "PORT": "5631"})   # server.py defaults to 0.0.0.0
        for _ in range(360):
            if self.search_ok():
                return
            if not self.running("search"):
                raise RuntimeError(f"search server exited; see {self.logs}/search.log")
            time.sleep(10)
        raise TimeoutError("search server did not come up")

    # ---------- rollouts ----------
    def write_rollout_cfg(self, out_dir, tag, task, problems, base_url, enable_thinking, threads=4,
                          temperature=0, max_tokens=8192):
        """run_rollout.py's config. Notes:
        - model "" : the endpoint is chosen by URL; vLLM accepts an empty model name.
        - thinking goes in chat_template_kwargs; a top-level {"enable_thinking": ...} (as in the upstream example
          configs) is silently ignored by vLLM.
        - max_tokens is per step (one reply), not per trajectory."""
        out_dir = Path(out_dir)
        rollout = out_dir.resolve() / f"rollout_{tag}_{task}.jsonl"
        cfg = {"task": task, "system_prompt_file": "src/agent/prompt/rollout.md", "synthesize_file": str(Path(problems).resolve()),
               "rollout_file": str(rollout), "threads": threads, "base_url": base_url, "api_key": "unused",
               "model_config": {"model": "", "temperature": temperature, "max_tokens": max_tokens, "stream": False,
                                "extra_body": {"chat_template_kwargs": {"enable_thinking": enable_thinking}}}}
        path = out_dir / f"cfg_{tag}_{task}.json"
        path.write_text(json.dumps(cfg, indent=2))
        return path, rollout

    def run_rollouts(self, jobs, poll_s=120):
        """jobs: {name: (cfg_path, rollout_path, n_problems)}. All jobs run concurrently; blocks until done.
        Resume-safe: run_rollout.py skips problems already in the rollout file, so an interrupted run continues
        where it stopped (interrupting this cell also stops the trajectories). Progress counts only trajectories for
        the job's own problems."""
        queries = {name: problem_queries(json.load(open(cfg))["synthesize_file"]) for name, (cfg, _, _) in jobs.items()}
        progress = lambda: {name: f"{len({t[0]['extra_info']['query'] for t in trajectories(r, queries[name])})}/{n}"
                            for name, (_, r, n) in jobs.items()}
        for name, (cfg, rollout, n) in jobs.items():
            if len(trajectories(rollout, queries[name])) < n:
                self.start(f"rollout_{name}", [str(self.py), "src/agent/run_rollout.py", str(cfg)])
        try:
            while any(self.running(f"rollout_{name}") for name in jobs):
                print(time.strftime("%H:%M"), progress(), flush=True)
                time.sleep(poll_s)
        finally:
            for name in jobs:
                self.stop(f"rollout_{name}")
        print("rollouts done:", progress())

    # ---------- scoring ----------
    def score(self, task, rollout, problems):
        """ShoppingBench's own rules via score_rollouts.py -> {"n", "asr", "exact_id", "recommend"}."""
        r = subprocess.run([str(self.py), str(SCRIPTS / "score_rollouts.py"), "--task", task, "--rollouts", str(Path(rollout).resolve()),
                            "--problems", str(Path(problems).resolve())], cwd=self.sb, env=self.env, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr[-3000:])
        return json.loads(r.stdout.strip().splitlines()[-1])

    @staticmethod
    def health(rollout, problems=None):
        """Run-quality checks (the same ones ShoppingBench's run.sh prints), as shares of trajectories:
        empty_content       model call returned nothing (endpoint errors, context overflow)
        null_tool_results   a tool call failed (usually the search server timing out)
        no_terminate        trajectory ended without the terminate tool: mostly a reply the harness could not parse
        steps_with_reasoning  share of steps with a non-empty <think> block (~1 with thinking on, ~0 off)
        problems: count only trajectories for these problems (default: all in the file)."""
        trajs = trajectories(rollout, problem_queries(problems) if problems else None)
        lines = [json.dumps(t, ensure_ascii=False, separators=(",", ":")) for t in trajs]
        n = max(len(lines), 1)
        steps = [s for t in trajs for s in t]
        return {"empty_content": round(sum('"reasoning_content":"","content":""' in l for l in lines) / n, 3),
                "null_tool_results": round(sum('"results":null' in l for l in lines) / n, 3),
                "no_terminate": round(sum('"name":"terminate"' not in l for l in lines) / n, 3),
                "steps_with_reasoning": round(sum(has_reasoning(s["completion"].get("content") or "") for s in steps)
                                              / max(len(steps), 1), 3)}
