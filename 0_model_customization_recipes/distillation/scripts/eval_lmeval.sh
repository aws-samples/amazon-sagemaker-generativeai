#!/usr/bin/env bash
# Non-thinking IFEval for a merged Qwen3.5-4B on a GPU box (HyperPod pod layout by default).
#
# Qwen3.5 thinks by default: its chat template opens a <think> block at add_generation_prompt, and strict
# IFEval scores the ENTIRE response string -- no reasoning stripped, no answer extracted -- so the
# reasoning preamble is graded against the prompt's format constraints and fails by construction. Serving
# with nonthink_template.jinja (the model's own template with the line that OPENS <think> replaced by a
# closed-empty <think></think>) is the only reliable fix; enable_thinking=false in model args and a
# /no_think instruction do NOT thread through (lm-eval issue #3161). --apply_chat_template is required or
# the template never takes effect.
#
# Usage:
#   eval_lmeval.sh <MERGED_DIR> <IFACE> <GPU> <TAG>
#   e.g. eval_lmeval.sh /data/agankta/mt/out/lora-r32-16k-merged enp74s0 0 lora-r32-16k
set -uo pipefail
M="$1"; IFACE="$2"; GPU="$3"; TAG="${4:-run}"
export CUDA_VISIBLE_DEVICES="$GPU"
export GLOO_SOCKET_IFNAME="$IFACE" NCCL_SOCKET_IFNAME="$IFACE"   # no eth0 on HyperPod
# Paths default to the HyperPod pod layout but are env-overridable so this runs on any GPU box:
#   LMENV=<lm-eval venv bin>  NONTHINK=<nonthink_template.jinja>  GOOD=<merged dir with the VLM cfgs>
LMENV="${LMENV:-/data/agankta/blog_seqkd/lmevalenv/bin}"
NONTHINK="${NONTHINK:-/data/agankta/blog_seqkd/nonthink_template.jinja}"
GOOD="${GOOD:-/data/agankta/blog_seqkd/out/qwen35-4b-shop-r64-merged}"   # source of preprocessor/processor cfgs
OUT="${OUT:-/data/agankta/blog_seqkd/lmeval/$TAG}"; mkdir -p "$OUT"

# vLLM needs these for this VLM; merged LoRA dirs often miss them -> copy from a known-good merged dir.
for f in preprocessor_config.json processor_config.json; do
  [ -f "$M/$f" ] || cp "$GOOD/$f" "$M/" 2>/dev/null || true
done
reap(){ for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null || true; done; sleep 3; }

echo "=== IFEval (NON-THINKING: nonthink_template forces closed-empty <think></think>) ==="
CUDA_VISIBLE_DEVICES=$GPU "$LMENV/vllm" serve "$M" --port 8000 --served-model-name student \
  --chat-template "$NONTHINK" --enforce-eager --max-num-seqs 8 --max-model-len 16384 \
  > "$OUT/serve.log" 2>&1 &
SP=$!
for i in $(seq 1 90); do curl -sf localhost:8000/v1/models >/dev/null 2>&1 && break; sleep 10; done
"$LMENV/lm_eval" --model local-chat-completions \
  --model_args base_url=http://localhost:8000/v1/chat/completions,model=student,num_concurrent=8 \
  --apply_chat_template --tasks ifeval --output_path "$OUT/ifeval" 2>&1 | tail -4
kill -9 $SP 2>/dev/null || true; reap

# VALIDITY GATE: a real non-thinking run has ZERO <think> in its predictions. If this prints a nonzero
# count, the template did not take effect and the score below is the thinking-mode artifact, not a result.
echo "=== validity gate: <think> occurrences in predictions (expect 0) ==="
grep -ho '<think>' "$OUT"/ifeval/**/samples_*.jsonl 2>/dev/null | wc -l

echo "=== SUMMARY ($TAG) — sanity: base Qwen3.5-4B scores IFEval pp-strict ~0.83 non-thinking, ~0.26 thinking-ON ==="
"$LMENV/python" - "$OUT" << 'PY'
import json, glob, sys
out = sys.argv[1]
def grab(task, metric):
    fs = glob.glob(f"{out}/{task}/**/results*.json", recursive=True)
    if not fs: return None
    r = json.load(open(sorted(fs)[-1]))["results"]
    keys = [k for k in r if task in k] or list(r)
    return r[keys[0]].get(metric)
print("IFEval prompt-strict:", grab("ifeval","prompt_level_strict_acc,none"))
print("IFEval inst-strict  :", grab("ifeval","inst_level_strict_acc,none"))
print("IFEval prompt-loose :", grab("ifeval","prompt_level_loose_acc,none"))
PY
