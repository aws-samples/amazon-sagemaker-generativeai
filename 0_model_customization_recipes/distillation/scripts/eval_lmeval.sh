#!/usr/bin/env bash
# Portfolio lm-eval for a merged Qwen3.5-4B on a HyperPod pod: MMLU + non-thinking IFEval -> summary.
# Runs the two correctly (one Qwen3.5 vLLM at a time): MMLU offline (loglikelihood, no server) first,
# reap, then serve non-thinking for IFEval. Usage:
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

echo "=== MMLU (offline vLLM, 5-shot; batch_size=auto DEADLOCKS the hybrid -> fixed 8 + enforce_eager) ==="
"$LMENV/lm_eval" --model vllm \
  --model_args pretrained=$M,tokenizer=$M,dtype=bfloat16,gpu_memory_utilization=0.85,max_model_len=4096,enforce_eager=True,max_num_seqs=8,trust_remote_code=True \
  --tasks mmlu --num_fewshot 5 --batch_size 8 --output_path "$OUT/mmlu" 2>&1 | tail -4
reap

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

echo "=== SUMMARY ($TAG) — base gate: IFEval pp-strict ~82.6 (NOT ~28); MMLU ~0.697 ==="
"$LMENV/python" - "$OUT" << 'PY'
import json, glob, sys
out = sys.argv[1]
def grab(task, metric):
    fs = glob.glob(f"{out}/{task}/**/results*.json", recursive=True)
    if not fs: return None
    r = json.load(open(sorted(fs)[-1]))["results"]
    keys = [k for k in r if task in k] or list(r)
    return r[keys[0]].get(metric)
print("MMLU acc            :", grab("mmlu",  "acc,none"))
print("IFEval prompt-strict:", grab("ifeval","prompt_level_strict_acc,none"))
print("IFEval inst-strict  :", grab("ifeval","inst_level_strict_acc,none"))
print("IFEval prompt-loose :", grab("ifeval","prompt_level_loose_acc,none"))
PY
