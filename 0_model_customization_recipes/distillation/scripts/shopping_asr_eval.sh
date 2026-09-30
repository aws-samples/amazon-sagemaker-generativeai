#!/usr/bin/env bash
# ShoppingBench tool-use ASR (Agent Success Rate) for a merged Qwen3.5-4B student.
# This is the real rollout+score wiring: serve the model with vLLM, run the ShoppingBench ReAct
# rollout against the search server, then score the trajectories with orm.py (run_evaluate.py).
#
# DEPENDENCIES (not vendored in this repo — this harness IS ShoppingBench):
#   - the ShoppingBench repo checked out at $SB, providing src/agent/run_rollout.py + run_evaluate.py
#     + a rollout config at config/rollout/<TAG>_qwen35-4b.json (points served-model-name + eval set)
#   - the ShoppingBench search server reachable (default :5631) — run_rollout.py calls it for catalog search
#   - a JDK at $JAVA_HOME (orm.py's scorer shells out to it)
#   - the ShoppingBench python env ($PY) with vLLM installed
#   - the merged model dir must contain preprocessor_config.json + processor_config.json (copy from the
#     base snapshot after merge_and_unload — vLLM needs them for this VLM class)
#
# Usage:
#   shopping_asr_eval.sh <MERGED_DIR> <SERVED_NAME> <ROLLOUT_CFG> [GPU] [PORT] [IFACE]
#   e.g. shopping_asr_eval.sh /path/qwen35-4b-shop-r64-merged shop-r64 config/rollout/r64_qwen35-4b.json 0 1078 enp75s0
set -uo pipefail

MERGED_DIR="$1"                       # merged student (or base) model dir
SERVED_NAME="${2:-student}"           # vLLM --served-model-name (must match the rollout config)
CFG="${3:?rollout config path, e.g. config/rollout/r64_qwen35-4b.json}"
GPU="${4:-0}"
PORT="${5:-1078}"
IFACE="${6:-enp75s0}"                 # pod NIC — NO eth0 on HyperPod

SB="${SB:-/data/agankta/shopping/ShoppingBench}"
PY="${PY:-/data/agankta/shopping/shopenv/bin/python}"
export JAVA_HOME="${JAVA_HOME:-/data/agankta/shopping/jdk}"
export PATH="$JAVA_HOME/bin:$PATH"
export GLOO_SOCKET_IFNAME="$IFACE" NCCL_SOCKET_IFNAME="$IFACE"
LOGD="${LOGD:-$SB/logs}"; mkdir -p "$LOGD"
cd "$SB" || { echo "ShoppingBench not found at $SB"; exit 2; }

# 1) Serve the merged model (one Qwen3.5 vLLM per GPU — concurrent instances deadlock on 0.18.1).
CUDA_VISIBLE_DEVICES="$GPU" nohup /usr/local/bin/vllm serve "$MERGED_DIR" \
  --served-model-name "$SERVED_NAME" --port "$PORT" --max-model-len 40960 \
  --gpu-memory-utilization 0.85 --trust-remote-code > "$LOGD/asr_serve_${SERVED_NAME}.log" 2>&1 &
SP=$!
echo "SERVE_PID=$SP model=$SERVED_NAME gpu=$GPU port=$PORT"
for i in $(seq 1 120); do
  curl -s "localhost:$PORT/v1/models" 2>/dev/null | grep -q "$SERVED_NAME" && { echo "SERVER_UP after $((i*15))s"; break; }
  sleep 15
done
if ! curl -s "localhost:$PORT/v1/models" 2>/dev/null | grep -q "$SERVED_NAME"; then
  echo SERVER_FAILED; kill -9 "$SP" 2>/dev/null; exit 1
fi

# 2) ReAct rollout against the search server (writes trajectories per the config).
"$PY" src/agent/run_rollout.py "$CFG" > "$LOGD/asr_rollout_${SERVED_NAME}.log" 2>&1
echo ROLLOUT_DONE

# 3) Score with orm.py -> Agent Success Rate (success rate = rule>=1 over the eval set).
"$PY" src/agent/run_evaluate.py "$CFG" > "$LOGD/score_${SERVED_NAME}.log" 2>&1
echo SCORE_DONE
grep -iE "success|asr|rate" "$LOGD/score_${SERVED_NAME}.log" | tail -5

# 4) Reap the serve (free the GPU for the next instance).
kill -9 "$SP" 2>/dev/null; pkill -9 -f "port $PORT" 2>/dev/null
echo "ASR_DONE_${SERVED_NAME}"
