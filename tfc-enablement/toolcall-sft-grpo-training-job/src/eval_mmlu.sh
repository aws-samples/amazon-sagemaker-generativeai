#!/bin/bash
#
# Step 5 entrypoint — MMLU via LightEval on base / SFT / SFT+GRPO (single GPU).
# Regression check only: is general knowledge preserved after tool-calling fine-tune?
#
set -euo pipefail

echo "=================================================================="
echo " STEP 5 — MMLU (LightEval)"
echo "=================================================================="

: "${SFT_MODEL_S3:?SFT_MODEL_S3 not set}"
: "${GRPO_MODEL_S3:?GRPO_MODEL_S3 not set}"
: "${OUTPUT_RESULTS_S3:?OUTPUT_RESULTS_S3 not set}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-0.8B}"
MMLU_TASKS="${MMLU_TASKS:-leaderboard|mmlu|5|0}"
EVAL_BASE="${EVAL_BASE:-1}"
MMLU_BATCH_SIZE="${MMLU_BATCH_SIZE:-0}"

cd "$(dirname "$0")"
echo "base model : ${BASE_MODEL}"
echo "tasks      : ${MMLU_TASKS}"
echo "output     : ${OUTPUT_RESULTS_S3}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

# LightEval install. If the CLI/task syntax differs on the installed version,
# adjust MMLU_TASKS and the command in eval_mmlu.py::run_lighteval.
pip install -q -U "lighteval[accelerate]" || pip install -q -U lighteval

# WORKAROUND: recent lighteval imports inspect_ai at startup, which does
# `import readline`. Some slim container Pythons ship no stdlib `readline`, so the
# CLI dies with ModuleNotFoundError before any eval runs. readline is only used for
# interactive line-editing (never in batch eval), so a no-op stub on PYTHONPATH is
# safe. PEP-562 module __getattr__ makes ANY attribute access a harmless no-op.
mkdir -p /tmp/pyshim
cat > /tmp/pyshim/readline.py <<'EOF'
def __getattr__(name):
    def _noop(*args, **kwargs):
        return ""
    return _noop
EOF
export PYTHONPATH="/tmp/pyshim:${PYTHONPATH:-}"
python -c "import readline; print('readline shim active:', readline.__file__)"

python eval_mmlu.py \
  --base-model        "${BASE_MODEL}" \
  --sft-model-s3      "${SFT_MODEL_S3}" \
  --grpo-model-s3     "${GRPO_MODEL_S3}" \
  --tasks             "${MMLU_TASKS}" \
  --eval-base         "${EVAL_BASE}" \
  --batch-size        "${MMLU_BATCH_SIZE}" \
  --output-results-s3 "${OUTPUT_RESULTS_S3}"

echo "STEP 5 complete."
