#!/bin/bash
#
# Step 6 entrypoint — BFCL (Berkeley Function Calling Leaderboard), single GPU.
# The EXTERNAL tool-calling benchmark: single-turn AST categories + irrelevance.
#
# SCAFFOLD: bfcl-eval's CLI + category names drift across versions. If a command
# below errors, run `bfcl --help` and adjust flags in eval_bfcl.py::run_bfcl.
#
set -euo pipefail

echo "=================================================================="
echo " STEP 6 — BFCL"
echo "=================================================================="

: "${SFT_MODEL_S3:?SFT_MODEL_S3 not set}"
: "${GRPO_MODEL_S3:?GRPO_MODEL_S3 not set}"
: "${OUTPUT_RESULTS_S3:?OUTPUT_RESULTS_S3 not set}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-0.8B}"
BFCL_CATEGORIES="${BFCL_CATEGORIES:-simple,multiple,parallel,parallel_multiple,irrelevance}"
EVAL_BASE="${EVAL_BASE:-1}"

cd "$(dirname "$0")"
echo "handler/model : ${BASE_MODEL}"
echo "categories    : ${BFCL_CATEGORIES}  (single-turn AST + irrelevance)"
echo "output        : ${OUTPUT_RESULTS_S3}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

# BFCL + vLLM install — DEFENSIVE.
# NEVER use `-U` here: it forces pip to upgrade the whole dependency graph, which
#   (a) explodes the resolver ('resolution-too-deep'), and
#   (b) swaps the container's torch and breaks its compiled extensions (flash-attn).
# We freeze torch to the image's version via a constraints file so pip keeps the graph
# small and leaves torch alone, and install the two packages separately. If vllm needs
# a different torch you'll now get an EXPLICIT conflict (actionable) instead of a hang.
# Pin BFCL_VERSION / VLLM_VERSION to a combo known to load Qwen3.5 on your image.
python -c "import torch; v=torch.__version__.split('+')[0]; open('/tmp/constraints.txt','w').write('torch=='+v+'\n')"
echo "freezing torch to: $(cat /tmp/constraints.txt)"
pip install -q -c /tmp/constraints.txt "bfcl-eval${BFCL_VERSION:+==$BFCL_VERSION}"
pip install -q -c /tmp/constraints.txt "vllm${VLLM_VERSION:+==$VLLM_VERSION}"

python eval_bfcl.py \
  --base-model        "${BASE_MODEL}" \
  --sft-model-s3      "${SFT_MODEL_S3}" \
  --grpo-model-s3     "${GRPO_MODEL_S3}" \
  --categories        "${BFCL_CATEGORIES}" \
  --eval-base         "${EVAL_BASE}" \
  --output-results-s3 "${OUTPUT_RESULTS_S3}"

echo "STEP 6 complete."
