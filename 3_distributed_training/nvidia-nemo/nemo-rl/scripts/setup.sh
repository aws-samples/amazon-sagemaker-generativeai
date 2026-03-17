#!/bin/bash
# Set PYTHONPATH so Python finds all nemo_rl subpackages (algorithms, data, etc.).
# The pip-installed nemo-rl only includes the top-level module (pyproject.toml
# declares packages=["nemo_rl"]). The NGC container ships the full source tree
# at /opt/nemo-rl which contains all subpackages.

set -e

echo "Verifying nemo_rl.algorithms is importable..."
PYTHONPATH=/opt/nemo-rl:$PYTHONPATH python -c "from nemo_rl.algorithms.grpo import grpo_train; print('nemo_rl.algorithms OK')"

export PYTHONPATH=/opt/nemo-rl:$PYTHONPATH
echo "PYTHONPATH set to: $PYTHONPATH"
