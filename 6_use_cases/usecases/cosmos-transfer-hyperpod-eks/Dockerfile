# Cosmos Transfer 2.5 — Training Container for HyperPod EKS
#
# Bakes the entire Cosmos Transfer environment into a container image so
# HyperPod EKS can launch training jobs without runtime setup.
#
# Build:
#   docker build -t cosmos-transfer-training:latest .
#
# Push to ECR:
#   export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
#   export REGION=us-east-1
#   aws ecr create-repository --repository-name cosmos-transfer-training --region $REGION 2>/dev/null || true
#   aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com
#   docker tag cosmos-transfer-training:latest $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/cosmos-transfer-training:latest
#   docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/cosmos-transfer-training:latest

FROM nvcr.io/nvidia/pytorch:25.03-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ── System dependencies ──
RUN apt-get update -qq && \
    apt-get install -y -qq curl ffmpeg libx11-dev libgl1 libglib2.0-0 git-lfs tree wget && \
    rm -rf /var/lib/apt/lists/*

# ── Install uv (NVIDIA's recommended package manager for Cosmos) ──
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# ── Clone Cosmos Transfer 2.5 ──
WORKDIR /workspace
RUN GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/nvidia-cosmos/cosmos-transfer2.5.git
WORKDIR /workspace/cosmos-transfer2.5
RUN git lfs install && git lfs pull

# ── Install Python environment with uv ──
RUN uv python install && uv sync --extra cu128

# ── Patch PyTorch CUDA version check (allows flash-attn/TE to compile) ──
RUN EXTFILE=$(uv run python3 -c "import torch.utils.cpp_extension as ext; import inspect; print(inspect.getfile(ext))" 2>/dev/null) && \
    if [ -n "$EXTFILE" ] && grep -q "raise RuntimeError(CUDA_MISMATCH_MESSAGE" "$EXTFILE" 2>/dev/null; then \
        sed -i 's/raise RuntimeError(CUDA_MISMATCH_MESSAGE, cuda_str_version, torch.version.cuda)/pass/' "$EXTFILE"; \
    fi

# ── Patch augmentor: don't split hdmap_bbox on underscore ──
RUN sed -i 's/for key in control_input_type.split("_")/for key in [control_input_type]/' \
    cosmos_transfer2/_src/transfer2/datasets/augmentors/control_input.py 2>/dev/null || true

# ── Copy entrypoint and helper scripts ──
COPY scripts/ /workspace/scripts/

# ── Set environment defaults ──
ENV IMAGINAIRE_OUTPUT_ROOT=/workspace/output
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV NCCL_DEBUG=WARN
ENV NCCL_TIMEOUT=1800
ENV WANDB_MODE=disabled

WORKDIR /workspace/cosmos-transfer2.5
ENTRYPOINT ["/bin/bash"]
