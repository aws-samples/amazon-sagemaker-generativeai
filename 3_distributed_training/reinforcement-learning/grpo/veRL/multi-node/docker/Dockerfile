# Start from the NVIDIA official image (ubuntu-22.04 + cuda-12.6 + python-3.10)
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-08.html
FROM nvcr.io/nvidia/pytorch:24.08-py3
ENV DEBIAN_FRONTEND=noninteractive

# EFA installer version
ENV EFA_INSTALLER_VERSION=1.44.0
ENV AWS_OFI_NCCL_VERSION=1.16.2

# Remove conflicting packages from base image
RUN apt-get update -y && \
    apt-get remove -y --allow-change-held-packages \
    libmlx5-1 ibverbs-utils libibverbs-dev libibverbs1 && \
    rm -rf /opt/hpcx/ompi /usr/local/mpi /opt/hpcx/nccl_rdma_sharp_plugin && \
    ldconfig

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    git \
    wget \
    curl \
    vim \
    htop \
    openssh-server \
    openssh-client \
    build-essential \
    kmod \
    autoconf \
    libtool \
    automake \
    cmake \
    apt-utils \
    libhwloc-dev \
    systemd \
    tini \
    && rm -rf /var/lib/apt/lists/*

# Change pip source
RUN pip config set global.index-url "${PIP_INDEX}" && \
    pip config set global.extra-index-url "${PIP_INDEX}" && \
    python -m pip install --upgrade pip

# Uninstall nv-pytorch fork
RUN pip uninstall -y torch torchvision torchaudio \
    pytorch-quantization pytorch-triton torch-tensorrt \
    xgboost transformer_engine flash_attn apex megatron-core grpcio

# Install verl
RUN pip install --no-cache-dir verl[vllm] -U

# Install pyairports from GitHub (required by outlines)
RUN git clone https://github.com/NICTA/pyairports.git /tmp/pyairports && \
    cd /tmp/pyairports && \
    pip install . && \
    cd - && \
    rm -rf /tmp/pyairports

# Install torch-2.4.0+cu124 + vllm-0.6.3 (compatible with VERL) and ## IMPORTANT ## Ray for distributing the workload in the cluster
# torch-2.4.0+cu124: cxx11abi=False
# Using older torch version for compatibility with vLLM 0.6.3
RUN pip install --no-cache-dir "vllm==0.6.3" "outlines==0.0.43" "torch==2.4.0" "torchvision==0.19.0" "torchaudio==2.4.0" "tensordict<0.6" torchdata
RUN pip install --no-cache-dir "transformers[hf_xet]==4.47.1" accelerate datasets peft hf-transfer "numpy<2.0.0" "pyarrow>=15.0.0" \
    pandas ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
    pytest py-spy pyext pre-commit ruff huggingface_hub[hf_transfer] trl==0.17.0

# Install flash-attn-2.6.3 (compatible with torch 2.4.0)
RUN wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl && \
    pip install --no-cache-dir flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Install flashinfer-0.1.6 (compatible with vLLM 0.6.3 and torch 2.4.0)
RUN wget -nv https://github.com/flashinfer-ai/flashinfer/releases/download/v0.1.6/flashinfer-0.1.6+cu124torch2.4-cp310-cp310-linux_x86_64.whl && \
    pip install --no-cache-dir flashinfer-0.1.6+cu124torch2.4-cp310-cp310-linux_x86_64.whl

# Fix packages
RUN pip uninstall -y pynvml nvidia-ml-py && \
    pip install --no-cache-dir --upgrade "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"

# Install EFA
RUN apt-get update && \
    apt-get install -y pciutils environment-modules tcl && \
    cd /tmp && \
    curl -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz && \
    tar -xf aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz && \
    cd aws-efa-installer && \
    ./efa_installer.sh -y -g -d --skip-kmod --no-verify --skip-limit-conf && \
    ldconfig && \
    rm -rf /tmp/aws-efa-installer /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/amazon/efa/bin:/opt/amazon/openmpi/bin:$PATH

# Install AWS OFI NCCL plugin
RUN cd /tmp && \
    curl -LO https://github.com/aws/aws-ofi-nccl/archive/refs/tags/v${AWS_OFI_NCCL_VERSION}.tar.gz && \
    tar -xzf v${AWS_OFI_NCCL_VERSION}.tar.gz && \
    rm v${AWS_OFI_NCCL_VERSION}.tar.gz && \
    cd aws-ofi-nccl-${AWS_OFI_NCCL_VERSION} && \
    ./autogen.sh && \
    ./configure --prefix=/opt/amazon/efa \
    --with-libfabric=/opt/amazon/efa \
    --with-cuda=/usr/local/cuda \
    --enable-platform-aws \
    --with-mpi=/opt/amazon/openmpi && \
    make -j$(nproc) install && \
    rm -rf /tmp/aws-ofi-nccl-${AWS_OFI_NCCL_VERSION}

# Configure library paths
RUN echo "/opt/amazon/openmpi/lib" >> /etc/ld.so.conf.d/efa.conf && \
    echo "/usr/local/lib" >> /etc/ld.so.conf.d/local.conf && \
    ldconfig

# Set NCCL configuration
RUN echo "NCCL_SOCKET_IFNAME=^docker0,lo" >> /etc/nccl.conf

# Set environment variables for optimal performance and compatibility
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    LD_LIBRARY_PATH="/usr/local/lib:/usr/local/cuda/lib64:/opt/amazon/efa/lib:${LD_LIBRARY_PATH}" \
    OMPI_MCA_pml=^cm,ucx \
    OMPI_MCA_btl=tcp,self \
    OMPI_MCA_btl_tcp_if_exclude=lo,docker0 \
    OPAL_PREFIX=/opt/amazon/openmpi \
    NCCL_SOCKET_IFNAME=^docker,lo

# Install sagemaker pytorch toolkit
RUN pip install sagemaker-pytorch-training

# Reset pip config
RUN pip config unset global.index-url && \
    pip config unset global.extra-index-url
