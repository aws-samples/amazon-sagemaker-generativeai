# NeMo RL container for Amazon SageMaker Training Jobs
#
# Uses the official pre-built NeMo RL image from NGC as the base,
# then adds EFA, AWS OFI NCCL, and SageMaker training toolkit.
#
# Build:
#   docker build -t nemo-rl-sagemaker:latest .
#
# Or use create-image.sh to build + push to ECR.

ARG NEMO_RL_IMAGE=nvcr.io/nvidia/nemo-rl:v0.5.0
FROM ${NEMO_RL_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# EFA and AWS OFI NCCL versions
ENV EFA_INSTALLER_VERSION=1.44.0
ENV AWS_OFI_NCCL_VERSION=1.16.2

# Install system dependencies for EFA build
# udev, iproute2, ethtool are needed by rdma-core (EFA dependency) on Ubuntu 24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    kmod \
    autoconf \
    libtool \
    automake \
    cmake \
    libhwloc-dev \
    pciutils \
    environment-modules \
    tcl \
    openssh-server \
    openssh-client \
    tini \
    udev \
    iproute2 \
    ethtool \
    dmidecode \
    libevent-core-2.1-7t64 \
    libevent-pthreads-2.1-7t64 \
    && rm -rf /var/lib/apt/lists/*

# Install EFA
RUN cd /tmp && \
    curl -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz && \
    tar -xf aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz && \
    cd aws-efa-installer && \
    ./efa_installer.sh -y -g -d --skip-kmod --no-verify --skip-limit-conf && \
    ldconfig && \
    rm -rf /tmp/aws-efa-installer* /var/lib/apt/lists/*

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

# Environment variables for networking and performance
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

# Force system Python for all Ray actors (no uv-managed venvs)
ENV NEMO_RL_PY_EXECUTABLES_SYSTEM=1

# Install all NeMo RL dependencies into the main venv (/opt/nemo_rl_venv/).
#
# The NGC image has pre-built uv venvs under /opt/ray_venvs/ for vllm, automodel,
# and mcore actors. Since NEMO_RL_PY_EXECUTABLES_SYSTEM=1 forces all Ray actors to
# use the main venv, we need everything available there.
#
# Use uv (already in the NGC image) to sync all extras into the main venv.
# This uses the existing uv.lock so all CUDA extensions (flash-attn, transformer-engine,
# mamba-ssm, vllm, etc.) are resolved from the pre-built cache — no compilation needed.
RUN cd /opt/nemo-rl && \
    uv sync --frozen --all-extras --no-dev && \
    uv pip install --python /opt/nemo_rl_venv/bin/python \
    "opencv-python-headless>=4.11.0" \
    "timm<=1.0.22" \
    "nvidia-modelopt[torch]>=0.39.0" \
    "setuptools>=80.10.2"

# Download Prometheus tar.gz (launcher.py expects the archive, not the extracted binary).
# Set prometheus_path env var so the launcher uses this instead of `ray metrics launch-prometheus`.
RUN curl -Lo /opt/prometheus-3.4.2.linux-amd64.tar.gz \
    https://github.com/prometheus/prometheus/releases/download/v3.4.2/prometheus-3.4.2.linux-amd64.tar.gz
ENV prometheus_path=/opt/prometheus-3.4.2.linux-amd64.tar.gz

# Install Ray Dashboard frontend
RUN pip install --no-cache-dir "ray[default]==2.49.2"
# Install SageMaker training toolkit
RUN pip install --no-cache-dir sagemaker-pytorch-training
