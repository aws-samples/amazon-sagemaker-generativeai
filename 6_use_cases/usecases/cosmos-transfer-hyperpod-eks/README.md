# Fine-Tuning NVIDIA Cosmos Transfer 2.5 on Amazon SageMaker HyperPod (EKS)

Fine-tuning Cosmos Transfer 2.5 (2B parameter video-to-video model) on the Drive-Dreams dataset for HD map-conditioned driving video generation. Runs on SageMaker HyperPod with EKS orchestration, supporting single-node and multi-node distributed training.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Folder Structure](#folder-structure)
4. [Prerequisites](#prerequisites)
5. [Step 1: Create HyperPod EKS Cluster](#step-1-create-hyperpod-eks-cluster)
6. [Step 2: Configure kubectl Access](#step-2-configure-kubectl-access)
7. [Step 3: Set Up FSx for Lustre](#step-3-set-up-fsx-for-lustre)
8. [Step 4: Build and Push Docker Image](#step-4-build-and-push-docker-image)
9. [Step 5: Create Kubernetes Secrets](#step-5-create-kubernetes-secrets)
10. [Step 6: Download and Prepare Dataset](#step-6-download-and-prepare-dataset)
11. [Step 7: Install Cosmos Environment on FSx](#step-7-install-cosmos-environment-on-fsx)
12. [Step 8: Launch Training (Single Node)](#step-8-launch-training-single-node)
13. [Step 9: Launch Training (Multi-Node)](#step-9-launch-training-multi-node)
14. [Step 10: Monitor Training](#step-10-monitor-training)
15. [Step 11: Sync Results to S3](#step-11-sync-results-to-s3)
16. [Step 12: Run Inference](#step-12-run-inference)
17. [Scaling GPU Nodes Up/Down](#scaling-gpu-nodes-updown)
18. [Training Results](#training-results)
19. [Code Patches Explained](#code-patches-explained)
20. [Troubleshooting](#troubleshooting)
21. [Cost Estimate](#cost-estimate)
22. [kubectl Cheat Sheet](#kubectl-cheat-sheet)

---

## Overview

**Model**: NVIDIA Cosmos Transfer 2.5 (2B parameters, video-to-video)
**Dataset**: Drive-Dreams — 12,000 HDMap control videos, 84,000 generation videos (7 weather variants), 84,000 captions
**Task**: Fine-tune the segmentation control branch to accept HD map control signals (`hdmap_bbox`) instead of segmentation masks
**Infrastructure**: SageMaker HyperPod EKS with p5.48xlarge (8x H100 80GB) nodes, FSx for Lustre shared storage
**Training Speed**: ~4.5 seconds/iteration on 2x p5.48xlarge (16x H100), ~20 seconds/iteration on 1x p5.48xlarge (8x H100)

### Key Results

| Run | Dataset | Nodes | GPUs | Iterations | Time/iter | Total Time |
|-----|---------|-------|------|------------|-----------|------------|
| 450-clip test | 450 clips | 1x p5 | 8x H100 | 200 | ~20s | ~67 min |
| 10K / 500-iter | 9,800 clips | 2x p5 | 16x H100 | 500 | ~4.5s | ~38 min |
| 10K / 5000-iter | 9,800 clips | 2x p5 | 16x H100 | 5,000 | ~4.5s | ~6.3 hrs |

## Architecture

```
+------------------------------------------------------------------+
|                    HyperPod EKS Cluster                          |
|                                                                  |
|  +------------------------------------------------------------+  |
|  |  Training Nodes (p5.48xlarge x 1-2)                        |  |
|  |  +----------+  +----------+                                |  |
|  |  | Master   |  | Worker 0 |    (optional for multi-node)   |  |
|  |  | 8x H100  |  | 8x H100  |                               |  |
|  |  +----+-----+  +----+-----+                                |  |
|  |       |   EFA (400 Gbps)  |                                |  |
|  |       +--------+----------+                                |  |
|  |                | FSDP + Context Parallelism                |  |
|  +----------------+-------------------------------------------+  |
|                   |                                              |
|  +----------------+-------------------------------------------+  |
|  |  FSx for Lustre (1.2 TB, shared across all nodes)          |  |
|  |  /fsx/datasets/          <- training + test data            |  |
|  |  /fsx/checkpoints/       <- model weights                   |  |
|  |  /fsx/output/            <- training checkpoints + logs     |  |
|  |  /fsx/cosmos-transfer2.5/ <- code + venv (runtime env)     |  |
|  +------------------------------------------------------------+  |
|                                                                  |
|  +------------------------------------------------------------+  |
|  |  System Node (c5.4xlarge x 1) - always on                  |  |
|  |  kubectl, data prep, Docker builds (via kaniko)             |  |
|  +------------------------------------------------------------+  |
+------------------------------------------------------------------+
```

## Folder Structure

```
Cosmos_Transfer_HyperPod_EKS/
|-- Dockerfile                          # Original NGC-based image (has version conflicts)
|-- Dockerfile.v2                       # Clean ubuntu:22.04 + uv sync image (recommended)
|-- README.md                           # This file
|-- scripts/
|   |-- entrypoint_train.sh            # Training entrypoint (FSx venv approach)
|   |-- entrypoint_train_v2.sh         # Training entrypoint (Dockerfile.v2 approach)
|   |-- entrypoint_eval.sh             # Async evaluation entrypoint
|   |-- download_drive_dreams.py       # Download raw data from HuggingFace
|   |-- prepare_drive_dreams_dataset.py # Convert raw data to Cosmos format
|   |-- prepare_multiview_dataset.py   # Multi-camera data prep
|   +-- generate_multiview_specs_from_singleview.py  # Stage 2 multiview
+-- hyperpod/
    |-- cluster-config.yaml            # EKS cluster reference config
    |-- fsx-storage.yaml               # FSx PV/PVC definitions
    |-- training-job.yaml              # Single-node training (1x p5)
    |-- training-job-2node.yaml        # Multi-node training (2x p5)
    |-- kaniko-build-job.yaml          # Build Docker image without Docker daemon
    |-- data-prep-job.yaml             # Download + prepare 500 clips
    |-- data-prepare-10k-job.yaml      # Prepare 10K clips
    |-- singleview-inference-job.yaml  # Run inference with fine-tuned model
    |-- multiview-inference-job.yaml   # Extend to 7 camera views
    |-- eval-job.yaml                  # Async evaluation watcher
    |-- scale-down.json                # Scale GPU nodes to 0
    +-- us-west-2/
        |-- scale-down.json            # Region-specific scale down
        |-- scale-up-1p5.json          # Scale up 1x p5
        +-- scale-up-2p5-reserved.json # Scale up 2x p5 with training plan
```


## Prerequisites

Before starting, you need:

- **AWS CLI** configured with permissions for SageMaker, EKS, ECR, FSx, and S3
- **kubectl** installed on your local machine
- **HuggingFace account** with access to:
  - [nvidia/Cosmos-Transfer2.5-2B](https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B) (gated model, must accept license)
  - [nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams)
- **Weights & Biases account** (optional, for training monitoring)
- **GPU quota** in your AWS region for p5.48xlarge or p4de.24xlarge instances
- **Training Plan** (optional) — reserved capacity for p5 instances reduces cost and guarantees availability

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Training | 1x p5.48xlarge (8x H100 80GB) | 2x p5.48xlarge (16x H100 80GB) |
| Inference | 1x p4de.24xlarge (8x A100 80GB) | Same |
| System | 1x c5.4xlarge | Same |
| Storage (FSx) | 1.2 TB | 2.4 TB |
| VRAM per GPU | 80 GB | 80 GB |

**Important**: g6e.48xlarge (8x L40S 46GB) is NOT sufficient — Cosmos Transfer 2.5 requires ~70GB VRAM for both training and inference.

---

## Step 1: Create HyperPod EKS Cluster

### Via AWS Console (recommended)

1. Go to **SageMaker > HyperPod > Clusters > Create cluster**
2. Choose **EKS** as the orchestrator
3. Configure instance groups:

| Group Name | Instance Type | Count | Purpose |
|------------|--------------|-------|---------|
| system | ml.c5.4xlarge | 1 | Always on, cluster services |
| training-p5 | ml.p5.48xlarge | 0 (scales up) | Training |
| eval-p4de | ml.p4de.24xlarge | 0 (scales up) | Inference/evaluation |

4. Configure networking:
   - VPC with EFA-enabled subnets
   - Security group allowing ports: 988 (Lustre), all TCP between nodes (NCCL)
   - All instances in the same Availability Zone (required for EFA)

5. Create the cluster and wait for it to reach "InService" status (~15-20 minutes)

### Via AWS CLI

```bash
# Reference config is in hyperpod/cluster-config.yaml
# Actual creation uses the SageMaker API:
aws sagemaker create-cluster \
    --cluster-name cosmos-training-cluster \
    --instance-groups file://hyperpod/cluster-config-create.json \
    --region us-west-2
```

---

## Step 2: Configure kubectl Access

After the cluster is created, configure kubectl to connect to the underlying EKS cluster:

```bash
# Find the EKS cluster name (shown in HyperPod console)
# Format: sagemaker-<cluster-name>-<hash>-eks

# Update kubeconfig
aws eks update-kubeconfig \
    --name sagemaker-cosmos-training-cluster-XXXXXXXX-eks \
    --region us-west-2

# Verify connection
kubectl get nodes
```

You should see the system node (c5.4xlarge) listed as Ready. GPU nodes won't appear until scaled up.

```bash
# Expected output:
# NAME                              STATUS   ROLES    AGE   VERSION
# hyperpod-i-XXXXXXXXXXXXXXXXX      Ready    <none>   5m    v1.33.5-eks-ecaa3a6
```

---

## Step 3: Set Up FSx for Lustre

### Create FSx Filesystem

1. Go to **FSx > Create filesystem > Amazon FSx for Lustre**
2. Configuration:
   - Deployment type: PERSISTENT_2
   - Storage capacity: 1200 GB minimum (2400 GB recommended)
   - Throughput: 250 MB/s per TiB
   - Subnet: Same as EKS nodes (same AZ required)
   - Security group: Allow port 988 from EKS node security group

### Create Kubernetes PV/PVC

Edit `hyperpod/fsx-storage.yaml` with your FSx filesystem ID, DNS name, and subnet, then apply:

```bash
# Update the YAML with your values first
kubectl apply -f hyperpod/fsx-storage.yaml

# Verify
kubectl get pvc
# Should show fsx-claim or fsx-cosmos-pvc as Bound
```

---

## Step 4: Build and Push Docker Image

There are two Docker images:

| Image | Base | Use Case |
|-------|------|----------|
| `Dockerfile` (v1) | NGC PyTorch 25.03 | Has PyTorch version conflicts, used with FSx venv workaround |
| `Dockerfile.v2` (recommended) | nvidia/cuda:12.8.0-devel-ubuntu22.04 | Clean build, all patches baked in, self-contained |

### Option A: Build with Kaniko on HyperPod (no Docker needed)

This builds the image inside a Kubernetes pod on the system node:

```bash
# 1. Copy Dockerfile.v2 and scripts to FSx
kubectl run cosmos-prep --rm -it \
    --image=ubuntu:22.04 \
    --overrides='{"spec":{"containers":[{"name":"prep","image":"ubuntu:22.04","command":["bash"],"stdin":true,"tty":true,"volumeMounts":[{"name":"fsx","mountPath":"/fsx"}]}],"volumes":[{"name":"fsx","persistentVolumeClaim":{"claimName":"fsx-claim"}}]}}' \
    -- bash

# Inside the pod:
mkdir -p /fsx/docker-build/scripts
# Copy Dockerfile.v2 content to /fsx/docker-build/Dockerfile.v2
# Copy scripts/apply_patches.py to /fsx/docker-build/scripts/apply_patches.py
exit

# 2. Create ECR repository
aws ecr create-repository --repository-name cosmos-transfer-training --region us-west-2

# 3. Create ECR auth secret for kaniko
TOKEN=$(aws ecr get-login-password --region us-west-2)
kubectl create secret docker-registry ecr-creds \
    --docker-server=<ACCOUNT_ID>.dkr.ecr.us-west-2.amazonaws.com \
    --docker-username=AWS \
    --docker-password=$TOKEN

# 4. Run kaniko build (~25 minutes)
kubectl apply -f hyperpod/kaniko-build-job.yaml
kubectl logs -f cosmos-kaniko-build
```

### Option B: Build on EC2 Instance

If you have an EC2 instance with Docker and 200GB+ disk:

```bash
cd Cosmos_Transfer_HyperPod_EKS
docker build -f Dockerfile.v2 -t cosmos-transfer-training:v2 .

# Push to ECR
aws ecr get-login-password --region us-west-2 | \
    docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.us-west-2.amazonaws.com
docker tag cosmos-transfer-training:v2 \
    <ACCOUNT_ID>.dkr.ecr.us-west-2.amazonaws.com/cosmos-transfer-training:v2
docker push <ACCOUNT_ID>.dkr.ecr.us-west-2.amazonaws.com/cosmos-transfer-training:v2
```

**Note**: SageMaker Studio instances have only 37GB overlay for Docker — not enough for this build.


---

## Step 5: Create Kubernetes Secrets

```bash
# HuggingFace token (required — for downloading model checkpoints)
kubectl create secret generic hf-token --from-literal=token=hf_XXXXXXXXXXXXX

# Weights & Biases token (optional — for training monitoring)
kubectl create secret generic wandb-token --from-literal=token=YOUR_WANDB_API_KEY
```

---

## Step 6: Download and Prepare Dataset

### Download Raw Data

Create a ConfigMap with the data prep scripts, then run the download job:

```bash
# Create ConfigMap with scripts
kubectl create configmap data-prep-scripts \
    --from-file=download_drive_dreams.py=scripts/download_drive_dreams.py \
    --from-file=prepare_drive_dreams_dataset.py=scripts/prepare_drive_dreams_dataset.py

# Run data download + preparation (500 clips for quick test)
kubectl apply -f hyperpod/data-prep-job.yaml
kubectl logs -f cosmos-data-prep
```

This downloads from HuggingFace and writes to FSx:
- HDMap control videos (~2 GB)
- Generation videos (weather variants, ~120 GB for parts 0-2)
- Captions

### Prepare 10K Dataset (for full training)

After the raw data is downloaded:

```bash
kubectl apply -f hyperpod/data-prepare-10k-job.yaml
kubectl logs -f cosmos-data-prepare-10k
```

### Flatten Dataset Structure (CRITICAL)

The prepared dataset has a `front_wide/` subdirectory that must be flattened. Without this, the dataloader hangs silently:

```bash
kubectl run cosmos-flatten --rm -it \
    --image=ubuntu:22.04 \
    --overrides='{"spec":{"containers":[{"name":"flatten","image":"ubuntu:22.04","command":["bash"],"stdin":true,"tty":true,"volumeMounts":[{"name":"fsx","mountPath":"/fsx"}]}],"volumes":[{"name":"fsx","persistentVolumeClaim":{"claimName":"fsx-claim"}}]}}' \
    -- bash
```

Inside the pod:

```bash
# Flatten each dataset
for ds in /fsx/datasets/drive_dreams_cosmos_dataset \
          /fsx/datasets/drive_dreams_cosmos_dataset_test \
          /fsx/datasets/drive_dreams_10k \
          /fsx/datasets/drive_dreams_10k_test; do
    if [ -d "$ds/videos/front_wide" ]; then
        echo "Flattening $ds ..."
        mv "$ds/videos/front_wide/"* "$ds/videos/"
        rmdir "$ds/videos/front_wide"
        mv "$ds/control_input_hdmap_bbox/front_wide/"* "$ds/control_input_hdmap_bbox/"
        rmdir "$ds/control_input_hdmap_bbox/front_wide"
        mv "$ds/captions/front_wide/"* "$ds/captions/"
        rmdir "$ds/captions/front_wide"
        echo "Done: $ds"
    else
        echo "Already flat: $ds"
    fi
done

# Verify
ls /fsx/datasets/drive_dreams_10k/videos/ | head -3
ls /fsx/datasets/drive_dreams_10k/videos/ | wc -l
exit
```

Expected dataset structure after flattening:

```
/fsx/datasets/drive_dreams_10k/
|-- videos/
|   |-- clip_0000.mp4
|   |-- clip_0001.mp4
|   +-- ... (9,800 files)
|-- control_input_hdmap_bbox/
|   |-- clip_0000.mp4
|   +-- ... (9,800 files)
|-- captions/
|   |-- clip_0000.json
|   +-- ... (9,800 files)
+-- dataset_info.json
```

---

## Step 7: Install Cosmos Environment on FSx

The Docker image (v1) based on NGC has a PyTorch version conflict (NGC ships PyTorch 2.9, but flash-attn and transformer_engine from `uv sync` are compiled for PyTorch 2.7). The workaround is to install a clean Python environment directly on FSx and run it from there.

**If using Dockerfile.v2**: Skip this step — the v2 image has everything baked in.

**If using the original Dockerfile (v1)**: You need to set up the environment on FSx once:

```bash
# Scale up a GPU node first (needed for CUDA compilation)
aws sagemaker update-cluster \
    --cluster-name cosmos-training-cluster \
    --instance-groups file://hyperpod/us-west-2/scale-up-2p5-reserved.json \
    --region us-west-2

# Wait for nodes to be Ready
kubectl get nodes -w

# Launch a setup pod on a GPU node
kubectl run cosmos-setup --rm -it \
    --image=ubuntu:22.04 \
    --overrides='{"spec":{"nodeSelector":{"sagemaker.amazonaws.com/instance-group-name":"training-p5"},"tolerations":[{"key":"nvidia.com/gpu","operator":"Exists","effect":"NoSchedule"}],"containers":[{"name":"setup","image":"ubuntu:22.04","command":["bash"],"stdin":true,"tty":true,"resources":{"limits":{"nvidia.com/gpu":"1"}},"volumeMounts":[{"name":"fsx","mountPath":"/fsx"}]}],"volumes":[{"name":"fsx","persistentVolumeClaim":{"claimName":"fsx-claim"}}]}}' \
    -- bash
```

Inside the setup pod:

```bash
# Install system dependencies
apt-get update -qq && apt-get install -y -qq curl git git-lfs ffmpeg libx11-dev libgl1-mesa-glx libglib2.0-0 build-essential

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Clone Cosmos Transfer 2.5
cd /fsx
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/nvidia-cosmos/cosmos-transfer2.5.git
cd cosmos-transfer2.5
git lfs install && git lfs pull

# Install Python environment (this takes ~20 minutes)
uv python install 3.10
uv sync --extra cu128

# Set LD_LIBRARY_PATH and verify
NVIDIA_LIB=$(find .venv -path "*/nvidia/cuda_nvrtc/lib" -type d | head -1)
export LD_LIBRARY_PATH="${NVIDIA_LIB}:${LD_LIBRARY_PATH}"

source .venv/bin/activate
python3 -c "
import torch; print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')
import transformer_engine as te; print(f'TE: {te.__version__}')
import flash_attn; print('flash_attn: OK')
import cosmos_transfer2; print('cosmos_transfer2: OK')
"
```

### Apply Code Patches

Still inside the setup pod, apply all 6 patches:

```bash
cd /fsx/cosmos-transfer2.5

# Patch 1: Add hdmap_bbox to CTRL_TYPE_INFO
sed -i '/"vis": {"folder": None},/a\    "hdmap_bbox": {"folder": "control_input_hdmap_bbox", "format": "mp4", "data_dict_key": "hdmap_bbox"},' \
    cosmos_transfer2/_src/transfer2/datasets/local_datasets/singleview_dataset.py

# Patch 2: Fix augmentor — don't split hdmap_bbox on underscore
sed -i 's/for key in control_input_type.split("_")/for key in [control_input_type]/' \
    cosmos_transfer2/_src/transfer2/datasets/augmentors/control_input.py

# Patch 3: Register hdmap_bbox dataloader (copy vis block and modify)
FILE="cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/defaults/dataloader_local.py"
sed -n '111,134p' "$FILE" > /tmp/vis_block.txt
sed -i 's/Vis (Blur)/HDMap BBox (Drive-Dreams)/' /tmp/vis_block.txt
sed -i 's/dataset_vis/dataset_hdmap_bbox/g' /tmp/vis_block.txt
sed -i 's/control_input_vis/control_input_hdmap_bbox/g' /tmp/vis_block.txt
sed -i 's/example_singleview_train_data_vis/example_singleview_train_data_hdmap_bbox/g' /tmp/vis_block.txt
sed -i 's/PLACEHOLDER_UPDATE_DATASET_PATH/assets\/drive_dreams_cosmos_dataset/' /tmp/vis_block.txt
cat /tmp/vis_block.txt >> "$FILE"

# Patch 4: Create experiment config
mkdir -p cosmos_transfer2/experiments/singleview
touch cosmos_transfer2/experiments/singleview/__init__.py
# (Write the experiment config — see cosmos_drive_dreams_hdmap.py in the repo)

# Patch 5: Create checkpoint symlink
mkdir -p checkpoints
ln -snf /fsx/checkpoints/Cosmos-Transfer2.5-2B checkpoints/Cosmos-Transfer2.5-2B

# Patch 6: Create .netrc for wandb
touch /root/.netrc

exit
```

### Download Model Checkpoints

```bash
kubectl run cosmos-ckpt --rm -it \
    --image=ubuntu:22.04 \
    --overrides='{"spec":{"containers":[{"name":"ckpt","image":"ubuntu:22.04","command":["bash"],"stdin":true,"tty":true,"volumeMounts":[{"name":"fsx","mountPath":"/fsx"}]}],"volumes":[{"name":"fsx","persistentVolumeClaim":{"claimName":"fsx-claim"}}]}}' \
    -- bash
```

Inside:

```bash
cd /fsx/cosmos-transfer2.5
source .venv/bin/activate
NVIDIA_LIB=$(find .venv -path "*/nvidia/cuda_nvrtc/lib" -type d | head -1)
export LD_LIBRARY_PATH="${NVIDIA_LIB}:${LD_LIBRARY_PATH}"

# Login to HuggingFace
python3 -c "from huggingface_hub import login; login(token='YOUR_HF_TOKEN')"

# Download checkpoints (~10 GB)
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('nvidia/Cosmos-Transfer2.5-2B', local_dir='/fsx/checkpoints/Cosmos-Transfer2.5-2B')
"

# Create symlink
mkdir -p /fsx/cosmos-transfer2.5/checkpoints
ln -snf /fsx/checkpoints/Cosmos-Transfer2.5-2B /fsx/cosmos-transfer2.5/checkpoints/Cosmos-Transfer2.5-2B

exit
```


---

## Step 8: Launch Training (Single Node)

Single-node training uses 1x p5.48xlarge (8x H100). This is the simplest setup and matches the original EC2 POC configuration.

### Scale Up GPU Nodes

```bash
aws sagemaker update-cluster \
    --cluster-name cosmos-training-cluster \
    --instance-groups file://hyperpod/us-west-2/scale-up-1p5.json \
    --region us-west-2

# Wait for nodes to appear (2-5 minutes)
kubectl get nodes -w
# Look for a node with label training-p5 and 8 GPUs
```

### Submit Training Job

```bash
# Edit training-job.yaml to set your:
#   - ECR image URI
#   - DATASET_PATH (e.g., /fsx/datasets/drive_dreams_cosmos_dataset)
#   - OUTPUT_PATH (e.g., /fsx/output/cosmos_transfer)

kubectl apply -f hyperpod/training-job.yaml

# Wait ~30 seconds for pod to start
kubectl get pods -l app=cosmos-training

# Follow logs
kubectl logs -f cosmos-transfer-train-master-0
```

### What to Expect

1. "Starting training..." — torchrun launches 8 processes
2. HuggingFace checkpoint download (if not already on FSx)
3. NCCL initialization
4. "Distributed parallelism mode: fsdp" (8 times, one per GPU)
5. Wandb login (if WANDB_API_KEY is set)
6. "Iteration 1: Hit counter: 1/300 | Loss: 0.0762 | Time: 139.43s" — first iteration is slow (warmup)
7. "Iteration 2: ... | Time: 19.73s" — steady state ~20s/iter on single node

---

## Step 9: Launch Training (Multi-Node)

Multi-node training uses 2x p5.48xlarge (16x H100) with FSDP + context parallelism. This is ~4x faster than single-node.

### Update Experiment Config for Multi-Node

The experiment config must have `context_parallel_size=8` (per-node, not total GPUs):

```bash
kubectl run cosmos-patch --rm -it \
    --image=ubuntu:22.04 \
    --overrides='{"spec":{"containers":[{"name":"patch","image":"ubuntu:22.04","command":["bash"],"stdin":true,"tty":true,"volumeMounts":[{"name":"fsx","mountPath":"/fsx"}]}],"volumes":[{"name":"fsx","persistentVolumeClaim":{"claimName":"fsx-claim"}}]}}' \
    -- bash
```

Inside:

```bash
cd /fsx/cosmos-transfer2.5
# Verify context_parallel_size
grep "context_parallel" cosmos_transfer2/experiments/singleview/cosmos_drive_dreams_hdmap.py

# If it shows context_parallel_size=1, update to 8:
sed -i 's/context_parallel_size=1/context_parallel_size=8/' \
    cosmos_transfer2/experiments/singleview/cosmos_drive_dreams_hdmap.py

# Update max_iter and checkpoint frequency as needed:
# sed -i 's/max_iter=200/max_iter=5000/' cosmos_transfer2/experiments/singleview/cosmos_drive_dreams_hdmap.py
# sed -i 's/cycle_lengths=\[200\]/cycle_lengths=[5000]/' cosmos_transfer2/experiments/singleview/cosmos_drive_dreams_hdmap.py
# sed -i 's/save_iter=50/save_iter=250/' cosmos_transfer2/experiments/singleview/cosmos_drive_dreams_hdmap.py

exit
```

### Scale Up 2 GPU Nodes

```bash
aws sagemaker update-cluster \
    --cluster-name cosmos-training-cluster \
    --instance-groups file://hyperpod/us-west-2/scale-up-2p5-reserved.json \
    --region us-west-2

# Wait for both nodes
kubectl get nodes -l sagemaker.amazonaws.com/instance-group-name=training-p5
# Should show 2 nodes with 8 GPUs each
```

### Submit Multi-Node Training Job

```bash
# Delete any existing job first
kubectl delete pytorchjob cosmos-transfer-train 2>/dev/null

# Submit 2-node job
kubectl apply -f hyperpod/training-job-2node.yaml

# Wait for both pods
kubectl get pods -l app=cosmos-training
# Should show master-0 and worker-0 both Running

# Follow master logs
kubectl logs -f cosmos-transfer-train-master-0
```

### What to Expect (Multi-Node)

1. Both master and worker start, download checkpoints
2. NCCL handshake across nodes via EFA (~1-2 minutes)
3. "Iteration 1: ... | Time: 143.38s" — first iteration warmup
4. "Iteration 2: ... | Time: 5.77s" — steady state ~4.5s/iter
5. Checkpoints saved every 250 iterations to FSx

**Why multi-node is 4x faster (not 2x)**: Context parallelism splits the 93-frame video sequence across 8 GPUs within each node. With `context_parallel_size=8`, each GPU processes ~12 frames instead of 93. Since attention is O(n^2), this dramatically reduces compute per GPU.

---

## Step 10: Monitor Training

### From kubectl

```bash
# Quick status
kubectl get pytorchjobs
kubectl logs --tail=10 cosmos-transfer-train-master-0

# GPU utilization
kubectl exec cosmos-transfer-train-master-0 -- \
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv

# Check worker (multi-node only)
kubectl logs --tail=10 cosmos-transfer-train-worker-0
```

### From Weights & Biases

If WANDB_API_KEY is configured, training metrics are logged to:
`https://wandb.ai/<your-username>/cosmos_transfer2_posttrain`

Metrics include: loss per iteration, iteration speed, GPU memory, sample images at checkpoint intervals.

### Check Saved Checkpoints

```bash
kubectl run cosmos-check --rm -it \
    --image=ubuntu:22.04 \
    --overrides='{"spec":{"containers":[{"name":"check","image":"ubuntu:22.04","command":["bash"],"stdin":true,"tty":true,"volumeMounts":[{"name":"fsx","mountPath":"/fsx"}]}],"volumes":[{"name":"fsx","persistentVolumeClaim":{"claimName":"fsx-claim"}}]}}' \
    -- bash

# Inside:
ls /fsx/output/cosmos_transfer*/cosmos_transfer2_posttrain/drive_dreams/transfer2_drive_dreams_hdmap_posttrain/checkpoints/
exit
```

---

## Step 11: Sync Results to S3

Training outputs are on FSx. To preserve them, sync to S3:

```bash
kubectl run cosmos-s3up --rm -it \
    --image=ubuntu:22.04 \
    --overrides='{"spec":{"containers":[{"name":"s3up","image":"ubuntu:22.04","command":["bash"],"stdin":true,"tty":true,"volumeMounts":[{"name":"fsx","mountPath":"/fsx"}]}],"volumes":[{"name":"fsx","persistentVolumeClaim":{"claimName":"fsx-claim"}}]}}' \
    -- bash
```

Inside — install AWS CLI and configure credentials:

```bash
apt-get update -qq && apt-get install -y -qq curl unzip
curl -s "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
unzip -q /tmp/awscliv2.zip -d /tmp
/tmp/aws/install

# Configure with your AWS credentials
aws configure
# Enter: Access Key, Secret Key, Region (us-east-1), Format (json)

# Sync training outputs
aws s3 sync /fsx/output/ s3://YOUR-BUCKET/cosmos-transfer2.5/hyperpod_output/ --no-progress

# Sync patched code for reproducibility
aws s3 sync /fsx/cosmos-transfer2.5/cosmos_transfer2/experiments/singleview/ \
    s3://YOUR-BUCKET/cosmos-transfer2.5/experiment_configs/ --no-progress

exit
```

**Note**: Pods on HyperPod do not have AWS credentials by default. You must configure them manually inside the pod, or use IRSA (IAM Roles for Service Accounts).

---

## Step 12: Run Inference

Inference requires a p4de.24xlarge or p5.48xlarge (80GB VRAM per GPU).

### Scale Up Inference Node

```bash
aws sagemaker update-cluster \
    --cluster-name cosmos-training-cluster \
    --instance-groups file://hyperpod/scale-up-p4de-1node.json \
    --region us-west-2
```

### Convert DCP Checkpoint to .pt

Training saves distributed checkpoints (DCP format). For inference, convert to a single .pt file:

```bash
kubectl run cosmos-convert --rm -it \
    --image=<ACCOUNT_ID>.dkr.ecr.us-west-2.amazonaws.com/cosmos-transfer-training:v2 \
    --overrides='{"spec":{"nodeSelector":{"sagemaker.amazonaws.com/instance-group-name":"eval-p4de"},"tolerations":[{"key":"nvidia.com/gpu","operator":"Exists","effect":"NoSchedule"}],"containers":[{"name":"convert","image":"<ACCOUNT_ID>.dkr.ecr.us-west-2.amazonaws.com/cosmos-transfer-training:v2","command":["bash"],"stdin":true,"tty":true,"resources":{"limits":{"nvidia.com/gpu":"8"}},"volumeMounts":[{"name":"fsx","mountPath":"/fsx"}]}],"volumes":[{"name":"fsx","persistentVolumeClaim":{"claimName":"fsx-claim"}}]}}' \
    -- bash
```

Inside:

```bash
cd /fsx/cosmos-transfer2.5
source .venv/bin/activate
NVIDIA_LIB=$(find .venv -path "*/nvidia/cuda_nvrtc/lib" -type d | head -1)
export LD_LIBRARY_PATH="${NVIDIA_LIB}:${LD_LIBRARY_PATH}"

# Convert the best checkpoint (e.g., iter_5000)
CKPT_DIR="/fsx/output/cosmos_transfer_10k_5000iter/cosmos_transfer2_posttrain/drive_dreams/transfer2_drive_dreams_hdmap_posttrain/checkpoints/iter_000005000"
python scripts/convert_distcp_to_pt.py "$CKPT_DIR/model" "$CKPT_DIR"

# Output: $CKPT_DIR/model_ema_bf16.pt
ls -lh "$CKPT_DIR/"*.pt
```

### Run Single-View Inference

```bash
kubectl apply -f hyperpod/singleview-inference-job.yaml
kubectl logs -f cosmos-singleview-inference
```

Or manually inside a GPU pod:

```bash
# Generate inference specs
python3 scripts/generate_inference_specs.py \
    --test-dir /fsx/datasets/drive_dreams_10k_test \
    --output-dir /fsx/output/inference_specs

# Run inference
torchrun --nproc_per_node=8 --master_port=12341 \
    -m examples.inference \
    -i /fsx/output/inference_specs/ \
    -o /fsx/output/inference_results/ \
    --checkpoint-path "$CKPT_DIR/model_ema_bf16.pt" \
    --disable-guardrails
```


---

## Scaling GPU Nodes Up/Down

GPU nodes cost $98/hr (p5) or $40/hr (p4de). Scale them to 0 when not in use.

### Scale Down (stop all GPU nodes)

```bash
aws sagemaker update-cluster \
    --cluster-name cosmos-training-cluster \
    --instance-groups file://hyperpod/us-west-2/scale-down.json \
    --region us-west-2
```

This sets p5 and p4de counts to 0, keeping only the system node (~$0.68/hr).

### Scale Up for Training

```bash
# 1x p5 (single-node training)
aws sagemaker update-cluster \
    --cluster-name cosmos-training-cluster \
    --instance-groups file://hyperpod/us-west-2/scale-up-1p5.json \
    --region us-west-2

# 2x p5 with training plan (multi-node, reserved capacity)
aws sagemaker update-cluster \
    --cluster-name cosmos-training-cluster \
    --instance-groups file://hyperpod/us-west-2/scale-up-2p5-reserved.json \
    --region us-west-2
```

### Verify Nodes

```bash
kubectl get nodes -l sagemaker.amazonaws.com/instance-group-name -o custom-columns=\
'NAME:.metadata.name,GROUP:.metadata.labels.sagemaker\.amazonaws\.com/instance-group-name,GPU:.status.capacity.nvidia\.com/gpu'
```

---

## Training Results

### Checkpoints on FSx

```
/fsx/output/
|-- cosmos_transfer/                    # 450-clip run (200 iter, 1x p5)
|   +-- checkpoints: iter_050, iter_100, iter_150, iter_200
|
|-- cosmos_transfer_10k/                # 10K-clip run (500 iter, 2x p5)
|   +-- checkpoints: iter_050 through iter_500 (every 50)
|
+-- cosmos_transfer_10k_5000iter/       # 10K-clip run (5000 iter, 2x p5)
    +-- checkpoints: iter_250 through iter_5000 (every 250)
```

Each checkpoint directory contains:
- `model/` — distributed checkpoint (DCP) shards
- `optim/` — optimizer state
- `scheduler/` — learning rate scheduler state
- `trainer/` — trainer state (iteration count, etc.)

### Wandb Runs

- 450-clip: `https://wandb.ai/amin-dm89-aws/cosmos_transfer2_posttrain/runs/hwpuf6pq`
- 10K/500-iter: `https://wandb.ai/amin-dm89-aws/cosmos_transfer2_posttrain/runs/1koju8wn`

---

## Code Patches Explained

Cosmos Transfer 2.5 ships with support for `depth`, `edge`, `seg`, and `vis` control types. To add HD map (`hdmap_bbox`) support, 6 patches are needed:

| # | File | What it does |
|---|------|-------------|
| 1 | `singleview_dataset.py` | Adds `hdmap_bbox` to `CTRL_TYPE_INFO` dict — tells the dataset where to find HD map video files |
| 2 | `singleview_dataset.py` | Adds `hdmap_bbox` branch to `_load_control_data()` — reads HD map MP4 frames using decord |
| 3 | `control_input.py` | Fixes augmentor to treat `hdmap_bbox` as a single key instead of splitting on underscore |
| 4 | `dataloader_local.py` | Registers `example_singleview_train_data_hdmap_bbox` dataloader with Hydra ConfigStore |
| 5 | `cosmos_drive_dreams_hdmap.py` | Creates the experiment config that ties everything together |
| 6 | `cpp_extension.py` | Bypasses PyTorch CUDA version check (allows flash-attn/TE to work with different CUDA versions) |

All patches are applied automatically in `Dockerfile.v2`. For the FSx venv approach, they must be applied manually (see Step 7).

---

## Troubleshooting

### Training hangs after "Starting training..."

**Cause**: Dataset not flattened — videos are in `front_wide/` subdirectory.
**Fix**: Flatten the dataset (see Step 6).

### "Could not find data_train/example_singleview_train_data_hdmap_bbox"

**Cause**: Dataloader patch not applied to `dataloader_local.py`.
**Fix**: Apply Patch 4 (see Code Patches section).

### "Checkpoint path checkpoints/Cosmos-Transfer2.5-2B/... does not exist"

**Cause**: Missing symlink from repo's `checkpoints/` to FSx checkpoint location.
**Fix**: `mkdir -p checkpoints && ln -snf /fsx/checkpoints/Cosmos-Transfer2.5-2B checkpoints/Cosmos-Transfer2.5-2B`

### wandb.sdk.mailbox.mailbox.MailboxClosedError

**Cause**: Wandb tries to init but `/root/.netrc` doesn't exist and no API key is set.
**Fix**: Either set `WANDB_API_KEY` env var, or create empty `/root/.netrc` and set `WANDB_MODE=offline`.

### Multi-node training hangs at FSDP initialization

**Cause**: `context_parallel_size` set to total GPU count instead of per-node count.
**Fix**: Set `context_parallel_size=8` (GPUs per node), not `WORLD_SIZE`.

### OOM (Out of Memory)

**Cause**: Instance type has insufficient VRAM. Cosmos Transfer 2.5 needs ~70GB per GPU.
**Fix**: Use p5.48xlarge (H100 80GB) or p4de.24xlarge (A100 80GB). L40S (46GB) is too small.

### "no space left on device" during Docker build

**Cause**: SageMaker Studio has only 37GB overlay for Docker.
**Fix**: Use kaniko on HyperPod (see Step 4, Option A) or build on an EC2 instance with 200GB+ disk.

### Pods stuck in Pending

**Cause**: No GPU nodes available.
**Fix**: Scale up GPU nodes (see Scaling section). Check `kubectl describe pod <POD>` for details.

### NCCL timeout

**Cause**: EFA not configured or security group blocking traffic.
**Fix**: Ensure security group allows all TCP between nodes. Verify EFA is enabled on the node group.

---

## Cost Estimate

| Component | Instance | Count | On-Demand $/hr |
|-----------|----------|-------|----------------|
| System node | c5.4xlarge | 1 | ~$0.68 |
| Training (single) | p5.48xlarge | 1 | ~$98 |
| Training (multi) | p5.48xlarge | 2 | ~$196 |
| Inference/eval | p4de.24xlarge | 1 | ~$40 |
| FSx for Lustre | 1.2 TB PERSISTENT_2 | 1 | ~$0.36 |

GPU nodes scale to 0 when idle. You only pay for the system node + FSx between runs.

| Training Run | Duration | Cost (on-demand) |
|-------------|----------|-----------------|
| 200 iter, 1x p5 | ~67 min | ~$110 |
| 500 iter, 2x p5 | ~38 min | ~$125 |
| 5000 iter, 2x p5 | ~6.3 hrs | ~$1,235 |

Training plans (reserved capacity) reduce p5 cost significantly.

---

## kubectl Cheat Sheet

```bash
# --- Cluster ---
kubectl get nodes                                    # List all nodes
kubectl get nodes -o wide                            # Nodes with IPs and instance info
kubectl describe node <NODE>                         # Node details (GPU count, memory)

# --- Training Jobs ---
kubectl get pytorchjobs                              # List training jobs
kubectl describe pytorchjob cosmos-transfer-train    # Job details and events
kubectl get pods -l app=cosmos-training              # Training pods

# --- Logs ---
kubectl logs cosmos-transfer-train-master-0          # Master logs
kubectl logs cosmos-transfer-train-worker-0          # Worker logs
kubectl logs -f cosmos-transfer-train-master-0       # Follow logs (live)
kubectl logs --tail=20 cosmos-transfer-train-master-0  # Last 20 lines

# --- Debugging ---
kubectl exec -it cosmos-transfer-train-master-0 -- bash           # Shell into pod
kubectl exec cosmos-transfer-train-master-0 -- nvidia-smi         # GPU status
kubectl describe pod cosmos-transfer-train-master-0               # Pod events
kubectl get events --sort-by='.lastTimestamp'                      # Recent events

# --- Storage ---
kubectl get pvc                                      # Persistent volume claims
kubectl get pv                                       # Persistent volumes

# --- Cleanup ---
kubectl delete pytorchjob cosmos-transfer-train      # Delete training job
kubectl delete pod <POD_NAME>                        # Delete a pod
kubectl delete secret hf-token                       # Delete a secret

# --- Quick Setup Pod (for FSx access) ---
kubectl run myshell --rm -it --image=ubuntu:22.04 \
    --overrides='{"spec":{"containers":[{"name":"shell","image":"ubuntu:22.04","command":["bash"],"stdin":true,"tty":true,"volumeMounts":[{"name":"fsx","mountPath":"/fsx"}]}],"volumes":[{"name":"fsx","persistentVolumeClaim":{"claimName":"fsx-claim"}}]}}' \
    -- bash
```
