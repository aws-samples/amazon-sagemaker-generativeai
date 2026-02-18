#!/bin/bash
#
# Local Training Job Launcher - Mimics SageMaker Training Jobs
# Launches a fresh Docker container for each training job, just like SageMaker
#
# Usage:
#   ./launch_local_training_job.sh --config <CONFIG_YAML> [OPTIONS]

set -euo pipefail

############################################
# Configuration
############################################
readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Docker configuration
DOCKER_IMAGE="763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.8.0-gpu-py312"
CONTAINER_NAME="mt-grpo-training-$(date +%Y%m%d-%H%M%S)"

# Training configuration
CONFIG_PATH=""
NUM_GPUS=""
VLLM_PORT="8000"
DETACHED="true"  # Run container in background by default
FOLLOW_LOGS="true"  # Follow logs after launching by default

# Paths
HOST_CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"  # mt-grpo directory
HOST_DATA_DIR="${HOME}/mt-grpo-outputs"  # Use home directory instead of /opt/ml
CONTAINER_CODE_DIR="/app/mt-grpo"
CONTAINER_TRAINING_DIR="/app/mt-grpo/local_training"

############################################
# Logging
############################################
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1" >&2; }

############################################
# Usage
############################################
show_usage() {
    cat << EOF
Usage: $SCRIPT_NAME --config <CONFIG_YAML> [OPTIONS]

This script mimics SageMaker training jobs by launching a fresh Docker container
for each training run on your local P5 instance.

Arguments:
  --config CONFIG_YAML    Path to training configuration YAML file
                          (relative to local_training directory)

Options:
  --num_gpus N           Total GPU count (default: auto-detect)
  --vllm_port PORT       Port for vLLM server (default: 8000)
  --image IMAGE          Docker image to use (default: PyTorch 2.8.0)
  --name NAME            Container name (default: auto-generated)
  --foreground           Run container in foreground (default: background)
  --no-follow            Don't follow logs after launching (default: follow)
  --help, -h             Show this help message

Examples:
  # Launch training job with default settings
  $SCRIPT_NAME --config hf_recipes/Qwen/Qwen3-0.6B--mt-grpo.yaml

  # Launch with specific GPU count
  $SCRIPT_NAME --config hf_recipes/Qwen/Qwen3-0.6B--mt-grpo.yaml --num_gpus 8

  # Run in foreground to see output
  $SCRIPT_NAME --config hf_recipes/Qwen/Qwen3-0.6B--mt-grpo.yaml --foreground

Container Behavior (like SageMaker):
  - Fresh container for each training job
  - Code mounted from host
  - Outputs saved to /opt/ml/model
  - Logs available via docker logs
  - Auto-cleanup on completion
EOF
}

############################################
# Argument parsing
############################################
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --config)        CONFIG_PATH="${2:-}"; shift 2 ;;
            --num_gpus)      NUM_GPUS="${2:-}"; shift 2 ;;
            --vllm_port)     VLLM_PORT="${2:-}"; shift 2 ;;
            --image)         DOCKER_IMAGE="${2:-}"; shift 2 ;;
            --name)          CONTAINER_NAME="${2:-}"; shift 2 ;;
            --foreground)    DETACHED="false"; shift ;;
            --no-follow)     FOLLOW_LOGS="false"; shift ;;
            --help|-h)       show_usage; exit 0 ;;
            *)               log_error "Unknown argument: $1"; show_usage; exit 1 ;;
        esac
    done
}

############################################
# Validation
############################################
validate_inputs() {
    [[ -z "$CONFIG_PATH" ]] && { log_error "--config is required"; show_usage; exit 1; }
    
    # Check if config file exists (relative to local_training dir)
    local full_config_path="${SCRIPT_DIR}/${CONFIG_PATH}"
    if [[ ! -f "$full_config_path" ]]; then
        log_error "Config file not found: $full_config_path"
        exit 1
    fi
    
    # Check if Docker is available
    if ! command -v docker &> /dev/null; then
        log_error "Docker not found. Please install Docker."
        exit 1
    fi
    
    # Check if Docker image exists
    if ! docker image inspect "$DOCKER_IMAGE" &> /dev/null; then
        log_warning "Docker image not found locally: $DOCKER_IMAGE"
        log_info "Pulling image... (this may take a while)"
        docker pull "$DOCKER_IMAGE" || { log_error "Failed to pull image"; exit 1; }
    fi
}

############################################
# GPU detection
############################################
detect_gpus() {
    if [[ -n "${NUM_GPUS:-}" ]]; then
        log_info "Using specified GPU count: $NUM_GPUS"
        return
    fi
    
    if command -v nvidia-smi &> /dev/null; then
        NUM_GPUS="$(nvidia-smi -L | wc -l | tr -d '[:space:]')"
        log_info "Auto-detected $NUM_GPUS GPUs"
    else
        log_error "nvidia-smi not found. Please specify --num_gpus"
        exit 1
    fi
    
    if (( NUM_GPUS < 2 )); then
        log_error "Need at least 2 GPUs (1 for vLLM, 1+ for training)"
        exit 1
    fi
}

############################################
# Create output directories
############################################
setup_directories() {
    log_info "Setting up directories..."
    
    # Create host directories for outputs in user's home directory
    mkdir -p "${HOST_DATA_DIR}/model"
    mkdir -p "${HOST_DATA_DIR}/output"
    mkdir -p "${HOST_DATA_DIR}/checkpoints"
    
    log_info "Output directory: ${HOST_DATA_DIR}"
    log_success "Directories ready"
}

############################################
# Launch training container
############################################
launch_container() {
    log_info "Launching training container..."
    log_info "  Container name: $CONTAINER_NAME"
    log_info "  Docker image: $DOCKER_IMAGE"
    log_info "  Config: $CONFIG_PATH"
    log_info "  GPUs: $NUM_GPUS"
    log_info "  Mode: $([ "$DETACHED" = "true" ] && echo "background" || echo "foreground")"
    
    # Build docker run command
    local docker_cmd=(
        docker run
        --gpus all
        --shm-size=256g
        --name "$CONTAINER_NAME"
        -v "${HOST_CODE_DIR}:${CONTAINER_CODE_DIR}"
        -v "${HOST_DATA_DIR}:/opt/ml"
        -e "JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64"
        -e "PATH=/usr/lib/jvm/java-21-openjdk-amd64/bin:/usr/local/bin:/usr/bin:/bin"
        -e "WANDB_API_KEY=${WANDB_API_KEY:-}"
        -e "WANDB_ENTITY=${WANDB_ENTITY:-}"
        -e "WANDB_PROJECT=${WANDB_PROJECT:-mt-grpo-training}"
        -w "${CONTAINER_TRAINING_DIR}"
    )
    
    # Add detached flag if running in background
    if [[ "$DETACHED" = "true" ]]; then
        docker_cmd+=(-d)
    else
        docker_cmd+=(-it)
    fi
    
    # Don't auto-remove on failure so we can debug
    # docker_cmd+=(--rm)  # Commented out for debugging
    
    # Add the image
    docker_cmd+=("$DOCKER_IMAGE")
    
    # Add the training command
    docker_cmd+=(
        bash -c "
            set -e
            
            # Install Java 21 (required for Pyserini)
            echo '==> Installing Java 21...'
            apt-get update -qq
            apt-get install -y -qq openjdk-21-jdk
            export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
            export PATH=\$JAVA_HOME/bin:\$PATH
            
            # Install Python dependencies
            echo '==> Installing Python dependencies...'
            pip install -U kernels  # Upgrade kernels first for Qwen3 support
            pip install -r requirements_local.txt
            
            # Run training
            echo '==> Starting training...'
            bash local_mt_grpo_train.sh \
                --config ${CONFIG_PATH} \
                --num_process ${NUM_GPUS} \
                --vllm_port ${VLLM_PORT}
        "
    )
    
    # Execute docker run
    if [[ "$DETACHED" = "true" ]]; then
        "${docker_cmd[@]}"
        log_success "Training container launched: $CONTAINER_NAME"
    else
        log_info "Running in foreground (Ctrl+C to stop)..."
        "${docker_cmd[@]}" || log_warning "Training exited with non-zero status"
    fi
}

############################################
# Show monitoring instructions
############################################
show_monitoring_instructions() {
    if [[ "$DETACHED" = "true" ]]; then
        echo ""
        log_info "╔══════════════════════════════════════════════════════════════════╗"
        log_info "║                    TRAINING JOB LAUNCHED                         ║"
        log_info "╠══════════════════════════════════════════════════════════════════╣"
        log_info "║  Container: $CONTAINER_NAME"
        log_info "╠══════════════════════════════════════════════════════════════════╣"
        log_info "║  MONITORING COMMANDS                                             ║"
        log_info "║                                                                  ║"
        log_info "║  View logs (follow):                                             ║"
        log_info "║    docker logs -f $CONTAINER_NAME"
        log_info "║                                                                  ║"
        log_info "║  View logs (last 100 lines):                                     ║"
        log_info "║    docker logs --tail 100 $CONTAINER_NAME"
        log_info "║                                                                  ║"
        log_info "║  Check container status:                                         ║"
        log_info "║    docker ps -a | grep $CONTAINER_NAME"
        log_info "║                                                                  ║"
        log_info "║  Enter container (for debugging):                                ║"
        log_info "║    docker exec -it $CONTAINER_NAME bash"
        log_info "║                                                                  ║"
        log_info "║  Stop training:                                                  ║"
        log_info "║    docker stop $CONTAINER_NAME"
        log_info "║                                                                  ║"
        log_info "║  Check GPU usage:                                                ║"
        log_info "║    nvidia-smi                                                    ║"
        log_info "╠══════════════════════════════════════════════════════════════════╣"
        log_info "║  OUTPUTS                                                         ║"
        log_info "║                                                                  ║"
        log_info "║  Output directory: ${HOST_DATA_DIR}"
        log_info "║    Model checkpoints: ${HOST_DATA_DIR}/checkpoints/"
        log_info "║    Final model:       ${HOST_DATA_DIR}/model/"
        log_info "║    Training output:   ${HOST_DATA_DIR}/output/"
        log_info "╚══════════════════════════════════════════════════════════════════╝"
        echo ""
    fi
}

############################################
# Follow container logs
############################################
follow_logs() {
    if [[ "$DETACHED" = "true" ]] && [[ "$FOLLOW_LOGS" = "true" ]]; then
        echo ""
        log_info "Following container logs (Ctrl+C to stop watching, training continues)..."
        echo ""
        sleep 2
        docker logs -f "$CONTAINER_NAME"
    fi
}

############################################
# Main
############################################
main() {
    echo ""
    log_info "╔══════════════════════════════════════════════════════════════════╗"
    log_info "║          Local Training Job Launcher (SageMaker-style)           ║"
    log_info "╚══════════════════════════════════════════════════════════════════╝"
    echo ""
    
    parse_arguments "$@"
    validate_inputs
    detect_gpus
    setup_directories
    launch_container
    show_monitoring_instructions
    follow_logs  # Automatically follow logs after launching
    
    log_success "Training job setup complete!"
}

main "$@"
