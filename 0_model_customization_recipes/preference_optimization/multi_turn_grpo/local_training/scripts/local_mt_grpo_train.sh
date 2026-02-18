#!/bin/bash
#
# Local P5 Multi-Turn GRPO Training Script
# Adapted from SageMaker version for local EC2 execution
# - Starts vLLM server on the last GPU (GPU N-1)
# - Trains using MT-GRPO on GPUs 0 to N-2 via Accelerate + DeepSpeed
#
# Usage:
#   ./local_mt_grpo_train.sh --config <CONFIG_YAML> [--num_process <N>]

set -euo pipefail

############################################
# Configuration
############################################
readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults
NUM_GPUS=""
CONFIG_PATH=""
TOOLS_SCRIPT=""
REWARD_FN=""
VLLM_MODEL=""
VLLM_PORT="8000"
VLLM_PID=""

# Default paths
DEFAULT_TOOLS_SCRIPT="${SCRIPT_DIR}/tools_funcs/wiki_search.py"
DEFAULT_REWARD_FN="${SCRIPT_DIR}/rewards/triviaqa_reward.py"

# Local assets
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements_local.txt"
ACCELERATE_CONFIG="${SCRIPT_DIR}/configs/accelerate/ds_zero3.yaml"
TRAINING_SCRIPT="${SCRIPT_DIR}/mt_grpo_trainer.py"

############################################
# Logging
############################################
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1" >&2; }

############################################
# Cleanup function for vLLM server
############################################
cleanup_vllm() {
    if [[ -n "${VLLM_PID:-}" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        log_info "Stopping vLLM server (PID: $VLLM_PID)..."
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
        log_info "vLLM server stopped"
    fi
}

trap cleanup_vllm EXIT INT TERM

############################################
# Usage
############################################
show_usage() {
    cat << EOF
Usage: $SCRIPT_NAME --config <CONFIG_YAML> [OPTIONS]

Arguments:
  --config CONFIG_YAML    Path to training configuration YAML file

Options:
  --num_process N         Total GPU count (default: auto-detect)
  --tools_script PATH     Path to custom tool functions script
  --reward_fn PATH        Path to custom reward function script
  --vllm_model MODEL      Model for vLLM server (default: extracted from config)
  --vllm_port PORT        Port for vLLM server (default: 8000)
  --help, -h              Show this help message

GPU Allocation:
  - GPU N-1: vLLM generation server
  - GPUs 0 to N-2: Training with Accelerate/DeepSpeed
  - Minimum 2 GPUs required

Examples:
  # Basic usage (auto-detect GPUs)
  $SCRIPT_NAME --config ${SCRIPT_DIR}/hf_recipes/Qwen/Qwen3-0.6B--mt-grpo.yaml

  # Specify GPU count explicitly
  $SCRIPT_NAME --config ${SCRIPT_DIR}/hf_recipes/Qwen/Qwen3-1.7B--mt-grpo.yaml --num_process 8
EOF
}

############################################
# Validators
############################################
validate_file_exists() {
    [[ -f "$1" ]] || { log_error "$2 not found: $1"; exit 1; }
}

validate_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]] || { log_error "$2 must be a positive integer, got: $1"; exit 1; }
}

############################################
# Argument parsing
############################################
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --num_process)   NUM_GPUS="${2:-}"; shift 2 ;;
            --config)        CONFIG_PATH="${2:-}"; shift 2 ;;
            --tools_script)  TOOLS_SCRIPT="${2:-}"; shift 2 ;;
            --reward_fn)     REWARD_FN="${2:-}"; shift 2 ;;
            --vllm_model)    VLLM_MODEL="${2:-}"; shift 2 ;;
            --vllm_port)     VLLM_PORT="${2:-}"; shift 2 ;;
            --help|-h)       show_usage; exit 0 ;;
            *)               log_error "Unknown argument: $1"; show_usage; exit 1 ;;
        esac
    done
}

############################################
# GPU discovery
############################################
resolve_num_gpus() {
    if [[ -n "${NUM_GPUS:-}" ]]; then
        NUM_GPUS="$(printf '%s' "$NUM_GPUS" | tr -d '[:space:]')"
        return
    fi
    if command -v nvidia-smi &> /dev/null; then
        NUM_GPUS="$(nvidia-smi -L | wc -l | tr -d '[:space:]')"
        [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]] && return
    fi
    log_error "Unable to determine GPU count. Please specify --num_process."
    exit 1
}

############################################
# Validate minimum GPU requirement
############################################
validate_gpu_count() {
    if (( NUM_GPUS < 2 )); then
        log_error "Minimum 2 GPUs required (1 for vLLM, 1+ for training)"
        log_error "Detected GPUs: $NUM_GPUS"
        exit 1
    fi
    log_info "Detected $NUM_GPUS GPUs"
}

############################################
# Extract model name from config YAML
############################################
extract_model_from_config() {
    if [[ -n "${VLLM_MODEL:-}" ]]; then
        log_info "Using specified vLLM model: $VLLM_MODEL"
        return
    fi

    # Try to extract model_name_or_path from YAML config
    if command -v yq &> /dev/null; then
        VLLM_MODEL=$(yq -r '.model_name_or_path // .model // empty' "$CONFIG_PATH" 2>/dev/null || true)
    fi

    # Fallback: grep for model_name_or_path
    if [[ -z "${VLLM_MODEL:-}" ]]; then
        VLLM_MODEL=$(grep -E '^\s*model_name_or_path:' "$CONFIG_PATH" 2>/dev/null | head -1 | sed 's/.*:\s*//' | tr -d '"' | tr -d "'" || true)
    fi

    if [[ -z "${VLLM_MODEL:-}" ]]; then
        log_error "Could not extract model name from config. Please specify --vllm_model"
        exit 1
    fi

    log_info "Extracted model from config: $VLLM_MODEL"
}

############################################
# Input validation
############################################
validate_inputs() {
    [[ -z "$CONFIG_PATH" ]] && { log_error "--config is required"; show_usage; exit 1; }

    validate_file_exists "$CONFIG_PATH" "Configuration file"
    validate_file_exists "$TRAINING_SCRIPT" "Training script"
    validate_file_exists "$ACCELERATE_CONFIG" "Accelerate configuration"

    resolve_num_gpus
    validate_positive_integer "$NUM_GPUS" "GPU count"
    validate_gpu_count
}

############################################
# Resolve tool functions and reward function defaults
############################################
resolve_tools_and_reward() {
    if [[ -z "${TOOLS_SCRIPT:-}" ]]; then
        TOOLS_SCRIPT="$DEFAULT_TOOLS_SCRIPT"
        log_info "Using default tools script: $TOOLS_SCRIPT"
    fi

    if [[ -z "${REWARD_FN:-}" ]]; then
        REWARD_FN="$DEFAULT_REWARD_FN"
        log_info "Using default reward function: $REWARD_FN"
    fi

    validate_file_exists "$TOOLS_SCRIPT" "Tools script"
    validate_file_exists "$REWARD_FN" "Reward function"
}

############################################
# Compute GPU allocation
############################################
compute_gpu_allocation() {
    # vLLM gets the last GPU (N-1)
    VLLM_GPU=$((NUM_GPUS - 1))
    
    # Training gets GPUs 0 to N-2
    TRAINING_GPU_COUNT=$((NUM_GPUS - 1))
    
    # Build CUDA_VISIBLE_DEVICES string for training (0,1,2,...,N-2)
    TRAINING_GPUS=""
    for ((i=0; i<TRAINING_GPU_COUNT; i++)); do
        if [[ -n "$TRAINING_GPUS" ]]; then
            TRAINING_GPUS="${TRAINING_GPUS},$i"
        else
            TRAINING_GPUS="$i"
        fi
    done

    log_info "GPU Allocation:"
    log_info "  vLLM Server:  GPU $VLLM_GPU"
    log_info "  Training:     GPUs $TRAINING_GPUS ($TRAINING_GPU_COUNT GPUs)"
}

############################################
# Start vLLM server
############################################
start_vllm_server() {
    log_info "Starting vLLM server on GPU $VLLM_GPU..."
    log_info "  Model: $VLLM_MODEL"
    log_info "  Port: $VLLM_PORT"

    # Start vLLM OpenAI-compatible server
    CUDA_VISIBLE_DEVICES="$VLLM_GPU" python -m vllm.entrypoints.openai.api_server \
        --model "$VLLM_MODEL" \
        --tensor-parallel-size 1 \
        --port "$VLLM_PORT" \
        --trust-remote-code \
        --disable-log-requests \
        > "${SCRIPT_DIR}/vllm_server.log" 2>&1 &
    
    VLLM_PID=$!
    log_info "vLLM server started with PID: $VLLM_PID"
}

############################################
# Wait for vLLM server to be ready
############################################
wait_for_vllm_server() {
    local max_attempts=60
    local attempt=0
    local wait_time=5

    log_info "Waiting for vLLM server to be ready (max ${max_attempts} attempts, ${wait_time}s interval)..."

    while (( attempt < max_attempts )); do
        attempt=$((attempt + 1))
        
        # Check if process is still running
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            log_error "vLLM server process died unexpectedly"
            log_error "Check logs at: ${SCRIPT_DIR}/vllm_server.log"
            cat "${SCRIPT_DIR}/vllm_server.log" 2>/dev/null | tail -50 || true
            exit 1
        fi

        # Try to connect to the server
        if curl -s --max-time 5 "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
            log_success "vLLM server is ready! (attempt $attempt)"
            return 0
        fi

        # Alternative: check /v1/models endpoint
        if curl -s --max-time 5 "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; then
            log_success "vLLM server is ready! (attempt $attempt)"
            return 0
        fi

        log_info "  Attempt $attempt/$max_attempts - server not ready yet, waiting ${wait_time}s..."
        sleep "$wait_time"
    done

    log_error "vLLM server failed to start within $((max_attempts * wait_time)) seconds"
    log_error "Check logs at: ${SCRIPT_DIR}/vllm_server.log"
    cat "${SCRIPT_DIR}/vllm_server.log" 2>/dev/null | tail -50 || true
    exit 1
}

############################################
# Print configuration summary
############################################
print_configuration() {
    echo ""
    log_info "╔══════════════════════════════════════════════════════════════════╗"
    log_info "║                  MT-GRPO TRAINING CONFIGURATION                  ║"
    log_info "╠══════════════════════════════════════════════════════════════════╣"
    log_info "║  Config File:     $(basename "$CONFIG_PATH")"
    log_info "║  Model Script:    $(basename "$TRAINING_SCRIPT")"
    log_info "╠══════════════════════════════════════════════════════════════════╣"
    log_info "║  vLLM SERVER                                                     ║"
    log_info "║    Model:         $VLLM_MODEL"
    log_info "║    GPU:           $VLLM_GPU"
    log_info "║    Port:          $VLLM_PORT"
    log_info "║    PID:           $VLLM_PID"
    log_info "╠══════════════════════════════════════════════════════════════════╣"
    log_info "║  TOOL FUNCTIONS                                                  ║"
    log_info "║    Path: $TOOLS_SCRIPT"
    log_info "╠══════════════════════════════════════════════════════════════════╣"
    log_info "║  REWARD FUNCTION                                                 ║"
    log_info "║    Path: $REWARD_FN"
    log_info "╠══════════════════════════════════════════════════════════════════╣"
    log_info "║  DISTRIBUTED TRAINING                                            ║"
    log_info "║    Total GPUs:        $NUM_GPUS"
    log_info "║    Training GPUs:     $TRAINING_GPUS ($TRAINING_GPU_COUNT GPUs)"
    log_info "║    Total Processes:   ${TRAINING_GPU_COUNT}"
    log_info "╚══════════════════════════════════════════════════════════════════╝"
    echo ""
}

############################################
# Dependencies installation
############################################
install_dependencies() {
    log_info "Installing dependencies..."
    
    # Install Java for Pyserini
    log_info "Checking Java installation..."
    if ! command -v java &> /dev/null; then
        log_error "Java not found. Please install OpenJDK 21:"
        log_error "  apt-get update && apt-get install -y openjdk-21-jdk"
        exit 1
    fi
    
    export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
    export PATH=$JAVA_HOME/bin:$PATH
    log_info "Java found: $(java -version 2>&1 | head -1)"
    
    # Install Python dependencies
    if [[ -f "$REQUIREMENTS_FILE" ]]; then
        log_info "Installing Python packages from $REQUIREMENTS_FILE..."
        pip install -q -r "$REQUIREMENTS_FILE"
    fi
    
    # Install yq if not present
    if ! command -v yq &> /dev/null; then
        log_info "Installing yq..."
        pip install -q yq
    fi
    
    # Pre-download Pyserini Wikipedia index
    log_info "Pre-downloading Pyserini Wikipedia index (10GB, may take several minutes)..."
    python3 -c "
import os
os.environ['JAVA_HOME'] = '${JAVA_HOME}'
from pyserini.search.lucene import LuceneSearcher
searcher = LuceneSearcher.from_prebuilt_index('wikipedia-kilt-doc')
print('Pyserini index ready')
" 2>&1 | tail -10 || log_warning "Failed to pre-download Pyserini index"
    log_success "Dependencies installed"
}

############################################
# Setup distributed environment
############################################
setup_distributed_environment() {
    log_info "Setting up distributed training environment"
    
    export MASTER_ADDR="127.0.0.1"
    export MASTER_PORT="29500"
    export WORLD_SIZE="$TRAINING_GPU_COUNT"
    export LOCAL_WORLD_SIZE="$TRAINING_GPU_COUNT"
}

############################################
# Training launch
############################################
launch_training() {
    log_info "Starting distributed MT-GRPO training on GPUs: $TRAINING_GPUS"

    if CUDA_VISIBLE_DEVICES="$TRAINING_GPUS" accelerate launch \
        --config_file "$ACCELERATE_CONFIG" \
        --num_processes "$TRAINING_GPU_COUNT" \
        "$TRAINING_SCRIPT" \
        --config "$CONFIG_PATH" \
        --tools_script "$TOOLS_SCRIPT" \
        --reward_fn "$REWARD_FN" \
        --vllm_server_host "localhost" \
        --vllm_server_port "$VLLM_PORT"
    then
        log_success "MT-GRPO Training completed successfully!"
    else
        local exit_code=$?
        log_error "MT-GRPO Training failed with exit code: $exit_code"
        exit "$exit_code"
    fi
}

############################################
# Main
############################################
main() {
    parse_arguments "$@"
    validate_inputs
    install_dependencies
    
    compute_gpu_allocation
    extract_model_from_config
    setup_distributed_environment
    resolve_tools_and_reward

    start_vllm_server
    wait_for_vllm_server

    print_configuration

    log_info "******************* DeepSpeed Configuration *******************"
    cat "$ACCELERATE_CONFIG"
    log_info "****************************************************************"

    launch_training

    log_success "All steps completed successfully"
}

main "$@"
