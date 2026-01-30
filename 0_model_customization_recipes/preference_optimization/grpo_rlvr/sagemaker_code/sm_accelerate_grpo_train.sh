#!/bin/bash
#
# SageMaker Accelerate GRPO Training Script
# - Trains a model using GRPO with tool calling via Accelerate + DeepSpeed
#
# Usage:
#   ./sm_accelerate_grpo_train.sh --config <CONFIG_YAML> [--num_process <N>]
#
# Notes:
#   --num_process is per-machine GPU process count (maps to accelerate --num_processes).
#   This script is safe under `set -euo pipefail` and avoids unbound-var issues.

set -euo pipefail

############################################
# Configuration (use absolute paths)
############################################
readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults (can be overridden by CLI)
NUM_GPUS=""
CONFIG_PATH=""
TOOLS_SCRIPT=""
REWARD_FN=""

# Default tool functions and reward function paths
DEFAULT_TOOLS_SCRIPT="${SCRIPT_DIR}/tools_funcs/financial_tools_complex.py"
DEFAULT_REWARD_FN="${SCRIPT_DIR}/rewards/financial_tools_reward.py"

# Repo-local assets (absolute)
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements.txt"
ACCELERATE_CONFIG="${SCRIPT_DIR}/configs/accelerate/ds_zero3.yaml"
TRAINING_SCRIPT="${SCRIPT_DIR}/grpo_trainer_v2.py"

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

Arguments:
  --config CONFIG_YAML    Path to training configuration YAML file

Options:
  --num_process N         Per-machine process count (usually = GPUs per node)
  --tools_script PATH     Path to custom tool functions script (must export TOOL_FUNCTIONS list)
  --reward_fn PATH        Path to custom reward function script (must export reward_func callable)
  --help, -h              Show this help message

Examples:
  # Basic usage with default tools and reward function
  $SCRIPT_NAME --config ${SCRIPT_DIR}/recipes/Qwen/Qwen3-0.6B--grpo.yaml

  # Custom tools and reward function
  $SCRIPT_NAME --config ${SCRIPT_DIR}/recipes/Qwen/Qwen3-0.6B--grpo.yaml \\
    --tools_script ${SCRIPT_DIR}/tools_funcs/my_custom_tools.py \\
    --reward_fn ${SCRIPT_DIR}/rewards/my_reward.py

  # Multi-GPU training
  $SCRIPT_NAME --config ${SCRIPT_DIR}/recipes/Qwen/Qwen3-1.7B--grpo.yaml --num_process 4
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
            --help|-h)       show_usage; exit 0 ;;
            *)               log_error "Unknown argument: $1"; show_usage; exit 1 ;;
        esac
    done
}

############################################
# GPU discovery (safe with set -u)
############################################
resolve_num_gpus() {
    if [[ -n "${NUM_GPUS:-}" ]]; then
        NUM_GPUS="$(printf '%s' "$NUM_GPUS" | tr -d '[:space:]')"
        return
    fi
    if [[ -n "${SM_NUM_GPUS:-}" ]]; then
        NUM_GPUS="$(printf '%s' "$SM_NUM_GPUS" | tr -d '[:space:]')"
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
# Input validation
############################################
validate_inputs() {
    [[ -z "$CONFIG_PATH" ]] && { log_error "--config is required"; show_usage; exit 1; }

    validate_file_exists "$CONFIG_PATH" "Configuration file"
    validate_file_exists "$TRAINING_SCRIPT" "Training script"
    validate_file_exists "$ACCELERATE_CONFIG" "Accelerate configuration"

    resolve_num_gpus
    validate_positive_integer "$NUM_GPUS" "GPU count"
}

############################################
# Resolve tool functions and reward function defaults
############################################
resolve_tools_and_reward() {
    # Set defaults if not provided
    if [[ -z "${TOOLS_SCRIPT:-}" ]]; then
        TOOLS_SCRIPT="$DEFAULT_TOOLS_SCRIPT"
        log_info "Using default tools script: $TOOLS_SCRIPT"
    fi

    if [[ -z "${REWARD_FN:-}" ]]; then
        REWARD_FN="$DEFAULT_REWARD_FN"
        log_info "Using default reward function: $REWARD_FN"
    fi

    # Validate files exist
    validate_file_exists "$TOOLS_SCRIPT" "Tools script"
    validate_file_exists "$REWARD_FN" "Reward function"
}

############################################
# Print configuration summary
############################################
print_configuration() {
    echo ""
    log_info "╔══════════════════════════════════════════════════════════════════╗"
    log_info "║                    GRPO TRAINING CONFIGURATION                   ║"
    log_info "╠══════════════════════════════════════════════════════════════════╣"
    log_info "║  Config File:     $(basename "$CONFIG_PATH")"
    log_info "║  Model Script:    $(basename "$TRAINING_SCRIPT")"
    log_info "╠══════════════════════════════════════════════════════════════════╣"
    log_info "║  TOOL FUNCTIONS                                                  ║"
    log_info "║    Path: $TOOLS_SCRIPT"
    log_info "╠══════════════════════════════════════════════════════════════════╣"
    log_info "║  REWARD FUNCTION                                                 ║"
    log_info "║    Path: $REWARD_FN"
    log_info "╠══════════════════════════════════════════════════════════════════╣"
    log_info "║  DISTRIBUTED SETUP                                               ║"
    log_info "║    Num Machines:      ${NUM_MACHINES}"
    log_info "║    GPUs per Machine:  ${PER_NODE_PROCS}"
    log_info "║    Total Processes:   ${TOTAL_PROCS}"
    log_info "║    Machine Rank:      ${MACHINE_RANK}"
    log_info "║    Master Address:    ${MASTER_ADDR}:${MASTER_PORT}"
    log_info "╚══════════════════════════════════════════════════════════════════╝"
    echo ""
}

############################################
# Dependencies (uv + tools used later)
############################################
install_dependencies() {
    # uv (fast pip)
    if ! command -v uv &> /dev/null; then
        python3 -m pip install --upgrade uv || { log_error "Failed to install uv"; exit 1; }
    fi

    # Project deps
    [[ -f "$REQUIREMENTS_FILE" ]] && uv pip install --system -r "$REQUIREMENTS_FILE"

    # yq is required for dynamic accelerate backend
    if ! command -v yq &> /dev/null; then
        uv pip install --system yq || { log_error "Failed to install yq"; exit 1; }
    fi
}

############################################
# Accelerate presence (binary or importable)
############################################
check_accelerate_installation() {
    if command -v accelerate &> /dev/null; then
        return
    fi
    python3 - <<'PY' || { echo >&2 "[ERROR] accelerate not found"; exit 1; }
import importlib, sys
try:
    importlib.import_module("accelerate")
except Exception:
    sys.exit(1)
PY
}

############################################
# Patch accelerate backend dynamically (via yq)
############################################
set_dynamic_rdzv_backend() {
    local desired="static"
    (( NUM_MACHINES > 1 )) && desired="c10d"
    log_info "Setting accelerate rdzv_backend -> ${desired} in: $ACCELERATE_CONFIG"

    if yq --version 2>&1 | grep -qi 'mikefarah'; then
        NEW_BACKEND="$desired" yq -i -y '.rdzv_backend = strenv(NEW_BACKEND)' "$ACCELERATE_CONFIG"
    else
        if ! command -v jq >/dev/null; then
            log_error "Python yq detected but jq is missing. Install jq or switch to mikefarah/yq."
            exit 1
        fi
        NEW_BACKEND="$desired" yq -yi '.rdzv_backend = env.NEW_BACKEND' "$ACCELERATE_CONFIG"
    fi

    yq -r '.rdzv_backend' "$ACCELERATE_CONFIG" | xargs -I{} echo "[INFO] Effective rdzv_backend: {}"
}

############################################
# Distributed environment (pure bash)
############################################
setup_distributed_environment() {
    log_info "Setting up distributed training environment variables"

    _trim() { local s="$1"; s="${s#"${s%%[![:space:]]*}"}"; s="${s%"${s##*[![:space:]]}"}"; printf '%s' "$s"; }
    _unquote() { local s="$1"; [[ "$s" == \"*\" ]] && s="${s#\"}"; [[ "$s" == *\" ]] && s="${s%\"}"; printf '%s' "$s"; }

    local hosts_json="${SM_HOSTS:-"[\"localhost\"]"}"
    local inner
    inner="$(printf '%s' "$hosts_json" | sed -e 's/^\s*\[\s*//' -e 's/\s*\]\s*$//')"
    IFS=',' read -r -a _items <<< "$inner"

    local hosts=()
    local item cleaned
    for item in "${_items[@]}"; do
        cleaned="$(_unquote "$(_trim "$item")")"
        [[ -n "$cleaned" ]] && hosts+=("$cleaned")
    done
    if [[ ${#hosts[@]} -eq 0 ]]; then hosts=("127.0.0.1"); fi

    NUM_MACHINES=${#hosts[@]}

    local current="${SM_CURRENT_HOST:-localhost}"
    MACHINE_RANK=0
    for i in "${!hosts[@]}"; do
        if [[ "${hosts[$i]}" == "$current" ]]; then MACHINE_RANK="$i"; break; fi
    done

    MASTER_ADDR="${hosts[0]}"
    MASTER_PORT="${MASTER_PORT:-29500}"

    if command -v getent &> /dev/null; then
        local ip; ip="$(getent ahostsv4 "$MASTER_ADDR" 2>/dev/null | awk 'NR==1{print $1}')"
        [[ -n "${ip:-}" ]] && MASTER_ADDR="$ip"
    fi

    # Compute per-node and total procs, export Elastic envs
    PER_NODE_PROCS="${NUM_GPUS}"
    TOTAL_PROCS=$(( NUM_MACHINES * PER_NODE_PROCS ))
    export MACHINE_RANK MASTER_ADDR MASTER_PORT NUM_MACHINES PER_NODE_PROCS TOTAL_PROCS
    export LOCAL_WORLD_SIZE="$PER_NODE_PROCS"
    export WORLD_SIZE="$TOTAL_PROCS"
    export NODE_RANK="$MACHINE_RANK"
}

############################################
# Training launch
############################################
launch_training() {
    log_info "Starting distributed GRPO training..."

    if accelerate launch \
        --config_file "$ACCELERATE_CONFIG" \
        --num_machines "$NUM_MACHINES" \
        --machine_rank "$MACHINE_RANK" \
        --num_processes "$TOTAL_PROCS" \
        --main_process_ip "$MASTER_ADDR" \
        --main_process_port "$MASTER_PORT" \
        "$TRAINING_SCRIPT" \
        --config "$CONFIG_PATH" \
        --tools_script "$TOOLS_SCRIPT" \
        --reward_fn "$REWARD_FN"
    then
        log_success "GRPO Training completed successfully!"
    else
        local exit_code=$?
        log_error "GRPO Training failed with exit code: $exit_code"
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
    setup_distributed_environment
    resolve_tools_and_reward
    set_dynamic_rdzv_backend

    # Print nice configuration summary
    print_configuration

    # Print the DeepSpeed configuration
    log_info "******************* Start of DeepSpeed Configuration *******************"
    more "$ACCELERATE_CONFIG"
    log_info "******************** End of DeepSpeed Configuration ********************"

    check_accelerate_installation
    launch_training

    log_success "All steps completed successfully"
}

main "$@"
