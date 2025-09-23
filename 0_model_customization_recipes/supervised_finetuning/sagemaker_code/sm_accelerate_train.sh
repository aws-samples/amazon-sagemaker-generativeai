#!/bin/bash

# SageMaker Accelerate Training Script
# Launches distributed fine-tuning using Accelerate + DeepSpeed Zero3
#
# Usage: ./sm_accelerate_train.sh --config <CONFIG_YAML>
# Example: ./sm_accelerate_train.sh --config recipes/llama_sft.yaml

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Script configuration
readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
NUM_GPUS=""
CONFIG_PATH=""
REQUIREMENTS_FILE="./requirements.txt"
ACCELERATE_CONFIG="configs/accelerate/ds_zero3.yaml"
TRAINING_SCRIPT="sft.py"

# Color codes for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Logging functions
log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1" >&2; }

show_usage() {
    cat << EOF
Usage: $SCRIPT_NAME --config <CONFIG_YAML>

Arguments:
    --config CONFIG_YAML      Path to training configuration YAML file

Options:
    --help                    Show this help message

Examples:
    $SCRIPT_NAME --config recipes/llama_sft.yaml
    $SCRIPT_NAME --config configs/custom_config.yaml

Environment Variables:
    REQUIREMENTS_FILE         Path to requirements file (default: ./requirements.txt)
    ACCELERATE_CONFIG         Path to accelerate config (default: accelerate/zero3.yaml)
    TRAINING_SCRIPT           Path to training script (default: run_sft.py)
    SM_NUM_GPUS               Auto-detected GPU count (if set by container)

Auto GPU Detection:
    Priority:
      1. --num_process (deprecated, still supported)
      2. SM_NUM_GPUS environment variable
      3. nvidia-smi device count
EOF
}

validate_file_exists() {
    local file_path="$1"
    local file_description="$2"
    if [[ ! -f "$file_path" ]]; then
        log_error "$file_description not found: $file_path"
        return 1
    fi
}

validate_positive_integer() {
    local value="$1"
    local param_name="$2"
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        log_error "$param_name must be a positive integer, got: $value"
        return 1
    fi
}

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --num_process)
                NUM_GPUS="${2:-}"
                shift 2
                ;;
            --config)
                CONFIG_PATH="${2:-}"
                shift 2
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown argument: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

resolve_num_gpus() {
    if [[ -n "$NUM_GPUS" ]]; then
        log_info "Using --num_process=$NUM_GPUS"
        return
    fi

    if [[ -n "${SM_NUM_GPUS:-}" ]]; then
        NUM_GPUS="$SM_NUM_GPUS"
        log_info "Using SM_NUM_GPUS=$NUM_GPUS"
        return
    fi

    if command -v nvidia-smi &> /dev/null; then
        local detected
        detected=$(nvidia-smi -L | wc -l)
        if [[ "$detected" -gt 0 ]]; then
            NUM_GPUS="$detected"
            log_info "Detected $NUM_GPUS GPU(s) via nvidia-smi"
            return
        fi
    fi

    log_error "Unable to determine GPU count. Please specify --num_process explicitly."
    exit 1
}

validate_inputs() {
    local validation_failed=false
    
    if [[ -z "$CONFIG_PATH" ]]; then
        log_error "--config is required"
        validation_failed=true
    fi
    
    if [[ "$validation_failed" == true ]]; then
        show_usage
        exit 1
    fi
    
    resolve_num_gpus
    validate_positive_integer "$NUM_GPUS" "GPU count" || exit 1

    validate_file_exists "$CONFIG_PATH" "Configuration file" || exit 1
    validate_file_exists "$TRAINING_SCRIPT" "Training script" || exit 1
    validate_file_exists "$ACCELERATE_CONFIG" "Accelerate configuration" || exit 1
    
    if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
        log_warning "Requirements file not found: $REQUIREMENTS_FILE (skipping dependency installation)"
    fi
}

install_dependencies() {
    log_info "Ensuring uv package manager is installed..."
    if ! command -v uv &> /dev/null; then
        python3 -m pip install --upgrade uv || {
            log_error "Failed to install uv"
            exit 1
        }
        log_success "uv installed successfully"
    fi

    if [[ -f "$REQUIREMENTS_FILE" ]]; then
        log_info "Installing Python dependencies from $REQUIREMENTS_FILE using uv..."
        if ! uv pip install --system -r "$REQUIREMENTS_FILE"; then
            log_error "Failed to install dependencies from $REQUIREMENTS_FILE"
            exit 1
        fi
        log_success "Dependencies installed successfully"
    else
        log_info "Skipping dependency installation (no requirements file found)"
    fi
}

check_accelerate_installation() {
    if ! command -v accelerate &> /dev/null; then
        log_error "accelerate command not found. Please install it with: uv pip install --system accelerate"
        exit 1
    fi
    log_info "Accelerate version: $(accelerate --version)"
}

launch_training() {
    log_info "Starting distributed training with:"
    log_info "  - Number of processes: $NUM_GPUS"
    log_info "  - Config file: $CONFIG_PATH"
    log_info "  - Accelerate config: $ACCELERATE_CONFIG"
    log_info "  - Training script: $TRAINING_SCRIPT"
    
    if accelerate launch \
        --config_file "$ACCELERATE_CONFIG" \
        --num_processes "$NUM_GPUS" \
        "$TRAINING_SCRIPT" \
        --config "$CONFIG_PATH"; then
        log_success "Training completed successfully!"
    else
        local exit_code=$?
        log_error "Training failed with exit code: $exit_code"
        exit $exit_code
    fi
}

main() {
    log_info "Starting SageMaker Accelerate Training Script"
    log_info "Script directory: $SCRIPT_DIR"
    
    parse_arguments "$@"
    validate_inputs
    check_accelerate_installation
    install_dependencies
    launch_training
    log_success "Script execution completed"
}

main "$@"
