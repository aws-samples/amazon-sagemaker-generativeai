#!/bin/bash

# SageMaker Accelerate Training Script
# Fine-tunes model using Accelerate + DeepSpeed Zero3,
# then optionally runs inference and evaluation harness.
#
# Usage: ./sm_accelerate_train.sh --config <CONFIG_YAML> [--skip-eval]

set -euo pipefail

# ---------------- Configuration ----------------
readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NUM_GPUS=""
CONFIG_PATH=""
SKIP_EVAL=false

REQUIREMENTS_FILE="./requirements.txt"
ACCELERATE_CONFIG="configs/accelerate/ds_zero3.yaml"
TRAINING_SCRIPT="sft.py"
INFERENCE_SCRIPT="inference.py"
EVAL_HARNESS_DIR="evaluation_harness"

# ---------------- Logging ----------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1" >&2; }

# ---------------- Usage ----------------
show_usage() {
    cat << EOF
Usage: $SCRIPT_NAME --config <CONFIG_YAML> [--skip-eval]

Arguments:
    --config CONFIG_YAML      Path to training configuration YAML file

Options:
    --skip-eval               Skip inference + evaluation harness
    --help, -h                Show this help message

Examples:
    $SCRIPT_NAME --config recipes/llama_sft.yaml
    $SCRIPT_NAME --config configs/custom.yaml --skip-eval
EOF
}

# ---------------- Validation helpers ----------------
validate_file_exists() {
    [[ -f "$1" ]] || { log_error "$2 not found: $1"; exit 1; }
}
validate_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]] || { log_error "$2 must be a positive integer, got: $1"; exit 1; }
}

# ---------------- Argument parsing ----------------
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --num_process) NUM_GPUS="${2:-}"; shift 2 ;;
            --config)      CONFIG_PATH="${2:-}"; shift 2 ;;
            --skip-eval)   SKIP_EVAL=true; shift ;;
            --help|-h)     show_usage; exit 0 ;;
            *)             log_error "Unknown argument: $1"; show_usage; exit 1 ;;
        esac
    done
}

resolve_num_gpus() {
    if [[ -n "$NUM_GPUS" ]]; then return; fi
    if [[ -n "${SM_NUM_GPUS:-}" ]]; then NUM_GPUS="$SM_NUM_GPUS"; return; fi
    if command -v nvidia-smi &> /dev/null; then
        NUM_GPUS=$(nvidia-smi -L | wc -l)
        [[ "$NUM_GPUS" -gt 0 ]] && return
    fi
    log_error "Unable to determine GPU count. Please specify --num_process."; exit 1
}

validate_inputs() {
    [[ -z "$CONFIG_PATH" ]] && { log_error "--config is required"; show_usage; exit 1; }
    resolve_num_gpus
    validate_positive_integer "$NUM_GPUS" "GPU count"
    validate_file_exists "$CONFIG_PATH" "Configuration file"
    validate_file_exists "$TRAINING_SCRIPT" "Training script"
    validate_file_exists "$ACCELERATE_CONFIG" "Accelerate configuration"
}

# ---------------- Dependencies ----------------
install_dependencies() {
    if ! command -v uv &> /dev/null; then
        python3 -m pip install --upgrade uv || { log_error "Failed to install uv"; exit 1; }
    fi
    [[ -f "$REQUIREMENTS_FILE" ]] && uv pip install --system -r "$REQUIREMENTS_FILE"
}

check_accelerate_installation() {
    command -v accelerate &> /dev/null || { log_error "accelerate not found"; exit 1; }
}

# ---------------- Pipeline ----------------
launch_training() {
    accelerate launch \
        --config_file "$ACCELERATE_CONFIG" \
        --num_processes "$NUM_GPUS" \
        "$TRAINING_SCRIPT" \
        --config "$CONFIG_PATH"
    log_success "Training completed!"
}

run_inference() {
    # log_warning "Upgrading vllm >= 0.10.2"
    # uv pip install --system "vllm>=0.10.2"
    python3 "$INFERENCE_SCRIPT" --config "$CONFIG_PATH"
    log_success "Inference completed!"
}

run_evaluation() {
    cd "$EVAL_HARNESS_DIR"
    poetry install
    # Extract output_dir from recipe YAML and append eval_config.yaml
    eval_config_path="$(yq -r .output_dir "../$CONFIG_PATH")eval_config.yaml"
    log_info "Running eval harness with config: $eval_config_path"
    poetry run evalharness --config "$eval_config_path"
    cd "$SCRIPT_DIR"
    log_success "Evaluation completed!"
}

main() {
    parse_arguments "$@"
    validate_inputs
    check_accelerate_installation
    install_dependencies
    launch_training

    if [[ "$SKIP_EVAL" == true ]]; then
        log_warning "Skipping inference and evaluation as requested (--skip-eval)."
    else
        run_inference
        run_evaluation
    fi

    log_success "All steps completed successfully"
}

main "$@"
