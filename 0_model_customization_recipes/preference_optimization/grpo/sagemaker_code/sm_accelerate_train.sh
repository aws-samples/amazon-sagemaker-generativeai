#!/bin/bash

# SageMaker Accelerate Training Script
# Launches distributed fine-tuning using Accelerate + DeepSpeed Zero3
#
# Usage: ./sm_accelerate_train.sh --num_process <NUM_GPUS> --config <CONFIG_YAML>
# Example: ./sm_accelerate_train.sh --num_process 8 --config recipes/llama_sft.yaml

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Script configuration
readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
NUM_GPUS=""
CONFIG_PATH=""
REQUIREMENTS_FILE="./requirements.txt"
ACCELERATE_CONFIG="configs/accelerate/ds_zero3.yaml"
TRAINING_SCRIPT="hf_recipes/grpo.py"

# Color codes for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

show_usage() {
    cat << EOF
Usage: $SCRIPT_NAME --num_process <NUM_GPUS> --config <CONFIG_YAML>

Arguments:
    --num_process NUM_GPUS    Number of GPU processes for distributed training
    --config CONFIG_YAML      Path to training configuration YAML file

Options:
    --help                    Show this help message

Examples:
    $SCRIPT_NAME --num_process 8 --config recipes/llama_sft.yaml
    $SCRIPT_NAME --num_process 4 --config configs/custom_config.yaml

Environment Variables:
    REQUIREMENTS_FILE         Path to requirements file (default: ./requirements.txt)
    ACCELERATE_CONFIG         Path to accelerate config (default: accelerate/zero3.yaml)
    TRAINING_SCRIPT          Path to training script (default: run_sft.py)
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
                if [[ -z "${2:-}" ]]; then
                    log_error "--num_process requires a value"
                    show_usage
                    exit 1
                fi
                NUM_GPUS="$2"
                shift 2
                ;;
            --config)
                if [[ -z "${2:-}" ]]; then
                    log_error "--config requires a value"
                    show_usage
                    exit 1
                fi
                CONFIG_PATH="$2"
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

validate_inputs() {
    local validation_failed=false
    
    # Check required arguments
    if [[ -z "$NUM_GPUS" ]]; then
        log_error "--num_process is required"
        validation_failed=true
    fi
    
    if [[ -z "$CONFIG_PATH" ]]; then
        log_error "--config is required"
        validation_failed=true
    fi
    
    if [[ "$validation_failed" == true ]]; then
        show_usage
        exit 1
    fi
    
    # Validate NUM_GPUS is a positive integer
    validate_positive_integer "$NUM_GPUS" "--num_process" || exit 1
    
    # Validate files exist
    validate_file_exists "$CONFIG_PATH" "Configuration file" || exit 1
    validate_file_exists "$TRAINING_SCRIPT" "Training script" || exit 1
    validate_file_exists "$ACCELERATE_CONFIG" "Accelerate configuration" || exit 1
    
    # Check if requirements file exists (optional)
    if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
        log_warning "Requirements file not found: $REQUIREMENTS_FILE (skipping dependency installation)"
    fi
}

install_dependencies() {
    if [[ -f "$REQUIREMENTS_FILE" ]]; then
        log_info "Installing Python dependencies from $REQUIREMENTS_FILE..."
        # pip install uv
        pip install "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1"  "setuptools<71.0.0"  --index-url https://download.pytorch.org/whl/cu121 
        pip install wheel packaging
        pip install --no-build-isolation flash-attn==2.7.2.post1
        pip install  --upgrade \
            "transformers==4.48.1" \
            "datasets==4.1.0" \
            "accelerate==1.10.0" \
            "hf-transfer==0.1.9" \
            "deepspeed==0.15.4" \
            "trl==0.14.0" \
            tensorboardX \
            wandb
        pip install "vllm==0.7.0"
        # if ! uv pip install -r "$REQUIREMENTS_FILE"; then
        #     log_error "Failed to install dependencies from $REQUIREMENTS_FILE"
        #     exit 1
        # fi
        
        log_success "Dependencies installed successfully"
    else
        log_info "Skipping dependency installation (no requirements file found)"
    fi
}

check_accelerate_installation() {
    if ! command -v accelerate &> /dev/null; then
        log_error "accelerate command not found. Please install it with: pip install accelerate"
        exit 1
    fi
    
    log_info "Accelerate version: $(accelerate --version)"
}

launch_training() {
    log_info "Starting distributed training with the following configuration:"
    log_info "  - Number of processes: $NUM_GPUS"
    log_info "  - Config file: $CONFIG_PATH"
    log_info "  - Accelerate config: $ACCELERATE_CONFIG"
    log_info "  - Training script: $TRAINING_SCRIPT"
    
    # Launch training with error handling
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
    
    # Parse command line arguments
    parse_arguments "$@"
    
    # Validate all inputs
    validate_inputs
    
    # Check accelerate installation
    # check_accelerate_installation
    
    # Install dependencies
    install_dependencies
    
    echo "$HF_TOKEN"
    # Launch training
    launch_training
    
    log_success "Script execution completed"
}

# Run main function with all arguments
main "$@"