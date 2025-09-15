#!/bin/bash

# SageMaker Accelerate Training Script
# Launches distributed fine-tuning using Accelerate + DeepSpeed Zero3
#
# Usage: ./sm_accelerate_train.sh --entrypoint <TRAINING_SCRIPT> --accelerate_config <ACCELERATE_CONFIG> --config <CONFIG_YAML>
# Example: ./sm_accelerate_train.sh --entrypoint train.py --accelerate_config distribution/zero_3.yaml --config recipes/llama_sft.yaml

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Script configuration
readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
CONFIG_PATH=""
REQUIREMENTS_FILE="./requirements.txt"
ACCELERATE_CONFIG=""
TRAINING_SCRIPT=""

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
Usage: $SCRIPT_NAME --entrypoint <ENTRYPOINT_SCRIPT> --accelerate_config <ACCELERATE_CONFIG_YAML> --config <CONFIG_YAML>

Arguments:
    --accelerate_config ACCELERATE_CONFIG_YAML   Path to accelerate config YAML file
    --config CONFIG_YAML      Path to training configuration YAML file
    --entrypoint ENTRYPOINT_SCRIPT   Path to training script

Options:
    --help                    Show this help message

Examples:
    $SCRIPT_NAME --entrypoint train.py --accelerate_config distribution/zero_3.yaml --config recipes/llama_sft.yaml
    $SCRIPT_NAME --entrypoint train.py --accelerate_config distribution/fsdp.yaml --config recipes/llama_sft.yaml

Environment Variables:
    REQUIREMENTS_FILE         Path to requirements file (default: ./requirements.txt)
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
            --accelerate_config)
                if [[ -z "${2:-}" ]]; then
                    log_error "--accelerate_config requires a value"
                    show_usage
                    exit 1
                fi
                ACCELERATE_CONFIG="$2"
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
            --entrypoint)
                if [[ -z "${2:-}" ]]; then
                    log_error "--entrypoint requires a value"
                    show_usage
                    exit 1
                fi
                TRAINING_SCRIPT="$2"
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
    if [[ -z "$ACCELERATE_CONFIG" ]]; then
        log_error "--accelerate_config is required"
        validation_failed=true
    fi
    
    if [[ -z "$CONFIG_PATH" ]]; then
        log_error "--config is required"
        validation_failed=true
    fi

    if [[ -z "$TRAINING_SCRIPT" ]]; then
        log_error "--entrypoint is required"
        validation_failed=true
    fi
    
    if [[ "$validation_failed" == true ]]; then
        show_usage
        exit 1
    fi
    
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
        
        if ! python3 -m pip install -r "$REQUIREMENTS_FILE"; then
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
        log_error "accelerate command not found. Please install it with: pip install accelerate"
        exit 1
    fi
    
    log_info "Accelerate version: $(accelerate --version)"
}

launch_training() {
    log_info "Starting distributed training with the following configuration:"
    log_info "  - Config file: $CONFIG_PATH"
    log_info "  - Accelerate config: $ACCELERATE_CONFIG"
    log_info "  - Training script: $TRAINING_SCRIPT"
    
    # Launch training with error handling
    if accelerate launch \
        --config_file "$ACCELERATE_CONFIG" \
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
    # install_dependencies
    
    # Launch training
    launch_training
    
    log_success "Script execution completed"
}

# Run main function with all arguments
main "$@"