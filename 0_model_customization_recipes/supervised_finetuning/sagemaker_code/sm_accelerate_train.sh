#!/bin/bash
#
# SageMaker Accelerate Training Script
# - Fine-tunes a model using Accelerate + (optionally) DeepSpeed Zero3
# - Optionally runs local inference and an evaluation harness
#
# Usage:
#   ./sm_accelerate_train.sh --config <CONFIG_YAML> [--num_process <N>] [--run-eval]
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
NUM_GPUS=""                # per machine (input); we’ll also compute totals
CONFIG_PATH=""
RUN_EVAL=false

# Repo-local assets (absolute)
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements.txt"
ACCELERATE_CONFIG="${SCRIPT_DIR}/configs/accelerate/ds_zero3.yaml"
TRAINING_SCRIPT="${SCRIPT_DIR}/sft.py"
INFERENCE_SCRIPT="${SCRIPT_DIR}/inference.py"
EVAL_HARNESS_DIR="${SCRIPT_DIR}/evaluation_harness"
MERGE_SCRIPT="${SCRIPT_DIR}/utils/merge_adapter_weights.py"

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
Usage: $SCRIPT_NAME --config <CONFIG_YAML> [--num_process <N>] [--run-eval]

Arguments:
  --config CONFIG_YAML    Path to training configuration YAML file

Options:
  --num_process N         Per-machine process count (usually = GPUs per node)
  --run-eval              Run local inference + evaluation harness after training
  --help, -h              Show this help message

Examples:
  $SCRIPT_NAME --config ${SCRIPT_DIR}/recipes/llama_sft.yaml
  $SCRIPT_NAME --config ${SCRIPT_DIR}/configs/custom.yaml --num_process 4 --run-eval
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
            --num_process) NUM_GPUS="${2:-}"; shift 2 ;;
            --config)      CONFIG_PATH="${2:-}"; shift 2 ;;
            --run-eval)   RUN_EVAL=true; shift ;;
            --help|-h)     show_usage; exit 0 ;;
            *)             log_error "Unknown argument: $1"; show_usage; exit 1 ;;
        esac
    done
}

############################################
# GPU discovery (safe with set -u)
############################################
resolve_num_gpus() {
    if [[ -n "${NUM_GPUS:-}" ]]; then
        NUM_GPUS="$(printf '%s' "$NUM_GPUS" | tr -d '[:space:]')"   # trim whitespace
        return
    fi
    if [[ -n "${SM_NUM_GPUS:-}" ]]; then
        NUM_GPUS="$(printf '%s' "$SM_NUM_GPUS" | tr -d '[:space:]')" # trim whitespace
        return
    fi
    if command -v nvidia-smi &> /dev/null; then
        NUM_GPUS="$(nvidia-smi -L | wc -l | tr -d '[:space:]')"      # trim whitespace
        [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]] && return
    fi
    log_error "Unable to determine GPU count. Please specify --num_process."
    exit 1
}

############################################
# Input validation (includes eval assets)
############################################
validate_inputs() {
    [[ -z "$CONFIG_PATH" ]] && { log_error "--config is required"; show_usage; exit 1; }

    validate_file_exists "$CONFIG_PATH" "Configuration file"
    validate_file_exists "$TRAINING_SCRIPT" "Training script"
    validate_file_exists "$ACCELERATE_CONFIG" "Accelerate configuration"

    resolve_num_gpus
    validate_positive_integer "$NUM_GPUS" "GPU count"

    if [[ "$RUN_EVAL" == true ]]; then
        validate_file_exists "$INFERENCE_SCRIPT" "Inference script"
        [[ -d "$EVAL_HARNESS_DIR" ]] || { log_error "Evaluation harness dir not found: $EVAL_HARNESS_DIR"; exit 1; }
    fi
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

    # yq is required for dynamic accelerate backend regardless of eval
    if ! command -v yq &> /dev/null; then
        uv pip install --system yq || { log_error "Failed to install yq"; exit 1; }
    fi

    if [[ "$RUN_EVAL" == true ]]; then
        if ! command -v poetry &> /dev/null; then
            uv pip install --system poetry || { log_error "Failed to install poetry"; exit 1; }
        fi
    fi
}

############################################
# Accelerate presence (binary or importable)
############################################
check_accelerate_installation() {
    if command -v accelerate &> /dev/null; then
        return
    fi
    # Some images expose accelerate only as a Python module
    python3 - <<'PY' || { echo >&2 "[ERROR] accelerate not found"; exit 1; }
import importlib, sys
try:
    importlib.import_module("accelerate")
except Exception:
    sys.exit(1)
PY
}

############################################
# Accelerate config sanity (warning-only)
############################################
verify_accelerate_config() {
    if grep -q "rdzv_backend: static" "$ACCELERATE_CONFIG"; then
        log_warning "Found 'rdzv_backend: static' in $ACCELERATE_CONFIG. Prefer 'c10d' for SageMaker multi-node."
    elif grep -q "rdzv_backend: c10d" "$ACCELERATE_CONFIG"; then
        log_info "Accelerate config uses rdzv_backend: c10d"
    else
        log_warning "Could not verify rdzv_backend in $ACCELERATE_CONFIG (CLI overrides will still work)."
    fi
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

    # --- NEW: compute per-node and total procs, export Elastic envs ---
    PER_NODE_PROCS="${NUM_GPUS}"
    TOTAL_PROCS=$(( NUM_MACHINES * PER_NODE_PROCS ))
    export MACHINE_RANK MASTER_ADDR MASTER_PORT NUM_MACHINES PER_NODE_PROCS TOTAL_PROCS
    export LOCAL_WORLD_SIZE="$PER_NODE_PROCS"
    export WORLD_SIZE="$TOTAL_PROCS"
    export NODE_RANK="$MACHINE_RANK"
    # ------------------------------------------------------------------

    log_info "Distributed setup:"
    log_info "  - Num machines: ${NUM_MACHINES}"
    log_info "  - Per-node procs (GPUs): ${PER_NODE_PROCS}"
    log_info "  - Total processes: ${TOTAL_PROCS}"
    log_info "  - Machine rank: ${MACHINE_RANK}"
    log_info "  - Master addr: ${MASTER_ADDR}"
    log_info "  - Master port: ${MASTER_PORT}"

    log_info "SageMaker Environment Variables:"
    log_info "  - SM_HOSTS: ${SM_HOSTS:-NOT SET}"
    log_info "  - SM_CURRENT_HOST: ${SM_CURRENT_HOST:-NOT SET}"
    log_info "  - SM_NUM_GPUS: ${SM_NUM_GPUS:-NOT SET}"
    log_info "  - SM_NUM_CPUS: ${SM_NUM_CPUS:-NOT SET}"
}

############################################
# Training launch
############################################
launch_training() {
    log_info "Starting distributed training:"
    log_info "  - Config file: $CONFIG_PATH"
    log_info "  - Accelerate config: $ACCELERATE_CONFIG"
    log_info "  - Training script: $TRAINING_SCRIPT"
    log_info "  - Machine rank: $MACHINE_RANK"
    log_info "  - Per-machine processes (GPUs): $PER_NODE_PROCS"
    log_info "  - Num machines: $NUM_MACHINES"
    log_info "  - Total processes: $TOTAL_PROCS"

    # minimal: pass TOTAL_PROCS (matches 'questionnaire' semantics) and keep topology flags
    if accelerate launch \
        --config_file "$ACCELERATE_CONFIG" \
        --num_machines "$NUM_MACHINES" \
        --machine_rank "$MACHINE_RANK" \
        --num_processes "$TOTAL_PROCS" \
        --main_process_ip "$MASTER_ADDR" \
        --main_process_port "$MASTER_PORT" \
        "$TRAINING_SCRIPT" \
        --config "$CONFIG_PATH"
    then
        log_success "Training completed successfully!"
    else
        local exit_code=$?
        log_error "Training failed with exit code: $exit_code"
        exit "$exit_code"
    fi
}

_scrub_dist_env_for_inference() {
    log_info "Resetting env for single-process inference on leader node"
    # Common torch/elastic/accelerate vars that can confuse inference frameworks
    for v in MASTER_ADDR MASTER_PORT RANK LOCAL_RANK NODE_RANK WORLD_SIZE LOCAL_WORLD_SIZE \
             ACCELERATE_USE_DEEPSPEED ACCELERATE_TORCH_DEVICE ACCELERATE_MIXED_PRECISION; do
        unset "$v" 2>/dev/null || true
    done
    # Remove any other ACCELERATE_* vars if present
    while IFS='=' read -r k _; do unset "$k" 2>/dev/null || true; done < <(env | awk -F= '/^ACCELERATE_/ {print $1"="}')
}

############################################
# Local inference (preps artifacts for eval)
############################################
run_inference() {
    log_info "Scrubbing distributed env vars for inference"
    _scrub_dist_env_for_inference
    log_info "Kick-starting local inference to prepare artifacts for evaluation harness"
    python3 "$INFERENCE_SCRIPT" --config "$CONFIG_PATH"
    log_success "Inference completed!"
}

############################################
# Evaluation (robust output_dir join)
############################################
run_evaluation() {
    # unset MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING if it was set in the first place
    export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=false
    log_info "unset MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING irrespective of initial configuration"
    cd "$EVAL_HARNESS_DIR"
    poetry install
    local outdir
    outdir="$(yq -r '.output_dir' "../$CONFIG_PATH")"
    outdir="${outdir%\"}"; outdir="${outdir#\"}"; outdir="${outdir%" "}"
    outdir="${outdir#" "}"
    [[ -z "$outdir" ]] && { log_error "output_dir missing in $CONFIG_PATH"; exit 1; }
    local eval_config_path="${outdir%/}/eval_config.yaml"
    log_info "Running eval harness with config: $eval_config_path"
    poetry run evalharness --config "$eval_config_path"
    cd "$SCRIPT_DIR"
    log_success "Evaluation completed!"
}


############################################
# Adapter merge with base helper
############################################
get_model_dir_from_config() {
    local cfg="$1"
    local base="${SM_MODEL_DIR:-/opt/ml/model}"
    local model_name

    # Read model_name_or_path from TRL/Transformers model config in YAML
    model_name="$(yq -r '.model_name_or_path // ""' "$cfg")"

    # Strip quotes / whitespace
    model_name="${model_name%\"}"
    model_name="${model_name#\"}"
    model_name="${model_name#"${model_name%%[![:space:]]*}"}"
    model_name="${model_name%"${model_name##*[![:space:]]}"}"

    if [[ -z "$model_name" || "$model_name" == "null" ]]; then
        log_warning "model_name_or_path not found in $cfg; cannot infer model dir for merge."
        return 1
    fi

    # This preserves HF-style paths like deepseek-ai/DeepSeek-R1-...
    # Final result: /opt/ml/model/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
    printf '%s/%s\n' "${base%/}" "$model_name"
}

maybe_merge_peft_adapter() {
    # Only run on leader node
    if [[ "${MACHINE_RANK:-0}" != "0" ]]; then
        log_info "Skipping PEFT merge on non-leader node (MACHINE_RANK=${MACHINE_RANK})."
        return 0
    fi

    # Check if this run is actually PEFT
    local use_peft
    use_peft="$(yq -r '.use_peft // false' "$CONFIG_PATH")"
    if [[ "$use_peft" != "true" && "$use_peft" != "True" ]]; then
        log_info "use_peft is not true in $CONFIG_PATH; skipping PEFT merge."
        return 0
    fi

    if [[ ! -f "$MERGE_SCRIPT" ]]; then
        log_error "Merge script not found: $MERGE_SCRIPT"
        return 1
    fi

    # Derive model_dir from config + SM_MODEL_DIR
    local model_dir
    if ! model_dir="$(get_model_dir_from_config "$CONFIG_PATH")"; then
        log_warning "Unable to determine model_dir from config; skipping PEFT merge."
        return 0
    fi

    local peft_dir="${model_dir%/}/peft_adapter"

    if [[ ! -d "$peft_dir" ]]; then
        log_info "No PEFT adapter directory found at $peft_dir; nothing to merge."
        return 0
    fi

    # Sanity check that this really looks like a PEFT adapter
    if ! compgen -G "${peft_dir}/*adapter_model*.bin" > /dev/null && \
       ! compgen -G "${peft_dir}/adapter_config.json" > /dev/null; then
        log_warning "Directory $peft_dir exists but doesn't look like a PEFT adapter; skipping merge."
        return 0
    fi

    log_info "Merging PEFT adapter from:"
    log_info "  - peft_dir:  $peft_dir"
    log_info "  - model_dir: $model_dir"

    # Call your merge_and_unload.py script
    if python3 "$MERGE_SCRIPT" \
        --peft_model_id "$peft_dir" \
        --output_dir "$model_dir" \
        --save_tokenizer True; then
        log_success "PEFT adapter merged and full model saved to: $model_dir"
    else
        local ec=$?
        log_error "PEFT merge_and_unload script failed with exit code: $ec"
        return "$ec"
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
    set_dynamic_rdzv_backend

    # print the deepspeed configuration
    log_info "******************* Start of DeepSpeed Configuration *******************"
    more "$ACCELERATE_CONFIG"
    log_info "******************** End of DeepSpeed Configuration ********************"

    check_accelerate_installation
    verify_accelerate_config
    launch_training

    # In main(), after launch_training
    if [[ "$RUN_EVAL" != true ]]; then
        log_warning "Skipping inference and evaluation (--run-eval not set)."
        # Merge PEFT adapter into full model if applicable
        log_warning "Running merge and unload of peft with main."
        maybe_merge_peft_adapter
    else
        if [[ "${MACHINE_RANK:-0}" != "0" ]]; then
            log_info "RUN_EVAL enabled but this is machine rank ${MACHINE_RANK}. Skipping eval on non-leader node."
        else
            log_warning "Running in-container evaluation (leader node)."
            run_inference
            run_evaluation
        fi
    fi

    log_success "All steps completed successfully"
}

main "$@"
