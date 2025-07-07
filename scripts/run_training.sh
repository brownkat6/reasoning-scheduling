#!/bin/bash
"""
training script for Predictive Scheduling framework.

This script provides a unified interface for training MLP and LoRA models
with proper environment detection, configuration management, and SLURM integration.

Usage:
    ./scripts/run_training.sh mlp --dataset gsm8k --hidden-layer 16
    ./scripts/run_training.sh lora --task difficulty --use-wandb
    ./scripts/run_training.sh --help
"""

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEFAULT_CONFIG="$PROJECT_ROOT/config.yaml"

# Default parameters
TASK_TYPE=""
PYTHON_CMD=""
SLURM_PARTITION=""
SLURM_ACCOUNT=""
USE_SLURM=false
DRY_RUN=false
VERBOSE=false

# Logging functions
log_info() {
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $*"
}

log_error() {
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - $*" >&2
}

log_debug() {
    if [[ "$VERBOSE" == "true" ]]; then
        echo "[DEBUG] $(date '+%Y-%m-%d %H:%M:%S') - $*"
    fi
}

# Help function
show_help() {
    cat << EOF
Professional Training Script for Predictive Scheduling Framework

USAGE:
    $0 <task_type> [OPTIONS]

TASK TYPES:
    mlp          Train MLP predictors for early stopping
    lora         Train LoRA fine-tuned models
    dynasor      Run Dynasor token deprivation experiments
    
COMMON OPTIONS:
    --config FILE           Configuration file (default: config.yaml)
    --dataset NAME          Dataset to use (gsm8k, math500, etc.)
    --output-dir DIR        Output directory for results
    --use-wandb             Enable Weights & Biases logging
    --verbose               Enable verbose logging
    --dry-run               Show commands without executing
    --help                  Show this help message

ENVIRONMENT OPTIONS:
    --python PATH           Python executable path (auto-detected if not provided)
    --slurm                 Submit as SLURM job
    --partition NAME        SLURM partition (required with --slurm)
    --account NAME          SLURM account (required with --slurm)
    --time DURATION         SLURM time limit (default: 2:00:00)
    --mem SIZE              SLURM memory limit (default: 64G)
    --gpus NUM              Number of GPUs (default: 1)

MLP-SPECIFIC OPTIONS:
    --hidden-layer NUM      Hidden layer to use (default: 16)
    --hidden-dims DIMS      Hidden layer dimensions (default: 256)
    --activation FUNC       Activation function (default: relu)
    --num-epochs NUM        Number of epochs (default: 20)
    --batch-size NUM        Batch size (default: 32)

LORA-SPECIFIC OPTIONS:
    --task TYPE             LoRA task (early_stopping, difficulty)
    --model-name NAME       Base model name
    --lora-rank NUM         LoRA rank (default: 16)
    --lora-alpha NUM        LoRA alpha (default: 32)

EXAMPLES:
    # Train MLP predictor locally
    $0 mlp --dataset gsm8k --hidden-layer 16 --use-wandb
    
    # Train LoRA model on SLURM cluster
    $0 lora --task difficulty --slurm --partition gpu --account mylab
    
    # Run Dynasor experiment with custom configuration
    $0 dynasor --config experiments/dynasor_config.yaml --dry-run

EOF
}

# Environment detection
detect_environment() {
    log_debug "Detecting computing environment"
    
    # Check if we're on a SLURM cluster
    if command -v sbatch >/dev/null 2>&1; then
        log_debug "SLURM detected"
        # Try to detect common cluster configurations
        if [[ -d "/n/netscratch" ]]; then
            log_debug "Harvard cluster environment detected"
            # Set Harvard-specific defaults
            SLURM_PARTITION="${SLURM_PARTITION:-gpu_requeue}"
            if [[ "$USER" == "katrinabrown" ]]; then
                SLURM_ACCOUNT="${SLURM_ACCOUNT:-kempner_dwork_lab}"
            elif [[ "$USER" == "amuppidi" ]]; then
                SLURM_ACCOUNT="${SLURM_ACCOUNT:-kempner_gershman_lab}"
            fi
        fi
    fi
    
    # Auto-detect Python executable
    if [[ -z "$PYTHON_CMD" ]]; then
        # Try conda environment first
        if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
            PYTHON_CMD="python"
            log_debug "Using conda environment: $CONDA_DEFAULT_ENV"
        # Try virtual environment
        elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
            PYTHON_CMD="python"
            log_debug "Using virtual environment: $VIRTUAL_ENV"
        # Fall back to system python
        else
            PYTHON_CMD="python3"
            log_debug "Using system Python"
        fi
    fi
    
    # Verify Python installation
    if ! command -v "$PYTHON_CMD" >/dev/null 2>&1; then
        log_error "Python executable not found: $PYTHON_CMD"
        log_error "Please install Python or specify --python PATH"
        exit 1
    fi
    
    log_debug "Using Python: $PYTHON_CMD ($($PYTHON_CMD --version 2>&1))"
}

# Create SLURM job script
create_slurm_script() {
    local job_name="$1"
    local python_script="$2"
    shift 2
    local script_args="$*"
    
    local slurm_script="$PROJECT_ROOT/jobs/${job_name}_$(date +%Y%m%d_%H%M%S).sh"
    mkdir -p "$(dirname "$slurm_script")"
    
    cat > "$slurm_script" << EOF
#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --time=${SLURM_TIME:-2:00:00}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:${SLURM_GPUS:-1}
#SBATCH --mem=${SLURM_MEM:-64G}
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/${job_name}_%A_%a.out
#SBATCH --error=logs/${job_name}_%A_%a.err

# Set up environment
set -euo pipefail
cd "$PROJECT_ROOT"

# Create logs directory
mkdir -p logs

# Print job information
echo "Job started at: \$(date)"
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURMD_NODENAME"
echo "Working directory: \$(pwd)"

# Print GPU information if available
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU information:"
    nvidia-smi
fi

# Run the training script
echo "Running: $PYTHON_CMD $python_script $script_args"
$PYTHON_CMD -u "$python_script" $script_args

echo "Job completed at: \$(date)"
EOF
    
    echo "$slurm_script"
}

# Parse command line arguments
parse_args() {
    if [[ $# -eq 0 ]]; then
        show_help
        exit 1
    fi
    
    TASK_TYPE="$1"
    shift
    
    if [[ "$TASK_TYPE" == "--help" || "$TASK_TYPE" == "-h" ]]; then
        show_help
        exit 0
    fi
    
    if [[ "$TASK_TYPE" != "mlp" && "$TASK_TYPE" != "lora" && "$TASK_TYPE" != "dynasor" ]]; then
        log_error "Invalid task type: $TASK_TYPE"
        log_error "Valid options: mlp, lora, dynasor"
        exit 1
    fi
    
    # Parse remaining arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help|-h)
                show_help
                exit 0
                ;;
            --verbose)
                VERBOSE=true
                ;;
            --dry-run)
                DRY_RUN=true
                ;;
            --python)
                PYTHON_CMD="$2"
                shift 2
                ;;
            --slurm)
                USE_SLURM=true
                shift
                ;;
            --partition)
                SLURM_PARTITION="$2"
                shift 2
                ;;
            --account)
                SLURM_ACCOUNT="$2"
                shift 2
                ;;
            --time)
                SLURM_TIME="$2"
                shift 2
                ;;
            --mem)
                SLURM_MEM="$2"
                shift 2
                ;;
            --gpus)
                SLURM_GPUS="$2"
                shift 2
                ;;
            *)
                # Pass remaining arguments to the training script
                break
                ;;
        esac
    done
    
    # Store remaining arguments for the training script
    SCRIPT_ARGS="$*"
}

# Main execution function
main() {
    log_info "Starting Predictive Scheduling training workflow"
    log_info "Task type: $TASK_TYPE"
    
    # Detect environment and setup
    detect_environment
    
    # Validate SLURM configuration if needed
    if [[ "$USE_SLURM" == "true" ]]; then
        if [[ -z "$SLURM_PARTITION" ]]; then
            log_error "SLURM partition must be specified with --partition when using --slurm"
            exit 1
        fi
        if [[ -z "$SLURM_ACCOUNT" ]]; then
            log_error "SLURM account must be specified with --account when using --slurm"
            exit 1
        fi
        log_info "SLURM configuration: partition=$SLURM_PARTITION, account=$SLURM_ACCOUNT"
    fi
    
    # Determine the Python script to run
    case "$TASK_TYPE" in
        "mlp")
            PYTHON_SCRIPT="$SCRIPT_DIR/train_mlp.py"
            JOB_NAME="mlp_training"
            ;;
        "lora")
            PYTHON_SCRIPT="$SCRIPT_DIR/train_lora.py"
            JOB_NAME="lora_training"
            ;;
        "dynasor")
            PYTHON_SCRIPT="$PROJECT_ROOT/Dynasor/benchmark/TokenDeprivation/run.py"
            JOB_NAME="dynasor_experiment"
            ;;
        *)
            log_error "Unknown task type: $TASK_TYPE"
            exit 1
            ;;
    esac
    
    # Verify script exists
    if [[ ! -f "$PYTHON_SCRIPT" ]]; then
        log_error "Training script not found: $PYTHON_SCRIPT"
        exit 1
    fi
    
    # Create full command
    FULL_COMMAND="$PYTHON_CMD $PYTHON_SCRIPT $SCRIPT_ARGS"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN - Would execute:"
        echo "$FULL_COMMAND"
        if [[ "$USE_SLURM" == "true" ]]; then
            echo "As SLURM job with partition=$SLURM_PARTITION, account=$SLURM_ACCOUNT"
        fi
        exit 0
    fi
    
    # Execute the command
    if [[ "$USE_SLURM" == "true" ]]; then
        log_info "Submitting SLURM job"
        SLURM_SCRIPT=$(create_slurm_script "$JOB_NAME" "$PYTHON_SCRIPT" $SCRIPT_ARGS)
        log_info "Created SLURM script: $SLURM_SCRIPT"
        
        JOB_ID=$(sbatch "$SLURM_SCRIPT" | grep -o '[0-9]*')
        log_info "Submitted SLURM job: $JOB_ID"
        log_info "Monitor with: squeue -j $JOB_ID"
        log_info "View logs with: tail -f logs/${JOB_NAME}_${JOB_ID}.out"
    else
        log_info "Running locally"
        log_debug "Command: $FULL_COMMAND"
        cd "$PROJECT_ROOT"
        exec $FULL_COMMAND
    fi
}

# Parse arguments and run
parse_args "$@"
main