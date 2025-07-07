#!/bin/bash
"""
Dynasor experiment runner for Predictive Scheduling framework.

This script provides a clean interface for running Dynasor token deprivation experiments
with proper configuration management and environment detection.

Usage:
    ./scripts/run_dynasor.sh --dataset gsm8k --end 100
    ./scripts/run_dynasor.sh --adaptive --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
    ./scripts/run_dynasor.sh --help
"""

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEFAULT_CONFIG="$PROJECT_ROOT/config.yaml"

# Default parameters
EXPERIMENT_TYPE="standard"
PYTHON_CMD=""
USE_SLURM=false
DRY_RUN=false
VERBOSE=false

# Experiment parameters
DATASET="gsm8k"
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MAX_TOKENS=256
STEP=32
START=0
END=100
PROBE_TOKENS=32
SPLIT="test"
TEMPERATURE=0.7
TOP_P=0.95
NUM_TRIALS=10

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
Professional Dynasor Experiment Runner

USAGE:
    $0 [OPTIONS]

EXPERIMENT TYPES:
    --standard          Run standard token deprivation experiment (default)
    --adaptive          Run adaptive allocation experiment
    --oracle            Run oracle allocation experiment

COMMON OPTIONS:
    --dataset NAME      Dataset to use (default: gsm8k)
    --model NAME        Model name (default: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
    --start NUM         Start index (default: 0)
    --end NUM           End index (default: 100)
    --split NAME        Data split (default: test)
    --output-dir DIR    Output directory (default: benchmark-output)
    --config FILE       Configuration file
    --verbose           Enable verbose logging
    --dry-run           Show commands without executing
    --help              Show this help message

GENERATION PARAMETERS:
    --max-tokens NUM    Maximum tokens per generation (default: 256)
    --step NUM          Token step size (default: 32)
    --probe-tokens NUM  Probe token count (default: 32)
    --temperature NUM   Sampling temperature (default: 0.7)
    --top-p NUM         Top-p sampling (default: 0.95)
    --num-trials NUM    Number of trials for adaptive (default: 10)

SLURM OPTIONS:
    --slurm             Submit as SLURM job
    --partition NAME    SLURM partition
    --account NAME      SLURM account
    --time DURATION     SLURM time limit (default: 2:00:00)
    --mem SIZE          SLURM memory limit (default: 128G)
    --gpus NUM          Number of GPUs (default: 2)

EXAMPLES:
    # Run standard experiment locally
    $0 --dataset gsm8k --end 50 --verbose
    
    # Run adaptive experiment on SLURM
    $0 --adaptive --slurm --partition gpu --account mylab
    
    # Run oracle experiment with custom parameters
    $0 --oracle --temperature 0.6 --num-trials 20

EOF
}

# Environment detection
detect_environment() {
    log_debug "Detecting environment"
    
    # Auto-detect Python executable
    if [[ -z "$PYTHON_CMD" ]]; then
        if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
            PYTHON_CMD="python"
            log_debug "Using conda environment: $CONDA_DEFAULT_ENV"
        elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
            PYTHON_CMD="python" 
            log_debug "Using virtual environment: $VIRTUAL_ENV"
        else
            PYTHON_CMD="python3"
            log_debug "Using system Python"
        fi
    fi
    
    # Verify Python installation
    if ! command -v "$PYTHON_CMD" >/dev/null 2>&1; then
        log_error "Python executable not found: $PYTHON_CMD"
        exit 1
    fi
    
    log_debug "Using Python: $PYTHON_CMD ($($PYTHON_CMD --version 2>&1))"
}

# Create probe string with proper escaping
get_probe_string() {
    echo "... Oh, I suddenly got the answer to the whole problem, **Final Answer**\n\n\\[ \\boxed{"
}

# Build command arguments
build_command_args() {
    local args=""
    
    # Core parameters
    args="$args --model \"$MODEL\""
    args="$args --dataset \"$DATASET\""
    args="$args --max-tokens $MAX_TOKENS"
    args="$args --step $STEP"
    args="$args --start $START"
    args="$args --end $END"
    args="$args --probe-tokens $PROBE_TOKENS"
    args="$args --split \"$SPLIT\""
    args="$args --temperature $TEMPERATURE"
    args="$args --top-p $TOP_P"
    args="$args --probe \"$(get_probe_string)\""
    
    # Adaptive-specific parameters
    if [[ "$EXPERIMENT_TYPE" == "adaptive" ]]; then
        args="$args --mlp_train_dataset \"$DATASET\""
        args="$args --mlp_train_split train"
        args="$args --num-trials $NUM_TRIALS"
    fi
    
    echo "$args"
}

# Get experiment script path
get_script_path() {
    local dynasor_dir="$PROJECT_ROOT/Dynasor/benchmark/TokenDeprivation"
    
    case "$EXPERIMENT_TYPE" in
        "standard")
            echo "$dynasor_dir/run.py"
            ;;
        "adaptive")
            echo "$dynasor_dir/run_adaptive.py"
            ;;
        "oracle")
            echo "$dynasor_dir/run_adaptive_oracle.py"
            ;;
        *)
            log_error "Unknown experiment type: $EXPERIMENT_TYPE"
            exit 1
            ;;
    esac
}

# Create SLURM job script
create_slurm_script() {
    local job_name="dynasor_${EXPERIMENT_TYPE}"
    local python_script="$1"
    local script_args="$2"
    
    local slurm_script="$PROJECT_ROOT/jobs/${job_name}_$(date +%Y%m%d_%H%M%S).sh"
    mkdir -p "$(dirname "$slurm_script")"
    
    cat > "$slurm_script" << EOF
#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH --partition=${SLURM_PARTITION:-gpu_requeue}
#SBATCH --account=${SLURM_ACCOUNT:-kempner_gershman_lab}
#SBATCH --time=${SLURM_TIME:-2:00:00}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:${SLURM_GPUS:-2}
#SBATCH --constraint=${SLURM_CONSTRAINT:-h100}
#SBATCH --mem=${SLURM_MEM:-128G}
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/${job_name}_%A_%a.out
#SBATCH --error=logs/${job_name}_%A_%a.err

# Set up environment
set -euo pipefail
cd "$PROJECT_ROOT"

# Create necessary directories
mkdir -p logs
mkdir -p benchmark-output

# Print job information
echo "Job started at: \$(date)"
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURMD_NODENAME"
echo "Working directory: \$(pwd)"

# Print GPU information
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU information:"
    nvidia-smi
fi

# Set execution flags for debugging
set -x

# Run the experiment
echo "Running: $PYTHON_CMD -u $python_script $script_args"
$PYTHON_CMD -u "$python_script" $script_args

set +x
echo "Job completed at: \$(date)"
EOF
    
    echo "$slurm_script"
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help|-h)
                show_help
                exit 0
                ;;
            --standard)
                EXPERIMENT_TYPE="standard"
                shift
                ;;
            --adaptive)
                EXPERIMENT_TYPE="adaptive"
                shift
                ;;
            --oracle)
                EXPERIMENT_TYPE="oracle"
                shift
                ;;
            --dataset)
                DATASET="$2"
                shift 2
                ;;
            --model)
                MODEL="$2"
                shift 2
                ;;
            --start)
                START="$2"
                shift 2
                ;;
            --end)
                END="$2"
                shift 2
                ;;
            --split)
                SPLIT="$2"
                shift 2
                ;;
            --max-tokens)
                MAX_TOKENS="$2"
                shift 2
                ;;
            --step)
                STEP="$2"
                shift 2
                ;;
            --probe-tokens)
                PROBE_TOKENS="$2"
                shift 2
                ;;
            --temperature)
                TEMPERATURE="$2"
                shift 2
                ;;
            --top-p)
                TOP_P="$2"
                shift 2
                ;;
            --num-trials)
                NUM_TRIALS="$2"
                shift 2
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
            --constraint)
                SLURM_CONSTRAINT="$2"
                shift 2
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Main execution function
main() {
    log_info "Starting Dynasor $EXPERIMENT_TYPE experiment"
    
    # Detect environment
    detect_environment
    
    # Get script and arguments
    PYTHON_SCRIPT=$(get_script_path)
    SCRIPT_ARGS=$(build_command_args)
    
    # Verify script exists
    if [[ ! -f "$PYTHON_SCRIPT" ]]; then
        log_error "Experiment script not found: $PYTHON_SCRIPT"
        exit 1
    fi
    
    # Create output directory
    mkdir -p "$PROJECT_ROOT/benchmark-output"
    mkdir -p "$PROJECT_ROOT/logs"
    
    # Build full command
    FULL_COMMAND="$PYTHON_CMD -u \"$PYTHON_SCRIPT\" $SCRIPT_ARGS"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN - Would execute:"
        echo "$FULL_COMMAND"
        if [[ "$USE_SLURM" == "true" ]]; then
            echo "As SLURM job with partition=${SLURM_PARTITION:-gpu_requeue}, account=${SLURM_ACCOUNT:-kempner_gershman_lab}"
        fi
        exit 0
    fi
    
    # Execute the command
    if [[ "$USE_SLURM" == "true" ]]; then
        log_info "Submitting SLURM job for $EXPERIMENT_TYPE experiment"
        SLURM_SCRIPT=$(create_slurm_script "$PYTHON_SCRIPT" "$SCRIPT_ARGS")
        log_info "Created SLURM script: $SLURM_SCRIPT"
        
        JOB_ID=$(sbatch "$SLURM_SCRIPT" | grep -o '[0-9]*')
        log_info "Submitted SLURM job: $JOB_ID"
        log_info "Monitor with: squeue -j $JOB_ID"
        log_info "View logs with: tail -f logs/dynasor_${EXPERIMENT_TYPE}_${JOB_ID}.out"
    else
        log_info "Running $EXPERIMENT_TYPE experiment locally"
        log_debug "Command: $FULL_COMMAND"
        cd "$PROJECT_ROOT"
        
        # Enable debugging if verbose
        if [[ "$VERBOSE" == "true" ]]; then
            set -x
        fi
        
        eval $FULL_COMMAND
        
        if [[ "$VERBOSE" == "true" ]]; then
            set +x
        fi
    fi
    
    log_info "Dynasor $EXPERIMENT_TYPE experiment started successfully"
}

# Parse arguments and run
parse_args "$@"
main