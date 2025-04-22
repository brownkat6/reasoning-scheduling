#!/bin/bash
#SBATCH --time=4:00:00             # Time limit (increased to account for all runs)
#SBATCH --partition=gpu_requeue
#SBATCH --output=logs/vis_adaptive_%A_%a.out   # Standard output and error log
#SBATCH --error=logs/vis_adaptive_%A_%a.err    # Standard error file
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --gres=gpu:2               # Request 2 GPUs
#SBATCH --constraint='h100'        # Request H100 GPUs
#SBATCH --mem=128G                 # Memory per node
#SBATCH --cpus-per-task=4          # Number of CPU cores per task

# Check if required arguments are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <mlp_model_path> <layer_index> [end_index]"
    echo "Example: $0 models/mlp_gsm8k_train_layer_16_arch_256_act_relu_drop_0.00.pt 16 100"
    exit 1
fi

MLP_MODEL="$1"
LAYER_IDX="$2"
END="${3:-100}"  # Default to 100 if not provided

# Extract dataset and split from MLP model path
# Example path: models/mlp_gsm8k_train_layer_16_arch_256_act_relu_drop_0.00.pt
DATASET=$(echo $MLP_MODEL | grep -o 'mlp_\([^_]*\)' | cut -d'_' -f2)
SPLIT=$(echo $MLP_MODEL | grep -o '_\([^_]*\)_layer' | cut -d'_' -f2)

echo "Running experiments with:"
echo "Dataset: $DATASET"
echo "Split: $SPLIT"
echo "MLP Model: $MLP_MODEL"
echo "Layer Index: $LAYER_IDX"
echo "End Index: $END"

# Create a logs directory for individual run outputs
mkdir -p logs/parallel_runs

# Function to run a command and log its output
run_with_log() {
    local log_prefix=$1
    shift  # Remove the first argument, leaving the command
    echo "Starting $log_prefix..."
    "$@" > "logs/parallel_runs/${log_prefix}_${END}.log" 2>&1 &
    echo "PID $! - Running $log_prefix"
}

# Run all three commands in parallel
echo "Launching all runs in parallel..."

# Run non-adaptive baseline
run_with_log "nonadaptive" /n/netscratch/dwork_lab/Lab/katrina/envs/reasoning/bin/python -u "Dynasor/benchmark/TokenDeprivation/run.py" \
    --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
    --dataset "$DATASET" \
    --step 32 \
    --max-tokens 256 \
    --start 0 \
    --end $END \
    --probe-tokens 32 \
    --split $SPLIT \
    --probe "... Oh, I suddenly got the answer to the whole problem, **Final Answer**\n\n\\[ \\boxed{"

# Run adaptive with MLP
run_with_log "adaptive" /n/netscratch/dwork_lab/Lab/katrina/envs/reasoning/bin/python -u "Dynasor/benchmark/TokenDeprivation/run_adaptive.py" \
    --dataset "$DATASET" \
    --mlp_train_dataset "$DATASET" \
    --mlp_train_split "$SPLIT" \
    --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
    --mlp_model "$MLP_MODEL" \
    --hidden_layer "$LAYER_IDX" \
    --max-tokens 256 \
    --step 32 \
    --start 0 \
    --end $END \
    --probe "... Oh, I suddenly got the answer to the whole problem, **Final Answer**\n\n\\[ \\boxed{"

# Run adaptive with oracle
run_with_log "oracle" /n/netscratch/dwork_lab/Lab/katrina/envs/reasoning/bin/python -u "Dynasor/benchmark/TokenDeprivation/run_adaptive.py" \
    --dataset "$DATASET" \
    --mlp_train_dataset "$DATASET" \
    --mlp_train_split "$SPLIT" \
    --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
    --mlp_model "$MLP_MODEL" \
    --hidden_layer "$LAYER_IDX" \
    --max-tokens 256 \
    --step 32 \
    --start 0 \
    --end $END \
    --use-oracle \
    --probe "... Oh, I suddenly got the answer to the whole problem, **Final Answer**\n\n\\[ \\boxed{"

# Wait for all background processes to complete
echo "Waiting for all runs to complete..."
wait
echo "All runs completed!"

# Extract model name and format paths
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_NAME_FORMATTED=$(echo $MODEL_NAME | sed 's/\//-/g')

# Construct directory paths based on the naming convention
ADAPTIVE_DIR="results/adaptive_${MODEL_NAME_FORMATTED}_${DATASET}_mlp${DATASET}_${SPLIT}_layer_${LAYER_IDX}_0_${END}"
NONADAPTIVE_DIR="results/${MODEL_NAME_FORMATTED}_${DATASET}_step32_max256_trials10_0_${END}"
ORACLE_DIR="results/oracle_${MODEL_NAME_FORMATTED}_${DATASET}_mlp${DATASET}_${SPLIT}_layer_${LAYER_IDX}_0_${END}"

echo -e "\nTo visualize results, run:"
echo "/n/netscratch/dwork_lab/Lab/katrina/envs/reasoning/bin/python -u vis_adaptive.py \\"
echo "    --dataset $DATASET \\"
echo "    --model $MODEL_NAME \\"
echo "    --split $SPLIT \\"
echo "    --start 0 \\"
echo "    --end $END"

echo -e "\nResults will be found in:"
echo "Adaptive: $ADAPTIVE_DIR"
echo "Non-adaptive: $NONADAPTIVE_DIR"
echo "Oracle: $ORACLE_DIR"

echo -e "\nCheck logs/parallel_runs/ for detailed output from each run"
