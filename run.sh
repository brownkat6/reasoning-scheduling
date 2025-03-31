#!/bin/bash
#SBATCH --time=1:00:00             # Time limit
#SBATCH --partition=gpu_requeue
#SBATCH --output=logs/run_%A_%a.out   # Standard output and error log
#SBATCH --error=logs/run_%A_%a.err    # Standard error file
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --gres=gpu:2               # Request 1 GPUs
#SBATCH --constraint='h100'        # Request H100 GPUs
#SBATCH --mem=128G                 # Memory per node
#SBATCH --cpus-per-task=4          # Number of CPU cores per task

# Get the directory containing this script
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# Create output directory if it doesn't exist
OUTPUT_DIR="${SCRIPT_DIR}/../benchmark-output"
mkdir -p "${OUTPUT_DIR}"

# Set default value for end
END=${1:-100}

# Run the token deprivation experiment
set -x
/n/netscratch/dwork_lab/Lab/katrina/envs/reasoning/bin/python -u "Dynasor/benchmark/TokenDeprivation/run.py" \
    --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
    --dataset "gsm8k" \
    --step 32 \
    --max-tokens 256 \
    --start 0 \
    --end $END \
    --probe-tokens 32 \
    --split train \
    --probe "... Oh, I suddenly got the answer to the whole problem, **Final Answer**\n\n\\[ \\boxed{" \
    "$@"
set +x