#!/bin/bash
#SBATCH --time=1:00:00             # Time limit
#SBATCH --partition=gpu_requeue
#SBATCH --output=logs/run_adaptive_%A_%a.out   # Standard output and error log
#SBATCH --error=logs/run_adaptive_%A_%a.err    # Standard error file
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --gres=gpu:2               # Request 1 GPUs
#SBATCH --constraint='h100'        # Request H100 GPUs
#SBATCH --mem=128G                 # Memory per node
#SBATCH --cpus-per-task=4          # Number of CPU cores per task

# TODO: 3/14 - get vis_adative to work with gsm8k standardized SAME FORMAT EVERYWHERE ground truth data. 

# Get the directory containing this script
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# Create output directory if it doesn't exist
OUTPUT_DIR="${SCRIPT_DIR}/../benchmark-output"
mkdir -p "${OUTPUT_DIR}"

# Run the token deprivation experiment
set -x

# Run the adaptive token deprivation experiment
/n/netscratch/dwork_lab/Lab/katrina/envs/reasoning/bin/python -u Dynasor/benchmark/TokenDeprivation/run_adaptive.py \
    --dataset gsm8k \
    --mlp_train_dataset gsm8k \
    --mlp_train_split train \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --max-tokens 256 \
    --step 32 \
    --num-trials 10 \
    --temperature 0.6 \
    --top-p 0.95 \
    --probe-tokens 32 \
    --probe "... Oh, I suddenly got the answer to the whole problem, **Final Answer**\n\n\\[ \\boxed{" \
    --use-oracle \
    "$@"
    