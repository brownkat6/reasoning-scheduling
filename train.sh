#!/bin/bash
#SBATCH --job-name=mlp_train # Job name
#SBATCH --partition=seas_gpu
#SBATCH --output=logs/sweep_mlp_%A_%a.out # Standard output and error log
#SBATCH --error=logs/sweep_mlp_%A_%a.err # Standard error file
#SBATCH --time=0:10:00 # Time limit
#SBATCH --nodes=1 # Number of nodes
#SBATCH --ntasks=1 # Number of tasks
#SBATCH --gres=gpu:1 # Request 1 GPUs
#SBATCH --mem=128G # Memory per node
#SBATCH --cpus-per-task=4 # Number of CPU cores per task

# X_STEM will be passed as command line argument
X_STEM=$1
LAYER_NAME=$2

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Running with X_STEM: $X_STEM"
echo "Layer name: $LAYER_NAME"

cd /n/home04/amuppidi/reasoning-scheduling
~/.conda/envs/torch/bin/python mlp_train_orig.py \
  --X-STEM "$X_STEM" \
  --layer "$LAYER_NAME"