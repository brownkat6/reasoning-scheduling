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

# if the environment variable $USER is katrinabrown, cd /n/home11/katrinabrown/thesis/reasoning-scheduling
if [ "$USER" == "katrinabrown" ]; then
    cd /n/home11/katrinabrown/thesis/reasoning-scheduling
    PYTHON="/n/netscratch/dwork_lab/Lab/katrina/envs/reasoning/bin/python"
else
    cd /n/home04/amuppidi/reasoning-scheduling
    PYTHON="~/.conda/envs/torch/bin/python"
fi
$PYTHON -u mlp_train_orig.py \
  --X-STEM "$X_STEM" \
  --layer "$LAYER_NAME"
