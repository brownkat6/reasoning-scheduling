#!/bin/bash
#SBATCH --job-name=sweep_mlp       # Job name
#SBATCH --partition=gpu
#SBATCH --output=logs/sweep_mlp_%A_%a.out   # Standard output and error log
#SBATCH --error=logs/sweep_mlp_%A_%a.err    # Standard error file
#SBATCH --time=0:40:00             # Time limit
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --gres=gpu:1               # Request 1 GPUs
#SBATCH --mem=128G                 # Memory per node
#SBATCH --cpus-per-task=4          # Number of CPU cores per task


# if the environment variable $USER is katrinabrown, cd /n/home11/katrinabrown/thesis/reasoning-scheduling
if [ "$USER" == "katrinabrown" ]; then
    cd /n/home11/katrinabrown/thesis/reasoning-scheduling
    PYTHON="/n/netscratch/dwork_lab/Lab/katrina/envs/reasoning/bin/python"
else
    cd /n/home04/amuppidi/reasoning-scheduling
    PYTHON="~/.conda/envs/torch/bin/python"
fi
$PYTHON -u mlp_train.py