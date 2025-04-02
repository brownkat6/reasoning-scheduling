#!/bin/bash
#SBATCH --job-name=reasoning_schedule_sweep # Job name
#SBATCH --partition=seas_gpu
#SBATCH --account=hankyang_lab # Account to charge for GPU usage
#SBATCH --output=logs/reasoning_schedule_sweep_%A_%a.out # Standard output and error log
#SBATCH --error=logs/reasoning_schedule_sweep_%A_%a.err # Standard error file
#SBATCH --array=0-91 # 60 jobs total
#SBATCH --time=0:05:00 # Time limit
#SBATCH --nodes=1 # Number of nodes
#SBATCH --ntasks=1 # Number of tasks
#SBATCH --gres=gpu:1 # Request 1 GPUs
#SBATCH --mem=128G # Memory per node
#SBATCH --cpus-per-task=4 # Number of CPU cores per task


SWEEP_ID="fh3odiup"

# Run the sweep agent for a single run
# if the environment variable $USER is katrinabrown, cd /n/home11/katrinabrown/thesis/reasoning-scheduling
if [ "$USER" == "katrinabrown" ]; then
    cd /n/home11/katrinabrown/thesis/reasoning-scheduling
    PYTHON="/n/netscratch/dwork_lab/Lab/katrina/envs/reasoning/bin/python"
else
    cd /n/home04/amuppidi/reasoning-scheduling
    PYTHON="~/.conda/envs/torch/bin/python"
fi
$PYTHON mlp_train.py --sweep-id $SWEEP_ID --use-wandb

echo "Completed sweep run ${SLURM_ARRAY_TASK_ID}"