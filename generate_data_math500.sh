#!/bin/bash
#SBATCH --job-name=math500_gen       # Job name
#SBATCH --partition=gpu_requeue
#SBATCH --output=logs/math500_%A_%a.out   # Standard output and error log
#SBATCH --error=logs/math500_%A_%a.err    # Standard error file
#SBATCH --array=0-4               # 5 train batches
#SBATCH --time=4:00:00             # Time limit
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --gres=gpu:2               # Request 1 GPUs
#SBATCH --constraint='h100'        # Request H100 GPUs
#SBATCH --mem=128G                 # Memory per node
#SBATCH --cpus-per-task=4          # Number of CPU cores per task

## TODO: launch math500 5 generation calls

# Create logs and data directories if they don't exist
mkdir -p logs
mkdir -p data/math500_results

# Run the Python script
python -u mlp_test.py \
    --generate \
    --split test \
    --batch_idx $SLURM_ARRAY_TASK_ID \
    --csv_file "data/math500_results/math500_results_${SPLIT}_${BATCH_IDX}.csv" \
    --dataset math500 \
    --S 1024

echo "Completed processing $SPLIT split, batch $BATCH_IDX" 