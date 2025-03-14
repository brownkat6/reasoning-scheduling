#!/bin/bash
#SBATCH --time=4:00:00             # Time limit
#SBATCH -p gpu_requeue         # Partition to submit to
#SBATCH --output=logs/numina_%A_%a.out   # Standard output and error log
#SBATCH --error=logs/numina_%A_%a.err    # Standard error file
#SBATCH --array=0-10        # Array jobs from 0 to 10 (11 total jobs)
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --gres=gpu:2               # Request 1 GPUs
#SBATCH --constraint='h100'        # Request H100 GPUs
#SBATCH --mem=128G                 # Memory per node
#SBATCH --cpus-per-task=4          # Number of CPU cores per task

mkdir -p logs
mkdir -p data/numina_results

# If it's the last array job (10), it's the test set
if [ $SLURM_ARRAY_TASK_ID -eq 10 ]; then
    python -u mlp_test.py --generate --batch_idx 0 --split test --dataset numina --csv_file data/numina_results/numina_results_test_0.csv --S 1024
else
    # Otherwise it's a train batch (0-9)
    python -u mlp_test.py --generate --batch_idx $SLURM_ARRAY_TASK_ID --split train --dataset numina --csv_file data/numina_results/numina_results_train_${SLURM_ARRAY_TASK_ID}.csv --S 1024
fi 