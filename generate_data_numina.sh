#!/bin/bash
#SBATCH -c 4                # Number of cores (-c)
#SBATCH -t 0-12:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p seas_gpu         # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=32G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=logs/numina_%A_%a.out   # Standard output and error log
#SBATCH --error=logs/numina_%A_%a.err    # Standard error file
#SBATCH --array=0-10        # Array jobs from 0 to 10 (11 total jobs)

mkdir -p logs
mkdir -p data/numina_results

# If it's the last array job (10), it's the test set
if [ $SLURM_ARRAY_TASK_ID -eq 10 ]; then
    python -u mlp_test.py --generate --batch_idx 0 --split test --dataset numina --csv_file data/numina_results/numina_results_test_0.csv --S 1024
else
    # Otherwise it's a train batch (0-9)
    python -u mlp_test.py --generate --batch_idx $SLURM_ARRAY_TASK_ID --split train --dataset numina --csv_file data/numina_results/numina_results_train_${SLURM_ARRAY_TASK_ID}.csv --S 1024
fi 