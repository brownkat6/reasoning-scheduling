#!/bin/bash
#SBATCH -c 4                # Number of cores (-c)
#SBATCH -t 0-12:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p seas_gpu         # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=32G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o data/numina_results/logs/myoutput_%A_%a.out  # File to which STDOUT will be written
#SBATCH -e data/numina_results/logs/myerrors_%A_%a.err  # File to which STDERR will be written
#SBATCH --array=0-10        # Array jobs from 0 to 10 (11 total jobs)

module load Anaconda3/2020.11
module load cuda/12.2.0-fasrc01
module load gcc/12.2.0-fasrc01

source activate dat

# Create the output directories if they don't exist
mkdir -p data/numina_results/logs

# If it's the last array job (10), it's the test set
if [ $SLURM_ARRAY_TASK_ID -eq 10 ]; then
    python mlp_test.py --generate --batch_idx 0 --split test --dataset numina --csv_file data/numina_results/numina_results_test_0.csv --S 1024
else
    # Otherwise it's a train batch (0-9)
    python mlp_test.py --generate --batch_idx $SLURM_ARRAY_TASK_ID --split train --dataset numina --csv_file data/numina_results/numina_results_train_${SLURM_ARRAY_TASK_ID}.csv --S 1024
fi 