#!/bin/bash
#SBATCH --job-name=gsm8k_gen       # Job name
#SBATCH --partition=gpu_requeue
#SBATCH --output=logs/gsm8k_%A_%a.out   # Standard output and error log
#SBATCH --error=logs/gsm8k_%A_%a.err    # Standard error file
#SBATCH --array=0-88               # 75 train batches + 14 test batches
#SBATCH --time=4:00:00             # Time limit
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --gres=gpu:1               # Request 1 GPUs
#SBATCH --constraint='h100'        # Request H100 GPUs
#SBATCH --mem=128G                 # Memory per node
#SBATCH --cpus-per-task=4          # Number of CPU cores per task

# Create logs and data directories if they don't exist
mkdir -p logs
mkdir -p data/gsm8k_results

# Activate conda environment (adjust as needed)
source ~/.bashrc
conda activate myenv  # Replace with your environment name

# Calculate which split and batch to process
# Train split has 75 batches (0-74), test has 14 batches (75-88)
if [ $SLURM_ARRAY_TASK_ID -lt 75 ]; then
    # Process train split
    SPLIT="train"
    BATCH_IDX=$SLURM_ARRAY_TASK_ID
else
    # Process test split
    SPLIT="test"
    BATCH_IDX=$((SLURM_ARRAY_TASK_ID - 75))
fi

# Run the Python script
python mlp_test.py \
    --generate \
    --split $SPLIT \
    --batch_idx $BATCH_IDX \
    --csv_file "data/gsm8k_results/gsm8k_results_${SPLIT}_${BATCH_IDX}.csv"

echo "Completed processing $SPLIT split, batch $BATCH_IDX" 