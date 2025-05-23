#!/bin/bash
#SBATCH --job-name=data_gen       # Job name
#SBATCH --partition=seas_gpu
#SBATCH --output=logs/datagen_%A_%a.out   # Standard output and error log
#SBATCH --error=logs/datagen_%A_%a.err    # Standard error file
#SBATCH --array=0-91               # 89 regular jobs + 3 special dataset jobs
#SBATCH --time=4:00:00             # Time limit
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --gres=gpu:1               # Request 1 GPUs
#SBATCH --constraint='h100'        # Request H100 GPUs
#SBATCH --mem=128G                 # Memory per node
#SBATCH --cpus-per-task=4          # Number of CPU cores per task

## TODO: launch math500 5 generation calls

# Create logs and data directories if they don't exist
mkdir -p logs
mkdir -p data/gsm8k_results

# Activate conda environment (adjust as needed)
#source ~/.bashrc
#conda activate dat  # Replace with your environment name

# Parse GENERATE_X from the first command line arg and GENERATE_Y from the second command line arg
GENERATE_X=$1
GENERATE_Y=$2

# Handle special cases for last three array indices
if [ $SLURM_ARRAY_TASK_ID -eq 89 ]; then
    # Process math500
    SPLIT="train"
    BATCH_IDX=0
    DATASET="math500"
elif [ $SLURM_ARRAY_TASK_ID -eq 90 ]; then
    # Process amc23
    SPLIT="train"
    BATCH_IDX=0
    DATASET="amc23"
elif [ $SLURM_ARRAY_TASK_ID -eq 91 ]; then
    # Process aime24
    SPLIT="train"
    BATCH_IDX=0
    DATASET="aime24"
else
    # Original GSM8K processing
    if [ $SLURM_ARRAY_TASK_ID -lt 75 ]; then
        # Process train split
        SPLIT="train"
        BATCH_IDX=$SLURM_ARRAY_TASK_ID
    else
        # Process test split
        SPLIT="test"
        BATCH_IDX=$((SLURM_ARRAY_TASK_ID - 75))
    fi
    DATASET="gsm8k"
fi

# Note that generating Y data is MUCH more expensive than generating X data
# Example generates only X data: $ sbatch generate_data_gsm8k.sh True False
# Example generates only Y data: $ sbatch generate_data_gsm8k.sh False True
# Example generates both X and Y data: $ sbatch generate_data_gsm8k.sh True True

# Run mlp_datagen.py with appropriate parameters
/n/netscratch/dwork_lab/Lab/katrina/envs/reasoning/bin/python -u mlp_datagen.py \
    --split $SPLIT \
    --batch_idx $BATCH_IDX \
    --dataset $DATASET \
    --generate-X-data $GENERATE_X \
    --generate-Y-data $GENERATE_Y

echo "Completed processing $SPLIT split, batch $BATCH_IDX for dataset $DATASET" 