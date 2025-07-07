#!/bin/bash
#SBATCH --job-name=lora_train # Job name
#SBATCH --account=kempner_gershman_lab
#SBATCH --partition=kempner_h100
#SBATCH --output=logs/lora_%A_%a.out # Standard output and error log
#SBATCH --error=logs/lora_%A_%a.err # Standard error file
#SBATCH --time=5:10:00 # Time limit
#SBATCH --nodes=1 # Number of nodes
#SBATCH --ntasks=1 # Number of tasks
#SBATCH --gres=gpu:1 # Request 1 GPUs
#SBATCH --mem=180G # Memory per node
#SBATCH --cpus-per-task=4 # Number of CPU cores per task


# Get script directory and change to project root
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
cd "$SCRIPT_DIR"

# Set Python command and output directory from environment or defaults
PYTHON_CMD=${PYTHON_CMD:-python3}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs"}

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

$PYTHON_CMD finetuning.py "--output_dir" "$OUTPUT_DIR" "--use_lora"