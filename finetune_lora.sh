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


cd /n/home04/amuppidi/reasoning-scheduling
~/.conda/envs/torch/bin/python finetuning.py  "--output_dir" "/n/netscratch/gershman_lab/Lab/amuppidi/reasoning" "--use_lora"