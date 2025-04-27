#!/bin/bash
#SBATCH --time=18:00:00           # Increased time limit for all models
#SBATCH --partition=seas_compute
#SBATCH --output=logs/model_pipeline_%A_%a.out
#SBATCH --error=logs/model_pipeline_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --array=0-3              # One job per model

# Create necessary directories
mkdir -p logs
mkdir -p models
mkdir -p figures

# Define the models array
declare -a models=(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
)

# Get the current model based on array task ID
MODEL="${models[$SLURM_ARRAY_TASK_ID]}"
echo "Processing model: $MODEL"

# Convert model path to a filename-safe format
MODEL_SAFE=$(echo $MODEL | sed 's/\//-/g')

# Step 1: Train MLP
echo "Starting MLP training for $MODEL"
/n/netscratch/dwork_lab/Lab/katrina/envs/reasoning/bin/python -u mlp_train.py \
    --train_dataset gsm8k \
    --train_split train \
    --test_dataset gsm8k \
    --test_split test \
    --hidden-layer 16 \
    --hidden-dims 256 \
    --activation relu \
    --learning-rate 1e-3 \
    --batch-size 32 \
    --num-epochs 25 \
    --dropout 0.0 \
    --use-wandb

# Check if MLP training was successful
if [ $? -ne 0 ]; then
    echo "MLP training failed for $MODEL"
    exit 1
fi

# Define MLP model path
MLP_MODEL="models/mlp_gsm8k_train_layer_16_arch_256_act_relu_drop_0.00.pt"

# Step 2: Run adaptive experiments
echo "Starting adaptive experiments for $MODEL"
./vis_adaptive.sh \
    "$MLP_MODEL" \
    16 \
    100

# Check if adaptive experiments were successful
if [ $? -ne 0 ]; then
    echo "Adaptive experiments failed for $MODEL"
    exit 1
fi

# Step 3: Visualize results
echo "Starting visualization for $MODEL"
/n/netscratch/dwork_lab/Lab/katrina/envs/reasoning/bin/python -u vis_adaptive.py \
    --dataset gsm8k \
    --model "$MODEL" \
    --split train \
    --start 0 \
    --end 100 \
    --mlp_model "$MLP_MODEL" \
    --hidden_layer 16 \
    --output "figures/adaptive_gsm8k_${MODEL_SAFE}_layer16_256"

# Check if visualization was successful
if [ $? -ne 0 ]; then
    echo "Visualization failed for $MODEL"
    exit 1
fi

echo "All steps completed successfully for $MODEL"
