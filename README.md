## Project Overview  
Predictive Scheduling is a plug-and-play framework for dynamically allocating inference-time token budgets across a batch of large language model (LLM) queries to maximize overall accuracy under fixed compute constraints. It pre-runs two lightweight predictors—a multilayer perceptron (MLP) on intermediate transformer hidden states and a LoRA-fine-tuned classifier on raw question text—to estimate per-query reasoning length or discrete difficulty before full generation. Using these predictions, a greedy batch allocator distributes tokens across queries, prioritizing those with the greatest expected accuracy gains. On the GSM8K arithmetic reasoning benchmark, this approach achieves up to 7.9 percentage-point accuracy improvements over uniform budgeting at identical token cost, closing more than 50% of the gap to an oracle with perfect foresight.

## Key Features  
- **Pre-Run Prediction**  
  Extract hidden-state features from a 28-layer transformer (DeepSeek-R1-Distill-Qwen-1.5B) to forecast needed token budgets per query without modifying the base model.  
- **LoRA Classification**  
  Fine-tune low-rank adapter modules to classify problem difficulty from raw text, leveraging the LoRA paradigm for parameter-efficient adaptation.  
- **Greedy Allocator**  
  Implement a simple yet effective greedy algorithm that maximizes expected accuracy gains under a total token budget constraint.  
- **Batch Scheduling Integration**  
  Compatible with modern inference engines such as vLLM, enabling seamless plug-in deployment for latency-sensitive applications.

## Repository Structure  
README.md
requirements.txt
src/
├── data_processing.py # GSM8K preprocessing and probe-based early-stopping data generation
├── predictors/
│ ├── mlp_predictor.py # MLP training and inference on hidden states
│ └── lora_predictor.py # LoRA fine-tuning and classification scripts
├── allocator/
│ ├── greedy_allocator.py # Greedy token allocation logic
│ └── difficulty_allocator.py# Difficulty-based allocation routines
└── experiments/
├── run_mlp_experiments.sh
└── run_lora_experiments.sh

## Installation  
1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-org/predictive-scheduling.git```
2. **Install dependencies**
    ```cd predictive-scheduling
    pip install -r requirements.txt```
3. **Ensure GPU support**
    ```PyTorch >=1.10 with CUDA is recommended for efficient training and inference.```
## Usage
### Data Processing
Generate early-stopping training data for GSM8K:
```python src/data_processing.py \
  --input_path data/gsm8k_train.json \
  --output_dir data/processed/ \
  --probe_interval 16 \
  --max_tokens 256```

### Training Predictors
MLP Predictor
```python src/predictors/mlp_predictor.py \
  --layer 16 \
  --train_data data/processed/train.pkl \
  --val_data data/processed/val.pkl \
  --out_dir models/mlp_layer16/```

LoRA Classifier
```python src/predictors/lora_predictor.py \
  --train_data data/processed/train.json \
  --val_data data/processed/val.json \
  --epochs 10 \
  --out_dir models/lora_classifier/```
### Token Allocation
Greedy Allocation with MLP Predictions
```python src/allocator/greedy_allocator.py \
  --predictions models/mlp_layer16/predictions.npy \
  --budget 96 \
  --window 16 \
  --output allocations/mlp_allocations.json```
Difficulty-Based Allocation
```python src/allocator/difficulty_allocator.py \
  --predictions models/lora_classifier/predictions.json \
  --budget 96 \
  --window 16 \
  --output allocations/difficulty_allocations.json```
