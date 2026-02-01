# Predictive Scheduling for Efficient Inference-Time Reasoning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2024.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2024.xxxxx)

This repository contains the official implementation of **Predictive Scheduling**, a plug-and-play framework for optimizing token budget allocation in large language model reasoning tasks.

> **Paper**: [Predictive Scheduling for Efficient Inference-Time Reasoning in Large Language Models](https://openreview.net/pdf?id=Mn3lrAWy20)  
> **Authors**: Katrina Brown, Aneesh Muppidi, Rana Shahout  

## Overview

LLMs achieve state-of-the-art accuracy on complex reasoning tasks by generating multiple chain-of-thought traces, but using fixed token budgets leads to over-computation on easy inputs and under-computation on hard ones. Our **Predictive Scheduling** framework addresses this by:

1. **Pre-run prediction**: Estimates optimal reasoning length or difficulty before generation
2. **Dynamic allocation**: Distributes token budgets based on predicted complexity  
3. **Plug-and-play design**: Works with existing LLMs without model modifications

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/brownkat6/reasoning-scheduling.git
cd reasoning-scheduling

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Basic Usage

```python
from predictive_scheduling import Config, load_config
from predictive_scheduling.models import MLPPredictor
from predictive_scheduling.allocation import GreedyAllocator

# Load configuration
config = load_config("config.yaml")

# Create MLP predictor for early stopping
mlp_config = {
    'input_dim': 1536,
    'hidden_dims': [256],
    'output_dim': 16,
    'activation': 'relu'
}
predictor = MLPPredictor(mlp_config)

# Create greedy allocator
allocator = GreedyAllocator(window_size=16, min_allocation=16)

# Allocate budget based on predictions
predictions = predictor.predict(hidden_states)  # Your hidden states
result = allocator.allocate(predictions, total_budget=1600)

print(f"Allocated budgets: {result.allocations}")
print(f"Average allocation: {result.average_allocation:.1f} tokens")
```

### Interactive Chat with Dynasor

```bash
# Start secure Dynasor chat client
dynasor-chat --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

# Or with custom settings
dynasor-chat --base-url http://localhost:8000/v1 --effort mid
```

## Project Structure

```
predictive_scheduling/
├── predictive_scheduling/          # Main package
│   ├── config.py                  # Configuration management  
│   ├── security.py                # Security utilities
│   ├── models/                    # Model implementations
│   │   ├── mlp.py                # MLP predictors
│   │   ├── lora_models.py        # LoRA fine-tuned models
│   │   └── base.py               # Base classes
│   ├── training/                  # Training utilities
│   │   ├── trainer.py            # Training loops
│   │   ├── data_loader.py        # Data loading
│   │   └── metrics.py            # Evaluation metrics
│   ├── allocation/                # Allocation algorithms
│   │   ├── greedy.py             # Greedy allocation
│   │   ├── baseline.py           # Baseline methods
│   │   └── base.py               # Base classes
│   └── dynasor/                   # Dynasor framework
│       ├── client.py             # Secure chat client
│       └── core.py               # Core functionality
├── examples/                      # Usage examples
├── scripts/                       # Training/evaluation scripts
├── config.yaml                   # Default configuration
└── requirements.txt               # Dependencies
```

## Methodology

### Early Stopping Prediction

**MLP Predictors**: Train lightweight MLPs on transformer hidden states to predict early stopping probabilities.

```python
from predictive_scheduling.models import MLPPredictor
from predictive_scheduling.training import MLPTrainer, create_data_loaders

# Load data and create predictors
train_loader, val_loader = create_data_loaders(train_data, val_data)
model = MLPPredictor(config)

# Train predictor
trainer = MLPTrainer(model, train_loader, val_loader)
results = trainer.train()
```

**LoRA Fine-tuning**: Fine-tune language models with LoRA for end-to-end prediction.

```python
from predictive_scheduling.models import EarlyStopFinetuner
from predictive_scheduling.training import LoRATrainer

# Create LoRA model
model = EarlyStopFinetuner(config)
trainer = LoRATrainer(model, train_loader, val_loader)
results = trainer.train()
```

### Difficulty Classification

Classify problems into easy/medium/hard categories for robust allocation:

```python
from predictive_scheduling.models import DifficultyClassifier

# Train difficulty classifier
classifier = DifficultyClassifier(config)
predictions = classifier.predict_class_names(["Calculate 2+2", "Prove Fermat's Last Theorem"])
# Output: ['easy', 'hard']
```

### Token Budget Allocation

**Greedy Allocation**: Dynamically allocate tokens based on predicted gains.

```python
from predictive_scheduling.allocation import GreedyAllocator

allocator = GreedyAllocator(window_size=16)
result = allocator.allocate(predictions, total_budget=1600)
```

**Difficulty-Based Allocation**: Allocate based on discrete difficulty categories.

```python
from predictive_scheduling.allocation import DifficultyBasedAllocator

allocator = DifficultyBasedAllocator(categories=['easy', 'medium', 'hard'])
result = allocator.allocate(difficulty_predictions, total_budget=1600)
```

## Reproducing Results

### Training Predictors

```bash
# Train MLP predictor on GSM8K
python scripts/train_mlp.py \
    --dataset gsm8k \
    --hidden-layer 16 \
    --hidden-dims 256 \
    --activation relu \
    --num-epochs 20

# Train LoRA difficulty classifier  
python scripts/train_difficulty.py \
    --model-name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --use-lora \
    --num-epochs 10
```

### Evaluation

```bash
# Evaluate allocation strategies
python scripts/evaluate_allocation.py \
    --predictors models/mlp_gsm8k_layer_16.pt \
    --test-data data/gsm8k_test.jsonl \
    --budget-range 16,256,16

# Generate paper figures
python scripts/generate_figures.py \
    --results results/ \
    --output figures/
```

## Configuration

The framework uses YAML configuration files for easy customization:

```yaml
# config.yaml
model:
  model_name: "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
  hidden_size: 1536
  use_flash_attention: true

training:
  batch_size: 32
  learning_rate: 0.0001
  num_epochs: 10
  seed: 42

allocation:
  window_size: 16
  min_allocation: 16
  max_allocation: 256
```


## Performance
| Model Type | Correlation | Training Time | Inference Speed |
|------------|-------------|---------------|-----------------|
| MLP (Layer 16) | 0.742 | ~10 min | <1ms/query |
| LoRA Early Stop | 0.444 | ~2 hours | ~50ms/query |
| LoRA Difficulty | 66.3% acc | ~30 min | ~50ms/query |


## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{brown2025predictive,
  title={Predictive Scheduling for Efficient Inference-Time
         Reasoning in Large Language Models},
  author={Brown, Katrina and Muppidi, Aneesh and Shahout, Rana},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## Contact

- **Aneesh Muppidi**: [aneeshmuppidi@college.harvard.edu](mailto:aneeshmuppidi@college.harvard.edu)
- **Katrina Brown**: [katrinabrown@college.harvard.edu](mailto:katrinabrown@college.harvard.edu)
- **Rana Shahout**: [rana@seas.harvard.edu ](mailto:rana@seas.harvard.edu )
