## Predictive Scheduling for Efficient Inference-Time Reasoning in Large Language Models
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

## Installation  
1. **Clone the repository**    
   ```
   bash
   git clone https://github.com/your-org/predictive-scheduling.git
   ```
2. **Ensure GPU support**   
    ```
    PyTorch >=1.10 with CUDA is recommended for efficient training and inference.
    ```
