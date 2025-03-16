import argparse
import torch
import numpy as np
from tqdm import tqdm
from utils import save_json, load_dataset, set_seed
from dynasor.core.evaluator import extract_answer, strip_string, math_equal, extract_first_boxed_answer
from clients import vllmClientModel
from transformers import AutoTokenizer, AutoModelForCausalLM
# add the directory containing mlp_test to the path
import sys
import os
#sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
#print(f"Adding {os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))} to path")
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim=1536, hidden_dim=256, output_dim=256):
        super(MLP, self).__init__()
        print(f"Input dim: {input_dim}, Hidden dim: {hidden_dim}, Output dim: {output_dim}")
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # Output layer
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def parse_args():
    parser = argparse.ArgumentParser(description="Adaptive Token Deprivation Experiment")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["amc23", "aime24", "GPQADiamond", "math500", "gsm8k"],
        help="Dataset to use"
    )
    # Add arguments for MLP loading
    parser.add_argument(
        "--mlp_train_dataset",
        type=str,
        required=True,
        help="Dataset used to train the MLP"
    )
    parser.add_argument(
        "--mlp_train_split",
        type=str,
        required=True,
        help="Split used to train the MLP"
    )
    # Keep existing arguments from run.py
    parser.add_argument("--output", type=str, default="", help="Path to output results file")
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        help="Name or path of the model to use"
    )
    parser.add_argument("--probe", type=str, default="**Final Answer**\n\n\\[ \\boxed{")
    parser.add_argument("--probe-tokens", type=int, default=10)
    parser.add_argument("--url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--api-key", type=str, default="token-abc123")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=10000)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--step", type=int, default=128)
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()

def load_model_and_tokenizer(model_name, url=None, api_key=None, cache_dir=None):
    """Load either local model or vllm endpoint"""
    if url:
        return vllmClientModel(model_name, url, api_key), None
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_hidden_states=True,
            device_map="auto",
            cache_dir=cache_dir
        )
        model.eval()
        return model, tokenizer

def get_hidden_states(model, tokenizer, prompts, is_vllm=False):
    """Get hidden states for a batch of prompts"""
    if is_vllm:
        # Use vllm client's forward pass functionality
        return model.get_hidden_states(prompts)
    else:
        # Local model forward pass
        device = next(model.parameters()).device
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1][:, -1, :].detach()
        return hidden_states

def optimize_token_allocation(predictions, token_budget, W=16):
    """
    Optimize token allocation given predictions and budget constraint.
    predictions: list of prediction vectors from MLP
    token_budget: average token limit across all queries
    W: window size (16)
    Returns: list of token allocations for each query
    """
    num_queries = len(predictions)
    max_positions = len(predictions[0])
    
    # Convert predictions to numpy for easier manipulation
    pred_array = np.array(predictions)
    
    # Initialize with minimum tokens
    allocations = np.ones(num_queries, dtype=int) * W
    
    # Calculate how many more tokens we can allocate
    remaining_budget = token_budget * num_queries - np.sum(allocations)
    
    while remaining_budget >= W:
        # For each query, calculate potential gain from adding W tokens
        gains = []
        for i in range(num_queries):
            current_pos = allocations[i] // W - 1
            if current_pos + 1 >= max_positions:
                gains.append(-1)  # Can't allocate more tokens
            else:
                gains.append(pred_array[i][current_pos + 1] - pred_array[i][current_pos])
        
        # Find query with maximum gain
        best_query = np.argmax(gains)
        if gains[best_query] <= 0:
            break
        
        # Allocate W more tokens to the best query
        allocations[best_query] += W
        remaining_budget -= W
    
    # print what the expected reward under prediction are under allocation, and compare it to the expected reward if all queries
    # were allocated token_budget uniformly
    expected_reward_uniform = np.mean([predictions[i][token_budget//W] for i in range(num_queries)])
    expected_reward_allocation = np.mean([predictions[i][allocations[i]//W] for i in range(num_queries)])
    print(f"Expected reward under uniform allocation: {expected_reward_uniform}")
    print(f"Expected reward under allocation: {expected_reward_allocation}")
    print(f"Allocation: {allocations}")
    
    return allocations.tolist()

def main():
    args = parse_args()
    set_seed(args.seed)
    data = load_dataset(args.dataset)
    cache_dir = "/n/holylabs/LABS/dwork_lab/Everyone/cache/transformers"

    # Load MLP model
    print(f"Loading MLP")
    mlp_path = f'models/mlp_{args.mlp_train_dataset}_{args.mlp_train_split}.pt'
    checkpoint = torch.load(mlp_path)
    mlp_model = MLP(
        input_dim=checkpoint['config']['input_dim'],
        hidden_dim=checkpoint['config']['hidden_dim'],
        output_dim=checkpoint['config']['output_dim']
    ).to('cuda')
    mlp_model.load_state_dict(checkpoint['model_state_dict'])
    mlp_model.eval()

    # Setup output directory
    import os
    from datetime import datetime
    if args.output:
        output_dir = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        model_name = args.model.replace("/", "-")
        output_dir = f"results/adaptive_{model_name}_{args.dataset}_mlp{args.mlp_train_dataset}_{args.mlp_train_split}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Load base model
    print(f"Loading base model")
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        args.url if not cache_dir else None,
        args.api_key,
        cache_dir
    )
    is_vllm = tokenizer is None

    # Process questions in range
    questions = data[args.start:args.end]
    prompts = [item["problem"].strip() for item in questions]
    targets = [strip_string(item["answer"]) for item in questions]

    # Get hidden states for all questions
    hidden_states = get_hidden_states(model, tokenizer, prompts, is_vllm)
    
    # Get MLP predictions for all questions
    with torch.inference_mode():
        predictions = mlp_model(hidden_states).cpu().numpy()

    # Process each token budget
    token_budgets = list(range(args.step, args.max_tokens + args.step, args.step))
    
    for token_budget in token_budgets:
        print(f"\nProcessing token budget: {token_budget}")
        
        # Optimize token allocation for this budget
        max_tokens = optimize_token_allocation(predictions, token_budget)
        
        # Execute questions with optimized token allocations
        for i, (prompt, target) in enumerate(zip(prompts, targets)):
            print(f"Question {i+args.start}: allocated {max_tokens[i]} tokens")
            execute_question_reuse(
                model,
                prompt,
                target,
                max_tokens=[max_tokens[i]],  # Pass as list for compatibility
                probe=args.probe,
                probe_tokens=args.probe_tokens,
                num_trials=args.num_trials,
                problem_id=i+args.start,
                output_dir=output_dir,
                top_p=args.top_p,
                temperature=args.temperature,
            )

    print(f"Saved results to {output_dir}")

if __name__ == "__main__":
    main()