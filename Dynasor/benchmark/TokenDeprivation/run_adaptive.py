import argparse
import torch
import numpy as np
from tqdm import tqdm
from utils import save_json, load_dataset, set_seed
from dynasor.core.evaluator import extract_answer, strip_string, math_equal, extract_first_boxed_answer
from clients import vllmClientModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from run import execute_question_reuse
# add the directory containing mlp_test to the path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
print(f"Adding {os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))} to path")
import torch.nn as nn
from datetime import datetime
import random
from collections import Counter
from mlp import MLP

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
    parser.add_argument(
        "--use-oracle",
        action="store_true",
        help="Use ground truth early stopping proportions instead of MLP predictions"
    )
    return parser.parse_args()

def load_model_and_tokenizer(model_name, url=None, api_key=None, cache_dir=None):
    """Load either local model or vllm endpoint"""
    # if url:
    if False:
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

def get_hidden_states(model, tokenizer, prompts, batch_size=16):
    """Get hidden states using local model in batches"""
    device = next(model.parameters()).device
    all_hidden_states = []
    print(f"Getting hidden states for {len(prompts)} prompts")
    
    # Process prompts in batches
    for i in range(0, len(prompts), batch_size):
        print(f"Processing batch {i//batch_size + 1} of {len(prompts)//batch_size}")
        batch_prompts = prompts[i:i + batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)
        
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=True)
            # Get hidden states from last layer, last token
            batch_hidden_states = outputs.hidden_states[-1][:, -1, :].detach()
            all_hidden_states.append(batch_hidden_states)
            
        # Clear CUDA cache after each batch
        torch.cuda.empty_cache()
    
    # Concatenate all batches
    hidden_states = torch.cat(all_hidden_states, dim=0)
    return hidden_states

def optimize_token_allocation(predictions, token_budget, W=16):
    """
    Optimize token allocation given predictions and budget constraint.
    predictions: list of prediction vectors from MLP
    token_budget: average token limit across all queries
    W: window size (16)
    Returns: list of token allocations for each query
    """
    print(f"Optimizing token allocation for {len(predictions)} queries with token budget {token_budget} and W {W}")
    num_queries = len(predictions)
    max_positions = len(predictions[0])
    
    # Convert predictions to numpy for easier manipulation
    pred_array = np.array(predictions)
    max_budget_per_query = max_positions * W
    
    # Initialize with minimum tokens
    allocations = np.ones(num_queries, dtype=int) * W
    # print(f"Allocation counts before increase: {Counter(allocations)}")
    
    # Calculate how many more tokens we can allocate
    remaining_budget = token_budget * num_queries - np.sum(allocations)
    print(remaining_budget,"remaining budget")
    while remaining_budget >= W:
        # For each query, calculate potential gain from adding W tokens
        best_gain=-1
        window_increase=1
        while best_gain < 0:
            gains = []
            for i in range(num_queries):
                current_pos = allocations[i] // W - 1
                if current_pos + window_increase >= max_positions:
                    gains.append(-1)  # Can't allocate more tokens
                else:
                    gains.append(pred_array[i][current_pos + window_increase] - pred_array[i][current_pos])
            
            # Find query with maximum gain
            best_query = np.argmax(gains)
            best_gain = gains[best_query]
            if gains[best_query] <= 0:
                # print("No more gains, breaking",gains)
                # break
                window_increase+=1
            # if allgains values are -1, break
            if all([g == -1 for g in gains]):
                print(f"No more gains, breaking after checking {window_increase} windows")
                break
        if best_gain == -1:
            break
        # Allocate W more tokens to the best query
        allocations[best_query] += W
        remaining_budget -= W
    print(remaining_budget,"remaining budget after allocating additional tokens")
    
    i=0
    while remaining_budget >= W:
        # if all allocations are >= max_budget_per_query
        if all([a >= max_budget_per_query for a in allocations]):
            break
        if allocations[i % (num_queries)] + W <= max_budget_per_query:
            allocations[i % (num_queries)] += W
            remaining_budget -= W
        i+=1
    print(remaining_budget,"remaining budget after distributing additional tokens")
    print(f"Allocation counts: {Counter(allocations)}")
    
    # print what the expected reward under prediction are under allocation, and compare it to the expected reward if all queries
    # were allocated token_budget uniformly
    # print(predictions,"predictions")
    expected_reward_uniform = np.mean([predictions[i][token_budget//W - 1] for i in range(num_queries)])
    expected_reward_allocation = np.mean([predictions[i][allocations[i]//W - 1] for i in range(num_queries)])
    print(f"Expected reward under uniform allocation: {np.round(expected_reward_uniform,2)}")
    print(f"Expected reward under allocation: {np.round(expected_reward_allocation,2)}")
    print(f"Mean token allocation: {np.mean(allocations)}, target token budget: {token_budget}")
    
    return allocations.tolist()

def set_deterministic_mode():
    """Set all random seeds and ensure deterministic operations."""
    # Set seeds for all sources of randomness
    torch.manual_seed(42)  # Use a fixed seed
    random.seed(42)
    np.random.seed(42)
    
    # Make CUDA operations deterministic
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    args = parse_args()
    
    # Instead of using args.seed, we set fixed seeds for all processes
    set_deterministic_mode()
    
    data = load_dataset(args.dataset)
    cache_dir = "/n/holylabs/LABS/dwork_lab/Everyone/cache/transformers"

    # Setup output directory
    if args.output:
        output_dir = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        model_name = args.model.replace("/", "-")
        prefix = "oracle" if args.use_oracle else "adaptive"
        output_dir = f"results/{prefix}_{model_name}_{args.dataset}_mlp{args.mlp_train_dataset}_{args.mlp_train_split}_{args.start}_{args.end}"
    os.makedirs(output_dir, exist_ok=True)

    # Process questions in range
    questions = data[args.start:args.end]
    prompts = [item["problem"].strip() for item in questions]
    targets = [strip_string(item["answer"]) for item in questions]

    if args.use_oracle:
        # Load oracle data from grouped CSV
        # STEM = "/n/netscratch/dwork_lab/Lab/katrina/reasoning_scheduling/"
        STEM = "/n/netscratch/dwork_lab/Lab/katrina/reasoning_scheduling_new/"
        oracle_file = f"{STEM}data/{args.dataset}_results/{args.dataset}_results_{args.mlp_train_split}_grouped.csv"
        if not os.path.exists(oracle_file):
            raise ValueError(f"Oracle data not found at {oracle_file}")
        
        import pandas as pd
        import ast
        oracle_data = pd.read_csv(oracle_file)
        print(oracle_data.shape)
        print(oracle_data.columns)
        oracle_data["question_id"] = oracle_data["question_id"].apply(lambda x: int(x.replace("test_","").replace("train_","")))
        oracle_data = oracle_data.drop_duplicates(subset=['question_id']).sort_values(by='question_id')
        print(f"Question_ids: {oracle_data['question_id'].tolist()}")
        print(oracle_data.shape)
        # Convert string representation of lists to actual lists
        oracle_data['early_stop_correct_proportions'] = oracle_data['early_stop_correct_proportions'].apply(ast.literal_eval)
        predictions = np.array(oracle_data['early_stop_correct_proportions'].tolist())
        predictions = predictions[args.start:args.end]
        
        print(f"Loaded oracle data from {oracle_file}")
    else:
        # Load and use MLP for predictions
        mlp_path = f'models/mlp_{args.mlp_train_dataset}_{args.mlp_train_split}.pt'
        checkpoint = torch.load(mlp_path, map_location='cuda', weights_only=False)
        if 'model' in checkpoint:
            mlp_model = checkpoint['model'].cuda()
        else:
            config = checkpoint['config']
            config['hidden_dim']=256
            mlp_model = MLP(
                input_dim=config['input_dim'],
                hidden_dim=config['hidden_dim'],
                output_dim=config['output_dim']
            ).cuda()
            mlp_model.load_state_dict({k: v.cuda() for k, v in checkpoint['model_state_dict'].items()})
        mlp_model.eval()

        # Get model and tokenizer for hidden states
        model, tokenizer = load_model_and_tokenizer(args.model, cache_dir)
        
        # Get hidden states and predictions
        hidden_states = get_hidden_states(model, tokenizer, prompts)
        with torch.inference_mode():
            predictions = mlp_model(hidden_states.cuda()).cpu().numpy()

    # Process each token budget
    token_budgets = list(range(args.step, args.max_tokens + args.step, args.step))
    print(token_budgets,"token budgets")
    for token_budget in token_budgets:
        print(f"\nProcessing token budget: {token_budget}")
        
        # Create subdirectory for this token budget
        budget_dir = os.path.join(output_dir, f"budget_{token_budget}")
        os.makedirs(budget_dir, exist_ok=True)
        
        # Calculate expected reward under uniform allocation
        uniform_reward = np.mean([predictions[i][token_budget//16 - 1] for i in range(len(predictions))])
        print(f"Expected reward under uniform allocation: {np.round(uniform_reward,3)}")
        
        # Optimize token allocation for this budget
        max_tokens = optimize_token_allocation(predictions, token_budget)
        
        # Calculate expected reward under optimized allocation
        optimized_reward = np.mean([predictions[i][max_tokens[i]//16 - 1] for i in range(len(predictions))])
        print(f"Expected reward under allocation: {np.round(optimized_reward,3)}")
        # print(f"Allocation: {max_tokens}")
        
        # Execute questions with optimized token allocations
        model, tokenizer = load_model_and_tokenizer(args.model, cache_dir)
        predicted_scores = []
        actual_scores = []
        for i, (prompt, target) in enumerate(zip(prompts, targets)):
            print(f"Question {i+args.start}: allocated {max_tokens[i]} tokens")
            
            # Calculate window index for predictions
            window_index = max_tokens[i]//16 - 1
            
            # Get appropriate prediction based on mode
            predicted_score = predictions[i][window_index] if window_index < len(predictions[i]) else 0.0
            
            proportion_correct,stats =execute_question_reuse(
                model,
                prompt,
                target,
                max_tokens=[max_tokens[i]],
                probe=args.probe,
                probe_tokens=args.probe_tokens,
                num_trials=args.num_trials,
                problem_id=i+args.start,
                output_dir=budget_dir,
                top_p=args.top_p,
                temperature=args.temperature,
                tokenizer=tokenizer,
                predicted_score=predicted_score  # Pass prediction to execute_question_reuse
            )
            predicted_scores.append(predicted_score)
            actual_scores.append(proportion_correct)

        # print predicted and actual scores across all questions in the given token budget
        print(f"Token budget: {token_budget}")
        print(f"Predicted scores: {predicted_scores}")
        print(f"Actual scores: {actual_scores}")
        # print mean predicted and actual scores
        print(f"Mean predicted score: {np.mean(predicted_scores)}")
        print(f"Mean actual score: {np.mean(actual_scores)}\n")

    print(f"Saved results to {output_dir}")

if __name__ == "__main__":
    main()