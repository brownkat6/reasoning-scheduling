import argparse
import re
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from scipy.stats import pearsonr
from datasets import load_dataset
from Dynasor.benchmark.TokenDeprivation import utils
from Dynasor.benchmark.TokenDeprivation.run import execute_question_reuse
from Dynasor.benchmark.TokenDeprivation import run
from Dynasor.dynasor.core.evaluator import math_equal

from transformers import AutoTokenizer, AutoModelForCausalLM

def load_dynasor_dataset(dataset_name, split="test"):
    """
    Load a Dynasor dataset using the Dynasor utils.
    """
    assert dataset_name in ["amc23", "aime24", "GPQADiamond", "math500", "gsm8k"], f"Dataset {dataset_name} not supported"
    print(f"Loading dataset {dataset_name} with dynasor utils")
    data = utils.load_dataset(dataset_name, split=split)
    test_data = []
    for i, sample in enumerate(data):
        question = sample["problem"]
        answer_field = str(sample["answer"])
        ground_truth = answer_field.strip()
        test_data.append({"id": f"{split}_{i}", "problem": question, "answer": ground_truth})
    return test_data

def get_model_and_tokenizer():
    """
    Load the distilled Qwen 1.5B model and its tokenizer.
    Throws an error if the model cannot be loaded.
    """
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir="/n/holylabs/LABS/dwork_lab/Everyone/cache/transformers")
        model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True, cache_dir="/n/holylabs/LABS/dwork_lab/Everyone/cache/transformers", device_map="cuda")
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Error loading model {model_name}: {e}")
    return model, tokenizer

def calculate_entropy(probs):
    """
    Calculate the entropy of a probability distribution.
    
    Args:
        probs: Tensor of probabilities (should sum to 1)
    
    Returns:
        entropy: The entropy of the distribution
    """
    # Ensure probabilities are valid (avoid log(0))
    valid_probs = probs[probs > 0]
    entropy = -torch.sum(valid_probs * torch.log2(valid_probs))
    return entropy.item()

def generate_data_entropy(batch_idx, split='train', num_samples=5, output_csv='entropy_results.csv', 
                         batch_size=100, dataset='gsm8k'):
    '''
    This function generates entropy-based predictors for each prompt.
    It captures:
    1. Entropy of the next token distribution (uncertainty)
    2. Max probability of the next token (confidence)
    3. Top-5 token probabilities
    4. KL divergence from uniform distribution (another measure of certainty)
    
    Args:
        batch_idx: Batch index for data generation
        split: Which split to process ('train' or 'test')
        num_samples: Number of next token samples to analyze for each prompt
        output_csv: Path to save results
        batch_size: Batch size for processing
        dataset: Dataset name
    '''
    questions = load_dynasor_dataset(dataset, split=split)
    print(f"Loaded {len(questions)} total questions")
    
    # Calculate batch bounds
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, len(questions))
    
    if start_idx >= len(questions):
        raise ValueError(f"Batch index {batch_idx} is too large for split {split} with {len(questions)} questions")
    
    # Get questions for this batch
    questions = questions[start_idx:end_idx]
    print(f"Loaded {len(questions)} batch questions")
    
    completed_question_ids = set()
    all_data = []
    
    # Check for existing results
    if os.path.exists(output_csv):
        try:
            print(f"Loading existing results from {output_csv}")
            existing_df = pd.read_csv(output_csv)
            # Count number of entries per question
            trace_counts = existing_df.groupby('question_id').size()
            # Get questions with entries completed
            completed_question_ids = set(trace_counts[trace_counts >= 1].index)
            all_data = existing_df.to_dict('records')
            print(f"Found {len(completed_question_ids)} completed questions")
        except Exception as e:
            print(f"Error loading existing results from {output_csv}: {e}")
    
    # Load the model and tokenizer
    model, tokenizer = get_model_and_tokenizer()
    
    # Prepare prompts
    batch_texts = [run.apply_chat_template(q["problem"], model.config._name_or_path) 
                   for q in questions if q['id'] not in completed_question_ids]
    
    if not batch_texts:  # Skip if all questions are completed
        print("All questions in batch already completed")
        return
    
    # Process in mini-batches for efficiency
    mini_batch_size = 8
    question_indices = [i for i, q in enumerate(questions) if q['id'] not in completed_question_ids]
    
    print(f"Computing entropy metrics for {len(batch_texts)} questions...")
    
    for i in range(0, len(batch_texts), mini_batch_size):
        batch_slice = batch_texts[i:i + mini_batch_size]
        current_indices = question_indices[i:i + mini_batch_size]
        
        # Tokenize inputs
        batch_inputs = tokenizer(batch_slice, return_tensors="pt", padding=True).to('cuda')
        
        # Get metrics for each prompt in the batch
        with torch.inference_mode():
            # Forward pass
            outputs = model(**batch_inputs)
            
            # For each example in the batch
            for j in range(len(batch_slice)):
                prompt = batch_slice[j]
                q_idx = current_indices[j]
                q_id = questions[q_idx]['id']
                
                # Get the logits for the last token in this example
                # Shape: [vocab_size]
                logits = outputs.logits[j, -1, :]
                
                # Convert logits to probabilities
                probs = F.softmax(logits, dim=0)
                
                # Calculate entropy
                entropy = calculate_entropy(probs)
                
                # Get max probability (confidence)
                max_prob, max_idx = torch.max(probs, dim=0)
                max_prob = max_prob.item()
                max_token = tokenizer.decode(max_idx.item())
                
                # Get top-5 token probabilities
                top_k = 5
                top_probs, top_indices = torch.topk(probs, top_k)
                top_probs = top_probs.cpu().numpy().tolist()
                top_tokens = [tokenizer.decode(idx.item()) for idx in top_indices]
                
                # Calculate KL divergence from uniform distribution
                vocab_size = probs.shape[0]
                uniform_probs = torch.ones_like(probs) / vocab_size
                # KL divergence: sum(p * log(p/q))
                kl_div = F.kl_div(torch.log(uniform_probs), probs, reduction='sum').item()
                
                # Calculate normalized entropy (0-1 scale where 1 is maximum uncertainty)
                # Maximum entropy is log2(vocab_size) for a uniform distribution
                max_possible_entropy = np.log2(vocab_size)
                normalized_entropy = entropy / max_possible_entropy
                
                # Store results
                all_data.append({
                    "dataset": dataset,
                    "question_id": q_id,
                    "question_text": prompt,
                    "split": split,
                    "entropy": entropy,
                    "normalized_entropy": normalized_entropy,
                    "max_prob": max_prob,
                    "max_token": max_token,
                    "top_k_probs": top_probs,
                    "top_k_tokens": top_tokens,
                    "kl_divergence": kl_div
                })
                
                print(f"Question {q_id}: Entropy={entropy:.4f}, Max Prob={max_prob:.4f}")
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
    
    # Save results
    df = pd.DataFrame(all_data)
    df.to_csv(output_csv, index=False)
    print(f"Data saved to {output_csv}")
    
    # Clean up
    del model, tokenizer
    torch.cuda.empty_cache()
    import gc
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Generate entropy-based predictors for reasoning traces")
    parser.add_argument("--batch_idx", type=int, required=True, help="Batch index for data generation")
    parser.add_argument("--split", type=str, choices=['train', 'test'], help="Which split to process")
    parser.add_argument("--dataset", type=str, default='gsm8k', choices=['gsm8k', 'math500', 'numina', 'amc23', 'aime24', 'GPQADiamond'], help="Which dataset to use")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to use when calculating entropy")
    
    args = parser.parse_args()
    
    STEM = "/n/netscratch/gershman_lab/Lab/amuppidi/"
    
    # Use the entropy predictor directory
    pred_dir = "entropy"
    csv_file = f"data/{args.dataset}_results/{pred_dir}/{args.dataset}_entropy_{args.split}_{args.batch_idx}.csv"
    
    # Create the directory if it doesn't exist
    full_dir = os.path.dirname(STEM + 'reasoning_scheduling_new_orig/' + csv_file)
    os.makedirs(full_dir, exist_ok=True)
    output_csv = STEM + 'reasoning_scheduling_new_orig/' + csv_file
    
    if args.batch_idx is None:
        parser.error("--batch_idx is required")
    if args.split is None:
        parser.error("--split is required")
    
    print(f"Generating entropy data for batch {args.batch_idx} of split {args.split} for dataset {args.dataset}")
    generate_data_entropy(
        batch_idx=args.batch_idx, 
        split=args.split, 
        output_csv=output_csv,
        dataset=args.dataset,
        num_samples=args.num_samples
    )


if __name__ == "__main__":
    main()