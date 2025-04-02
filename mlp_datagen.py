import argparse
import re
import numpy as np
import pandas as pd
import os

import torch
import torch.nn as nn
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
        #from accelerate import dispatch_model
        #model = dispatch_model(model, device_map="balanced")
        model.eval()
        #model.gradient_checkpointing_enable()
    except Exception as e:
        raise RuntimeError(f"Error loading model {model_name}: {e}")
    return model, tokenizer

def get_hidden_layer_index(layer_position, total_layers):
    """
    Determine the index of the hidden layer based on the specified position.
    
    Args:
        layer_position (str): 'first', 'middle', or 'last'
        total_layers (int): Total number of hidden layers in the model
    
    Returns:
        int: Index of the hidden layer to use
    """
    if layer_position == 'first':
        return 0
    elif layer_position == 'middle':
        return total_layers // 2
    elif layer_position == 'last':
        return -1
    else:
        # If a specific index is provided
        try:
            idx = int(layer_position)
            if idx < 0 or idx >= total_layers:
                print(f"Warning: Layer index {idx} out of bounds. Using last layer.")
                return -1
            return idx
        except ValueError:
            print(f"Warning: Invalid layer position '{layer_position}'. Using last layer.")
            return -1

def generate_data_X(batch_idx, split='train', num_traces=100, W=16, S=256, output_csv='gsm8k_results.csv', 
                   batch_size=100, dataset='gsm8k', hidden_layer='last'):
    '''
    This function generates the X data, which is the hidden states of the model.
    This script is MUCH cheaper to run than generate_data_Y because it doesn't require generating any reasoning traces.
    This function should be modified to generate predictors other than the hidden states if we are sweeping over multiple predictors.
    
    Args:
        hidden_layer (str): Which hidden layer to use as X data. Options: 'first', 'middle', 'last', or a specific index.
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
    '''
    if os.path.exists(output_csv):
        try:
            print(f"Loading existing results from {output_csv}")
            existing_df = pd.read_csv(output_csv)
            # Count number of traces per question
            trace_counts = existing_df.groupby('question_id').size()
            # Get questions with all traces completed
            completed_question_ids = set(trace_counts[trace_counts >= 1].index)
            all_data = existing_df.to_dict('records')
            print(f"Found {len(completed_question_ids)} completed questions")
        except Exception as e:
            print(f"Error loading existing results from {output_csv}: {e}")
    '''
    # Load the Qwen model and tokenizer
    model, tokenizer = get_model_and_tokenizer()
    
    # Get total number of hidden layers
    # For most models, can be determined from config
    total_hidden_layers = len(model.config.hidden_sizes) if hasattr(model.config, 'hidden_sizes') else model.config.num_hidden_layers
    # Get the index of the hidden layer to use
    hidden_layer_idx = get_hidden_layer_index(hidden_layer, total_hidden_layers)
    print(f"Using hidden layer at index {hidden_layer_idx} out of {total_hidden_layers} total layers")

    # Pre-compute hidden states for all questions in batch at once
    print("Computing hidden states for all questions in batch...")
    batch_texts = [run.apply_chat_template(q["problem"], model.config._name_or_path) for q in questions if q['id'] not in completed_question_ids]
    if not batch_texts:  # Skip if all questions are completed
        print("All questions in batch already completed")
        return
    
    # Compute hidden states
    # Process in batches of 16
    batch_size = 16
    all_hidden_states = []
    
    for i in range(0, len(batch_texts), batch_size):
        batch_slice = batch_texts[i:i + batch_size]
        batch_inputs = tokenizer(batch_slice, return_tensors="pt", padding=True).to('cuda')
        
        with torch.inference_mode():
            batch_outputs = model(**batch_inputs, output_hidden_states=True)
            # Get hidden states for this batch from the specified layer
            hidden_states = batch_outputs.hidden_states[hidden_layer_idx][:, -1, :].detach()  # Shape: [batch_size, hidden_dim]
            all_hidden_states.append(hidden_states)
            
        # Clear CUDA cache after each batch
        torch.cuda.empty_cache()
    
    # Concatenate all batches
    batch_hidden_states = torch.cat(all_hidden_states, dim=0)  # Shape: [total_size, hidden_dim]
    
    print(f"Finished computing hidden states for all questions in batch")
    hidden_dim = batch_hidden_states.shape[1]
    # Assert hidden states have correct dimensions
    assert hidden_dim > 0, f"Hidden state dimension is {hidden_dim}, which is invalid"
    print(f"Hidden state dimension is {hidden_dim}")
    
    # Create mapping from question to its hidden state
    hidden_states_map = {}
    batch_texts_map = {}
    current_idx = 0
    for q in questions:
        if q['id'] not in completed_question_ids:
            hidden_states_map[q['id']] = batch_hidden_states[current_idx]
            batch_texts_map[q['id']] = batch_texts[current_idx]
            current_idx += 1
    
    # Create the dataframe data
    for i,(qid, hidden_state) in enumerate(hidden_states_map.items()):
        all_data.append({
                "dataset": dataset,
                "question_id": qid,
                "question_text": batch_texts_map[qid],
                "split": split,
                "hidden_state": hidden_state.cpu().numpy().tolist(),  # Only convert to CPU/numpy when storing
                # TODO: add other predictors here! 
        })
    
    del model, tokenizer
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    # Final save
    df = pd.DataFrame(all_data)
    df.to_csv(output_csv, index=False)
    print(f"Data saved to {output_csv}")
    
def generate_data_Y(batch_idx, split='train', num_traces=100, W=16, S=256, output_csv='gsm8k_results.csv', batch_size=100, dataset='gsm8k'):
    '''
    This function generates the Y data, which is the early stopping correct proportions.
    It doesn't generate any of the predictors.
    This script is VERY expensive to run because it requires generating 100 reasoning traces for each question,
    and for each reasoning trace, probing the model with the probe string and extracting an answer every W tokens.
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
    if os.path.exists(output_csv):
        try:
            print(f"Loading existing results from {output_csv}")
            existing_df = pd.read_csv(output_csv)
            # Count number of traces per question
            trace_counts = existing_df.groupby('question_id').size()
            # Get questions with all traces completed
            completed_question_ids = set(trace_counts[trace_counts >= 1].index)
            all_data = existing_df.to_dict('records')
            print(f"Found {len(completed_question_ids)} completed questions")
        except Exception as e:
            print(f"Error loading existing results from {output_csv}: {e}")
    
    # Load the Qwen model and tokenizer
    model, tokenizer = get_model_and_tokenizer()
    # Move model to GPU once
    model = model.to('cuda')
    # move model to half-point precision to avoid OOM-ing
    # model = model.half()  # Converts all floating point parameters to float16

    print(f"Starting data generation for batch {batch_idx} of split {split}...")
    for problem_id, question in enumerate(questions):
        qid = question['id']
        
        # Skip if question is already completed
        if qid in completed_question_ids:
            print(f"Skipping completed question {qid}")
            continue
        prompt = run.apply_chat_template(question["problem"], model.config._name_or_path)
        target = question["answer"]
        probe="... Oh, I suddenly got the answer to the whole problem, **Final Answer**\n\n\\[ \\boxed{"
        token_budgets = list(range(W, S + 1, W))
        print(f"Executing question {problem_id} with token budgets {token_budgets}")
        print(f"Prompt: {prompt}")
        print(f"Target: {target}")
        
        # Run execute_question_reuse 10 times with 10 trials each
        all_round_results = []
        for run_idx in range(10):
            _, round_results = execute_question_reuse(
                model,
                prompt,
                target,
                max_tokens=token_budgets,
                probe=probe,
                probe_tokens=10,
                num_trials=10,
                problem_id=problem_id,
                output_dir=None,
                top_p=0.95,
                temperature=0.6,
                tokenizer=tokenizer,
            )
            all_round_results.append(round_results)

        # For each token budget, average the correct proportions across the 10 runs
        early_stop_correct_proportions = []
        for token_idx in range(len(token_budgets)):
            # Get results for this token budget from each run
            run_results = [sorted(results, key=lambda x: x["max_tokens"])[token_idx] 
                         for results in all_round_results]
            # Average the correct proportions
            avg_proportion = np.mean([sum(r["is_corrects"])/len(r["is_corrects"]) 
                                    for r in run_results])
            early_stop_correct_proportions.append(np.round(avg_proportion, 2))
        
        all_data.append({
                "dataset": dataset,
                "question_id": qid,
                "question_text": prompt,
                "split": split,
                "early_stop_correct_proportions": early_stop_correct_proportions,
        })
        print(f"Early stop correct proportions: {early_stop_correct_proportions}")
        
        
        # Save intermediate results after each question
        df = pd.DataFrame(all_data)
        df.to_csv(output_csv, index=False)
        print(f"Saved intermediate results to {output_csv}")
    
    # Final save (though this should be the same as the last intermediate save)
    df = pd.DataFrame(all_data)
    df.to_csv(output_csv, index=False)
    print(f"Data saved to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="MLP test experiment for reasoning traces")
    parser.add_argument("--batch_idx", type=int, required=True, help="Batch index for data generation (required with --generate)")
    parser.add_argument("--split", type=str, choices=['train', 'test'], help="Which split to process (required with --generate)")
    parser.add_argument("--dataset", type=str, default='gsm8k', choices=['gsm8k', 'math500', 'numina', 'amc23', 'aime24', 'GPQADiamond'], help="Which dataset to use")
    parser.add_argument("--S", type=int, default=256, help="Maximum number of new tokens")
    parser.add_argument("--generate-X-data", type=str, default="False", choices=["True","False"], help="Generate X data")
    parser.add_argument("--generate-Y-data", type=str, default="False", choices=["True","False"], help="Generate Y data")
    parser.add_argument("--hidden-layer", type=str, default="last", 
                        help="Which hidden layer to use for X data. Options: 'first', 'middle', 'last', or specific index")
    
    args = parser.parse_args()
    print("generate_X_data: ", args.generate_X_data)
    print("generate_Y_data: ", args.generate_Y_data)
    print("hidden_layer: ", args.hidden_layer)
    
    # STEM="/n/netscratch/gershman_lab/Lab/amuppidi/"
    from constants import STEM
    #STEM="/n/netscratch/dwork_lab/Lab/katrina"
    
    # Include hidden layer information in the directory name only
    layer_dir = f"layer_{args.hidden_layer}" if args.generate_X_data == "True" else ""
    
    csv_file_X = f"data/{args.dataset}_results/{layer_dir}/{args.dataset}_X_{args.split}_{args.batch_idx}.csv"
    csv_file_Y = f"data/{args.dataset}_results/{args.dataset}_Y_{args.split}_{args.batch_idx}.csv"
    
    # Create the directory if it doesn't exist
    if args.generate_X_data == "True":
        os.makedirs(os.path.dirname(STEM + 'reasoning_scheduling_new_orig/' + csv_file_X), exist_ok=True)
    
    csv_file_X = STEM + 'reasoning_scheduling_new_orig/' + csv_file_X
    csv_file_Y = STEM + 'reasoning_scheduling_new/' + csv_file_Y

    if args.batch_idx is None:
        parser.error("--batch_idx is required when using --generate")
    if args.split is None:
        parser.error("--split is required when using --generate")
    if args.generate_X_data=="True":
        print(f"Generating X data for batch {args.batch_idx} of split {args.split} for dataset {args.dataset} using hidden layer {args.hidden_layer}")
        generate_data_X(batch_idx=args.batch_idx, split=args.split, output_csv=csv_file_X, 
                       dataset=args.dataset, S=args.S, hidden_layer=args.hidden_layer)
    if args.generate_Y_data=="True":
        print(f"Generating Y data for batch {args.batch_idx} of split {args.split} for dataset {args.dataset}")
        generate_data_Y(batch_idx=args.batch_idx, split=args.split, output_csv=csv_file_Y, dataset=args.dataset, S=args.S)


if __name__ == "__main__":
    main()