'''
# Proof of concept: can MLP on hidden states predict required size? 
Model: distilled Qwen 1.5B model. Datasets: "amc23", "aime24", "GPQADiamond", "math500", "gsm8k". $W=16$. Max new tokens $S=4096$.    
MLP predictor: $\mathbb{R}^{1536}$ -> H=128 -> $\mathbb{R}^{S/W}.  
For each question $q$, sample 100 reasoning traces. For each reasoning trace, test
post-hoc whether we would have generated the answer if we had stopped generating tokens after W, 2W, 3W,...,S tokens, inserted the suffix "...Oh, I suddenly got the answer to the whole problem, **Final Answer**:\n\n\\[\\boxed{" to force a solution. Compute the proportion of the 100 samples for which the early terminated answer would have been correct. This yields a $W/S$ element vector of proportions in range [0,1]. If a sampled reasoning trace only has, for instance, 100 tokens, then the early stopping probability after 200 tokens is just the proportion after 100 tokens elapsed.   

On the train/test set, compute MSE as well as the pearson correlation between predicted and actual proportions.  Ideally not overfit within GSM8K. If this goes well, test correlation as well on another dataset, maybe Math500 (secondary math domain which might require secondary model as evaluator) or MMLU (MCQ domain). In the best of all worlds the performance generalizes.   

1) Get GSM8K questions from train and test split and load the distill qwen 1.5b model.  
2) Sample 100 reasoning traces up to S=4096 new tokens per question, generated with temperature 0.6. For each sampled trace, generate the answer if we had stopped after W,2W,3W,... tokens. For each sampled trace, store a list of the W/S generated answers, a W/S length list of 0/1 indicating whether a given answer was correct, and the proportion of answers that were correct.  
3) Generate a pandas dataframe with 1 rows per question in GSM8K. The columns should be the GSM8K question id, the question text itself, whether the question is from the train or test split, the hidden state from the model from the first forward pass of the question (before any reasoning/answer tokens are generated), the reasoning trace id, the generated reasoning trace, the list of W/S generated answers if we had stopped early after any point, inserted the answer suffix and generated and extracted a numerical answer, the list of W/S extracted answers, and the corresponding list of W/S proportions across all 100 traces for this query that were correct. Note that within the group of 100 rows corresponding to a single GSM8K question, the question id, question text itself, hidden state, and list of proportions after early stopping after each point will all be shared. Save the computed data to a csv.      
4) Write separate python code that loads the pandas dataframe from csv after it has been computed.  The new python code should construct an MLP that takes in the 1536 dimension hidden states as input, has one hidden layer with dimension 128, and outputs a W/S dimensional vector. It should be trained on a dataset with one entry per question in the GSM8K dataset, where the input is the hidden state and the output is the W/S dimensional list of proportions. The code should report the MSE on both the train and the test set, and it should also output data about, for each W/S possible early stopping position, the pearson correlation between the predicted value and the actual value.  It should report MSE both across the entire predicted and actual output and across each individual entry of the W/S output individually.  

Data quantities
    GSM8K: ~13XX test, 74XX train
    numina: 859K train, 100 test
    amc23: 50 test
    aime24: 90 test
'''

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
        answer_field = sample["answer"]
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
        model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True, use_flash_attention_2=True, cache_dir="/n/holylabs/LABS/dwork_lab/Everyone/cache/transformers")
        model.eval()
        model.gradient_checkpointing_enable()
    except Exception as e:
        raise RuntimeError(f"Error loading model {model_name}: {e}")
    return model, tokenizer

def generate_data_X(batch_idx, split='train', num_traces=100, W=16, S=256, output_csv='gsm8k_results.csv', batch_size=100, dataset='gsm8k'):
    '''
    This function generates the X data, which is the hidden states of the model.
    This script is MUCH cheaper to run than generate_data_Y because it doesn't require generating any reasoning traces.
    This function should be modified to generate predictors other than the hidden states if we are sweeping over multiple predictors.
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

    # Pre-compute hidden states for all questions in batch at once
    print("Computing hidden states for all questions in batch...")
    print(questions[0].keys())
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
            # Get hidden states for this batch
            hidden_states = batch_outputs.hidden_states[-1][:, -1, :].detach()  # Shape: [batch_size, 1536]
            all_hidden_states.append(hidden_states)
            
        # Clear CUDA cache after each batch
        torch.cuda.empty_cache()
    
    # Concatenate all batches
    batch_hidden_states = torch.cat(all_hidden_states, dim=0)  # Shape: [total_size, 1536]
    
    print(f"Finished computing hidden states for all questions in batch")
    # Assert hidden states have correct dimensions
    assert batch_hidden_states.shape[1] == 1536, f"Hidden state dimension is {batch_hidden_states.shape[1]}, expected 1536"
    
    # Create mapping from question to its hidden state
    hidden_states_map = {}
    current_idx = 0
    for q in questions:
        if q['id'] not in completed_question_ids:
            hidden_states_map[q['id']] = batch_hidden_states[current_idx]
            current_idx += 1
    
    # Create the dataframe data
    for qid, hidden_state in hidden_states_map.items():
        all_data.append({
                "dataset": dataset,
                "question_id": qid,
                "split": split,
                "hidden_state": hidden_state.cpu().numpy().tolist(),  # Only convert to CPU/numpy when storing
                # TODO: add other predictors here! 
        })
    
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
        prop_correct, round_results_arr = execute_question_reuse(
            model,
            prompt,
            target,
            max_tokens=token_budgets,
            probe=probe,
            probe_tokens=10,
            num_trials=100,
            problem_id=problem_id,
            output_dir=None,
            top_p=0.95,
            temperature=0.6,
            tokenizer=tokenizer,  # Pass tokenizer to execute_question_reuse
        )
        
        early_stop_correct_proportions = [sum(round_results["is_corrects"])/len(round_results["is_corrects"]) for round_results in sorted(round_results_arr, key=lambda x: x["max_tokens"])]
        
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
    parser.add_argument("--dataset", type=str, default='gsm8k', choices=['gsm8k', 'math500', 'numina'], help="Which dataset to use")
    parser.add_argument("--S", type=int, default=256, help="Maximum number of new tokens")
    parser.add_argument("--generate-X-data", type=str, default="False", choices=["True","False"], help="Generate X data")
    parser.add_argument("--generate-Y-data", type=str, default="False", choices=["True","False"], help="Generate Y data")
    
    args = parser.parse_args()
    print("generate_X_data: ", args.generate_X_data)
    print("generate_Y_data: ", args.generate_Y_data)
    
    STEM="/n/netscratch/dwork_lab/Lab/katrina/reasoning_scheduling_new/"
    csv_file_X = f"data/{args.dataset}_results/{args.dataset}_X_{args.split}_{args.batch_idx}.csv"
    csv_file_Y = f"data/{args.dataset}_results/{args.dataset}_Y_{args.split}_{args.batch_idx}.csv"
    csv_file_X = STEM+csv_file_X
    csv_file_Y = STEM+csv_file_Y
    

    if args.batch_idx is None:
        parser.error("--batch_idx is required when using --generate")
    if args.split is None:
        parser.error("--split is required when using --generate")
    if args.generate_X_data=="True":
        print(f"Generating X data for batch {args.batch_idx} of split {args.split} for dataset {args.dataset}")
        generate_data_X(batch_idx=args.batch_idx, split=args.split, output_csv=csv_file_X, dataset=args.dataset, S=args.S)
    if args.generate_Y_data=="True":
        print(f"Generating Y data for batch {args.batch_idx} of split {args.split} for dataset {args.dataset}")
        generate_data_Y(batch_idx=args.batch_idx, split=args.split, output_csv=csv_file_Y, dataset=args.dataset, S=args.S)


if __name__ == "__main__":
    main()