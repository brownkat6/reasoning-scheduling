'''
# Proof of concept: can MLP on hidden states predict required size? 
Model: distilled Qwen 1.5B model. Dataset: "amc23", "aime24", "GPQADiamond", "math500", "gsm8k". $W=16$. Max new tokens $S=4096$.    
MLP predictor: $\mathbb{R}^{1536}$ -> H=128 -> $\mathbb{R}^{S/W}.  
For each question $q$ in GSM8K, sample 100 reasoning traces. For each reasoning trace, test
post-hoc whether we would have generated the answer if we had stopped generating tokens after W, 2W, 3W,...,S tokens, inserted the suffix "...Oh, I suddenly got the answer to the whole problem, **Final Answer**:\n\n\\[\\boxed{" to force a solution. Compute the proportion of the 100 samples for which the early terminated answer would have been correct. This yields a $W/S$ element vector of proportions in range [0,1]. If a sampled reasoning trace only has, for instance, 100 tokens, then the early stopping probability after 200 tokens is just the proportion after 100 tokens elapsed.   

On the train/test set, compute MSE as well as the pearson correlation between predicted and actual proportions.  Ideally not overfit within GSM8K. If this goes well, test correlation as well on another dataset, maybe Math500 (secondary math domain which might require secondary model as evaluator) or MMLU (MCQ domain). In the best of all worlds the performance generalizes.   

1) Get GSM8K questions from train and test split and load the distill qwen 1.5b model.  
2) Sample 100 reasoning traces up to S=4096 new tokens per question, generated with temperature 0.6. For each sampled trace, generate the answer if we had stopped after W,2W,3W,... tokens. For each sampled trace, store a list of the W/S generated answers, a W/S length list of 0/1 indicating whether a given answer was correct, and the proportion of answers that were correct.  
3) Generate a pandas dataframe with 1 rows per question in GSM8K. The columns should be the GSM8K question id, the question text itself, whether the question is from the train or test split, the hidden state from the model from the first forward pass of the question (before any reasoning/answer tokens are generated), the reasoning trace id, the generated reasoning trace, the list of W/S generated answers if we had stopped early after any point, inserted the answer suffix and generated and extracted a numerical answer, the list of W/S extracted answers, and the corresponding list of W/S proportions across all 100 traces for this query that were correct. Note that within the group of 100 rows corresponding to a single GSM8K question, the question id, question text itself, hidden state, and list of proportions after early stopping after each point will all be shared. Save the computed data to a csv.      
4) Write separate python code that loads the pandas dataframe from csv after it has been computed.  The new python code should construct an MLP that takes in the 1536 dimension hidden states as input, has one hidden layer with dimension 128, and outputs a W/S dimensional vector. It should be trained on a dataset with one entry per question in the GSM8K dataset, where the input is the hidden state and the output is the W/S dimensional list of proportions. The code should report the MSE on both the train and the test set, and it should also output data about, for each W/S possible early stopping position, the pearson correlation between the predicted value and the actual value.  It should report MSE both across the entire predicted and actual output and across each individual entry of the W/S output individually.  
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

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    raise ImportError("Transformers library not found. Please install via pip install transformers")

def load_dynasor_dataset(dataset_name):
    """
    Load a Dynasor dataset using the Dynasor utils.
    """
    assert dataset_name in ["amc23", "aime24", "GPQADiamond", "math500", "gsm8k"], f"Dataset {dataset_name} not supported"
    print(f"Loading dataset {dataset_name} with dynasor utils")
    data = utils.load_dataset(dataset_name)
    test_data = []
    for i, sample in enumerate(data):
        question = sample["problem"]
        answer_field = sample["answer"]
        ground_truth = answer_field.strip()
        test_data.append({"id": f"test_{i}", "problem": question, "answer": ground_truth})
    return test_data

def get_model_and_tokenizer():
    """
    Load the distilled Qwen 1.5B model and its tokenizer.
    Throws an error if the model cannot be loaded.
    """
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir="/n/holylabs/LABS/dwork_lab/Everyone/cache/transformers")
        model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True, cache_dir="/n/holylabs/LABS/dwork_lab/Everyone/cache/transformers")
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Error loading model {model_name}: {e}")
    return model, tokenizer

from Dynasor.dynasor.core.evaluator import math_equal

def generate_data(batch_idx, split='train', num_traces=100, W=16, S=256, output_csv='gsm8k_results.csv', batch_size=100, dataset='gsm8k'):
    questions = load_dynasor_dataset(dataset)
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
            print(existing_df.shape)
            print(existing_df.columns)
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
    
    
    print(f"Starting data generation for batch {batch_idx} of split {split}...")
    
    
    for problem_id, question in enumerate(questions):
        qid = question['id']
        
        # Skip if question is already completed
        if qid in completed_question_ids:
            print(f"Skipping completed question {qid}")
            continue
        prompt = run.apply_chat_template(question["problem"], model.config._name_or_path)
        target = question["answer"]
        print(f"Prompt: {prompt}")
        print(f"Target: {target}")
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
        
        #early_generated_answers = []
        #early_extracted_answers = []
        #for round_results in sorted(round_results_arr, key=lambda x: x["max_tokens"]):
        #    early_generated_answers.append(round_results["probe_prompts"])
        #    early_extracted_answers.append(round_results["probe_responses"])
        early_stop_correct_proportions = [sum(round_results["is_corrects"])/len(round_results["is_corrects"]) for round_results in sorted(round_results_arr, key=lambda x: x["max_tokens"])]
             
        all_data.append({
                "dataset": dataset,
                "question_id": qid,
                "question_text": prompt,
                "split": split,
                "hidden_state": hidden_states_map[qid].cpu().numpy().tolist(),  # Only convert to CPU/numpy when storing
                "trace_id": -1, # only 1 row per question
                "early_stop_correct_proportions": early_stop_correct_proportions,
                #"reasoning_trace": reasoning_trace,
                #"early_generated_answers": early_generated_answers,
                #"early_extracted_answers": early_extracted_answers
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

def train_mlp(train_data_dir='', train_split='train', train_dataset='gsm8k',
              test_data_dir='', test_split='test', test_dataset='gsm8k',
              num_epochs=20, batch_size=4, learning_rate=1e-3):
    """
    Train an MLP to predict the early stopping correctness proportions from the hidden state.
    Allows training on one dataset/split and testing on another.
    
    Args:
        train_data_dir: Directory containing the training data files
        train_split: Split to use for training ('train' or 'test')
        train_dataset: Dataset name for training data
        test_data_dir: Directory containing the test data files
        test_split: Split to use for testing ('train' or 'test')
        test_dataset: Dataset name for test data
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
    """
    # Validate input directories
    if not os.path.exists(train_data_dir):
        raise ValueError(f"Training data directory {train_data_dir} not found.")
    if not os.path.exists(test_data_dir):
        raise ValueError(f"Test data directory {test_data_dir} not found.")
    
    train_dataset_name=train_dataset
    test_dataset_name=test_dataset
    
    # Define grouped file paths
    train_grouped_file = os.path.join(train_data_dir, f"{train_dataset}_results_{train_split}_grouped.csv")
    test_grouped_file = os.path.join(test_data_dir, f"{test_dataset}_results_{test_split}_grouped.csv")
    
    # Process training data if grouped file doesn't exist
    if not os.path.exists(train_grouped_file):
        print("Loading and combining training data files...")
        train_files = [os.path.join(train_data_dir, f"{train_dataset}_results_{train_split}_{i}.csv") for i in range(100)]
        train_files = [f for f in train_files if os.path.exists(f)]
        
        if not train_files:
            raise ValueError(f"No training data files found in {train_data_dir}")
        
        train_grouped_dfs = []
        for f in train_files:
            try:
                df = pd.read_csv(f)
                grouped = df.groupby(['question_id', 'split', 'question_text']).first().reset_index()
                train_grouped_dfs.append(grouped)
            except Exception as e:
                print(f"Error loading {f}: {e}")
        train_grouped = pd.concat(train_grouped_dfs, ignore_index=True)
        train_grouped.to_csv(train_grouped_file, index=False)
        print(f"Saved grouped training data with {len(train_grouped)} questions")
    else:
        print("Loading pre-grouped training data...")
        train_grouped = pd.read_csv(train_grouped_file)
    
    # Process test data if grouped file doesn't exist
    if not os.path.exists(test_grouped_file):
        print("Loading and combining test data files...")
        test_files = [os.path.join(test_data_dir, f"{test_dataset}_results_{test_split}_{i}.csv") for i in range(100)]
        test_files = [f for f in test_files if os.path.exists(f)]
        
        if not test_files:
            raise ValueError(f"No test data files found in {test_data_dir}")
        
        test_grouped_dfs = []
        for f in test_files:
            try:
                df = pd.read_csv(f)
                grouped = df.groupby(['question_id', 'split', 'question_text']).first().reset_index()
                test_grouped_dfs.append(grouped)
            except Exception as e:
                print(f"Error loading {f}: {e}")
        test_grouped = pd.concat(test_grouped_dfs, ignore_index=True)
        test_grouped.to_csv(test_grouped_file, index=False)
        print(f"Saved grouped test data with {len(test_grouped)} questions")
    else:
        print("Loading pre-grouped test data...")
        test_grouped = pd.read_csv(test_grouped_file)

    # Parse lists from string representation
    import ast
    def parse_list(x):
        try:
            return ast.literal_eval(x)
        except Exception:
            return x

    train_grouped['hidden_state'] = train_grouped['hidden_state'].apply(parse_list)
    train_grouped['early_stop_correct_proportions'] = train_grouped['early_stop_correct_proportions'].apply(parse_list)
    test_grouped['hidden_state'] = test_grouped['hidden_state'].apply(parse_list)
    test_grouped['early_stop_correct_proportions'] = test_grouped['early_stop_correct_proportions'].apply(parse_list)

    # Prepare training data
    X_train = np.vstack(train_grouped['hidden_state'].values)
    Y_train = np.vstack(train_grouped['early_stop_correct_proportions'].values)
    
    # Prepare test data
    X_test = np.vstack(test_grouped['hidden_state'].values)
    Y_test = np.vstack(test_grouped['early_stop_correct_proportions'].values)

    print(f"Training questions: {X_train.shape[0]}, Testing questions: {X_test.shape[0]}")

    from mlp import MLP

    model_mlp = MLP(input_dim=1536, hidden_dim=256, output_dim=Y_train.shape[1]).to('cuda')  # Move model to CUDA
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_mlp.parameters(), lr=learning_rate)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to('cuda')
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).to('cuda')
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to('cuda')
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32).to('cuda')
    
    print(f"Training data shape: {X_train_tensor.shape}, {Y_train_tensor.shape}")
    print(f"Testing data shape: {X_test_tensor.shape}, {Y_test_tensor.shape}")
    
    # if Y_train has shape (N, D) and Y_test has shape (N, F) where F>D, truncate Y_test to have shape (N,D)
    if Y_test_tensor.shape[1] > Y_train_tensor.shape[1]:
        Y_test_tensor = Y_test_tensor[:, :Y_train_tensor.shape[1]]
        print(f"Truncated testing data shape: {Y_test_tensor.shape}")

    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, Y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print("Starting MLP training...")
    for epoch in range(num_epochs):
        model_mlp.train()
        running_loss = 0.0
        for batch_X, batch_Y in train_loader:
            # No need to move batches to cuda since the dataset tensors are already on cuda
            optimizer.zero_grad()
            outputs = model_mlp(batch_X)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_X.size(0)
        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    model_mlp.eval()
    with torch.no_grad():
        train_pred = model_mlp(X_train_tensor).cpu().numpy()  # Only move to CPU for final numpy conversion
        test_pred = model_mlp(X_test_tensor).cpu().numpy()  # Only move to CPU for final numpy conversion

    # Move Y tensors to CPU only when needed for numpy operations
    Y_train_np = Y_train_tensor.cpu().numpy()
    Y_test_np = Y_test_tensor.cpu().numpy()

    # Calculate mean predictions from both training and test sets for each position
    train_means = np.mean(Y_train_np, axis=0)  # Shape: (num_positions,)
    test_means = np.mean(Y_test_np, axis=0)    # Shape: (num_positions,)
    
    # Create baseline predictions using train means
    baseline_train_pred = np.tile(train_means, (Y_train_np.shape[0], 1))  # Shape: (train_size, num_positions)
    baseline_test_pred = np.tile(train_means, (Y_test_np.shape[0], 1))    # Shape: (test_size, num_positions)
    
    # Create baseline predictions using test means
    baseline_train_pred_test_means = np.tile(test_means, (Y_train_np.shape[0], 1))  # Shape: (train_size, num_positions)
    baseline_test_pred_test_means = np.tile(test_means, (Y_test_np.shape[0], 1))    # Shape: (test_size, num_positions)
    
    # Calculate MSE for model and both baselines
    mse_train_overall = np.mean((train_pred - Y_train_np)**2)
    mse_test_overall = np.mean((test_pred - Y_test_np)**2)
    
    # MSE for train means baseline
    baseline_mse_train = np.mean((baseline_train_pred - Y_train_np)**2)
    baseline_mse_test = np.mean((baseline_test_pred - Y_test_np)**2)
    
    # MSE for test means baseline
    baseline_mse_train_test_means = np.mean((baseline_train_pred_test_means - Y_train_np)**2)
    baseline_mse_test_test_means = np.mean((baseline_test_pred_test_means - Y_test_np)**2)
    
    print("\nModel Performance:")
    print(f"Overall MSE on train: {mse_train_overall:.4f}")
    print(f"Overall MSE on test: {mse_test_overall:.4f}")
    
    print("\nBaseline Performance (Predicting Train Means):")
    print(f"Overall MSE on train: {baseline_mse_train:.4f}")
    print(f"Overall MSE on test: {baseline_mse_test:.4f}")
    
    print("\nBaseline Performance (Predicting Test Means):")
    print(f"Overall MSE on train: {baseline_mse_train_test_means:.4f}")
    print(f"Overall MSE on test: {baseline_mse_test_test_means:.4f}")
    
    # Calculate relative improvement over train means baseline
    train_improvement = ((baseline_mse_train - mse_train_overall) / baseline_mse_train) * 100
    test_improvement = ((baseline_mse_test - mse_test_overall) / baseline_mse_test) * 100
    
    # Calculate relative improvement over test means baseline
    train_improvement_test_means = ((baseline_mse_train_test_means - mse_train_overall) / baseline_mse_train_test_means) * 100
    test_improvement_test_means = ((baseline_mse_test_test_means - mse_test_overall) / baseline_mse_test_test_means) * 100
    
    print("\nRelative Improvement Over Train Means Baseline:")
    print(f"Train improvement: {train_improvement:.1f}%")
    print(f"Test improvement: {test_improvement:.1f}%")
    
    print("\nRelative Improvement Over Test Means Baseline:")
    print(f"Train improvement: {train_improvement_test_means:.1f}%")
    print(f"Test improvement: {test_improvement_test_means:.1f}%")

    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save the model
    model_filename = f'models/mlp_{train_dataset_name}_{train_split}.pt'
    torch.save({
        'model_state_dict': model_mlp.state_dict(),
        'train_means': train_means,  # Save training means for future reference
        'config': {
            'input_dim': 1536,
            'hidden_dim': 128,
            'output_dim': Y_train.shape[1],
            'train_dataset': train_dataset,
            'train_split': train_split,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }
    }, model_filename)
    print(f"\nModel saved to {model_filename}")

    # Also print out the means themselves for comparison
    print("\nMean Values:")
    print("Position  Train_Mean  Test_Mean   Diff")
    print("-" * 45)
    for i, (train_mean, test_mean) in enumerate(zip(train_means, test_means)):
        diff = abs(train_mean - test_mean)
        print(f"{i+1:8d}  {train_mean:.4f}     {test_mean:.4f}     {diff:.4f}")

    mse_train_individual = np.mean((train_pred - Y_train_np)**2, axis=0)
    mse_test_individual = np.mean((test_pred - Y_test_np)**2, axis=0)

    pearson_train = [pearsonr(train_pred[:, i], Y_train_np[:, i])[0] for i in range(Y_train_np.shape[1])]
    pearson_test = [pearsonr(test_pred[:, i], Y_test_np[:, i])[0] for i in range(Y_test_np.shape[1])]

    print("\nEarly stopping position-wise MSE and Pearson correlation (Train):")
    for i, (mse_val, p_val) in enumerate(zip(mse_train_individual, pearson_train)):
        print(f"Position {i+1}: MSE: {mse_val:.4f}, Pearson: {p_val:.4f}")

    print("\nEarly stopping position-wise MSE and Pearson correlation (Test):")
    for i, (mse_val, p_val) in enumerate(zip(mse_test_individual, pearson_test)):
        print(f"Position {i+1}: MSE: {mse_val:.4f}, Pearson: {p_val:.4f}")

    # Create figures directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)
    
    import matplotlib.pyplot as plt
    from scipy import stats
    
    # Calculate number of subplots needed (all positions + 1 for aggregate)
    n_positions = Y_train_np.shape[1]
    n_rows = (n_positions + 2) // 3  # 3 plots per row, +2 for ceiling division
    
    plt.figure(figsize=(15, 5*n_rows))
    
    # Plot for each position
    for i in range(n_positions):
        plt.subplot(n_rows, 3, i+1)
        
        # Train data
        slope_train, intercept_train, r_train, _, _ = stats.linregress(train_pred[:, i], Y_train_np[:, i])
        line_train = slope_train * np.array([0, 1]) + intercept_train
        
        # Test data
        slope_test, intercept_test, r_test, _, _ = stats.linregress(test_pred[:, i], Y_test_np[:, i])
        line_test = slope_test * np.array([0, 1]) + intercept_test
        
        plt.scatter(train_pred[:, i], Y_train_np[:, i], alpha=0.5, label='Train', color='blue')
        plt.scatter(test_pred[:, i], Y_test_np[:, i], alpha=0.5, label='Test', color='red')
        
        plt.plot([0, 1], line_train, color='blue', linestyle='--', alpha=0.8, label='Train fit')
        plt.plot([0, 1], line_test, color='red', linestyle='--', alpha=0.8, label='Test fit')
        plt.plot([0, 1], [0, 1], color='black', linestyle=':', alpha=0.5, label='Perfect')
        
        plt.xlabel('Predicted Probability')
        plt.ylabel('Actual Probability')
        plt.title(f'Position {i+1}\nTrain r={r_train:.3f}, Test r={r_test:.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Set axis limits
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
    
    # Plot for aggregate data
    plt.subplot(n_rows, 3, n_positions+1)
    
    # Flatten all predictions and actuals
    train_pred_flat = train_pred.flatten()
    test_pred_flat = test_pred.flatten()
    Y_train_flat = Y_train_np.flatten()
    Y_test_flat = Y_test_np.flatten()
    
    # Compute aggregate correlations
    slope_train, intercept_train, r_train, _, _ = stats.linregress(train_pred_flat, Y_train_flat)
    slope_test, intercept_test, r_test, _, _ = stats.linregress(test_pred_flat, Y_test_flat)
    
    line_train = slope_train * np.array([0, 1]) + intercept_train
    line_test = slope_test * np.array([0, 1]) + intercept_test
    
    plt.scatter(train_pred_flat, Y_train_flat, alpha=0.5, label='Train', color='blue')
    plt.scatter(test_pred_flat, Y_test_flat, alpha=0.5, label='Test', color='red')
    
    plt.plot([0, 1], line_train, color='blue', linestyle='--', alpha=0.8, label='Train fit')
    plt.plot([0, 1], line_test, color='red', linestyle='--', alpha=0.8, label='Test fit')
    plt.plot([0, 1], [0, 1], color='black', linestyle=':', alpha=0.5, label='Perfect')
    
    plt.xlabel('Predicted Probability')
    plt.ylabel('Actual Probability')
    plt.title(f'All Positions\nTrain r={r_train:.3f}, Test r={r_test:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    
    plt.tight_layout()
    figure_name = f'prediction_correlation_plots_{train_dataset_name}_{train_split}_to_{test_dataset_name}_{test_split}.png'
    plt.savefig(os.path.join('figures', figure_name), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to figures/{figure_name}")

    # Create a new figure for residuals normality plots
    plt.figure(figsize=(15, 5*n_rows))
    
    # Plot for each position
    for i in range(n_positions):
        plt.subplot(n_rows, 3, i+1)
        
        # Calculate residuals
        train_residuals = Y_train_np[:, i] - train_pred[:, i]
        test_residuals = Y_test_np[:, i] - test_pred[:, i]
        
        # Create Q-Q plots
        from scipy import stats
        
        # Train residuals
        sorted_train_residuals = np.sort(train_residuals)
        train_theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(train_residuals)))
        
        # Test residuals
        sorted_test_residuals = np.sort(test_residuals)
        test_theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(test_residuals)))
        
        # Plot Q-Q lines
        plt.plot(train_theoretical_quantiles, sorted_train_residuals, 'bo', alpha=0.5, markersize=2, label='Train')
        plt.plot(test_theoretical_quantiles, sorted_test_residuals, 'ro', alpha=0.5, markersize=2, label='Test')
        
        # Add reference line
        min_q = min(train_theoretical_quantiles.min(), test_theoretical_quantiles.min())
        max_q = max(train_theoretical_quantiles.max(), test_theoretical_quantiles.max())
        plt.plot([min_q, max_q], [min_q, max_q], 'k:', label='Normal')
        
        plt.xlabel('Theoretical Quantiles')
        plt.ylabel('Sample Quantiles')
        plt.title(f'Position {i+1} Q-Q Plot\nTrain skew={stats.skew(train_residuals):.3f}\nTest skew={stats.skew(test_residuals):.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Plot for aggregate residuals
    plt.subplot(n_rows, 3, n_positions+1)
    
    # Calculate aggregate residuals
    train_residuals_flat = Y_train_flat - train_pred_flat
    test_residuals_flat = Y_test_flat - test_pred_flat
    
    # Create Q-Q plots for aggregate data
    sorted_train_residuals = np.sort(train_residuals_flat)
    train_theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(train_residuals_flat)))
    
    sorted_test_residuals = np.sort(test_residuals_flat)
    test_theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(test_residuals_flat)))
    
    plt.plot(train_theoretical_quantiles, sorted_train_residuals, 'bo', alpha=0.5, markersize=2, label='Train')
    plt.plot(test_theoretical_quantiles, sorted_test_residuals, 'ro', alpha=0.5, markersize=2, label='Test')
    
    min_q = min(train_theoretical_quantiles.min(), test_theoretical_quantiles.min())
    max_q = max(train_theoretical_quantiles.max(), test_theoretical_quantiles.max())
    plt.plot([min_q, max_q], [min_q, max_q], 'k:', label='Normal')
    
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.title(f'All Positions Q-Q Plot\nTrain skew={stats.skew(train_residuals_flat):.3f}\nTest skew={stats.skew(test_residuals_flat):.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    residuals_figure_name = f'residuals_normality_plots_{train_dataset_name}_{train_split}_to_{test_dataset_name}_{test_split}.png'
    plt.savefig(os.path.join('figures', residuals_figure_name), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to figures/{residuals_figure_name}")


def main():
    parser = argparse.ArgumentParser(description="MLP test experiment for reasoning traces")
    parser.add_argument("--generate", action="store_true", help="Run data generation experiment")
    parser.add_argument("--train", action="store_true", help="Train the MLP on generated data")
    parser.add_argument("--batch_idx", type=int, help="Batch index for data generation (required with --generate)")
    parser.add_argument("--split", type=str, choices=['train', 'test'], help="Which split to process (required with --generate)")
    parser.add_argument("--csv_file", type=str, default="gsm8k_results.csv", help="CSV file to load/save generated data")
    parser.add_argument("--dataset", type=str, default='gsm8k', choices=['gsm8k', 'math500', 'numina'], help="Which dataset to use")
    parser.add_argument("--S", type=int, default=256, help="Maximum number of new tokens")
    
    # New arguments for refactored train_mlp
    parser.add_argument("--train_data_dir", type=str, help="Directory containing training data files")
    parser.add_argument("--train_split", type=str, default='train', choices=['train', 'test'], help="Split to use for training")
    parser.add_argument("--train_dataset", type=str, default='gsm8k', choices=['gsm8k', 'math500', 'numina'], help="Dataset for training")
    parser.add_argument("--test_data_dir", type=str, help="Directory containing test data files")
    parser.add_argument("--test_split", type=str, default='test', choices=['train', 'test'], help="Split to use for testing")
    parser.add_argument("--test_dataset", type=str, default='gsm8k', choices=['gsm8k', 'math500', 'numina'], help="Dataset for testing")
    
    args = parser.parse_args()
    
    STEM="/n/netscratch/dwork_lab/Lab/katrina/reasoning_scheduling_new/"
    csv_file = STEM+args.csv_file
    

    if args.generate:
        if args.batch_idx is None:
            parser.error("--batch_idx is required when using --generate")
        if args.split is None:
            parser.error("--split is required when using --generate")
        generate_data(batch_idx=args.batch_idx, split=args.split, output_csv=csv_file, dataset=args.dataset, S=args.S)
    elif args.train:
        if args.train_data_dir is None or args.test_data_dir is None:
            parser.error("--train_data_dir and --test_data_dir are required when using --train")
        train_data_dir = STEM+args.train_data_dir
        test_data_dir = STEM+args.test_data_dir
        train_mlp(
            train_data_dir=train_data_dir,
            train_split=args.train_split,
            train_dataset=args.train_dataset,
            test_data_dir=test_data_dir,
            test_split=args.test_split,
            test_dataset=args.test_dataset
        )
    else:
        print("Please specify --generate or --train.")


if __name__ == "__main__":
    main()