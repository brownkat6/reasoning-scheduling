'''
# Proof of concept: can MLP on hidden states predict required size? 
Model: distilled Qwen 1.5B model. Dataset: GSM8K. $W=16$. Max new tokens $S=4096$.    
MLP predictor: $\mathbb{R}^{1024}$ -> H=128 -> $\mathbb{R}^{S/W}.  
For each question $q$ in GSM8K, sample 100 reasoning traces. For each reasoning trace, test
post-hoc whether we would have generated the answer if we had stopped generating tokens after W, 2W, 3W,...,S tokens, inserted the suffix "...Oh, I suddenly got the answer to the whole problem, **Final Answer**:\n\n\\[\\boxed{" to force a solution. Compute the proportion of the 100 samples for which the early terminated answer would have been correct. This yields a $W/S$ element vector of proportions in range [0,1]. If a sampled reasoning trace only has, for instance, 100 tokens, then the early stopping probability after 200 tokens is just the proportion after 100 tokens elapsed.   

On the train/test set, compute MSE as well as the pearson correlation between predicted and actual proportions.  Ideally not overfit within GSM8K. If this goes well, test correlation as well on another dataset, maybe Math500 (secondary math domain which might require secondary model as evaluator) or MMLU (MCQ domain). In the best of all worlds the performance generalizes.   

1) Get GSM8K questions from train and test split and load the distill qwen 1.5b model.  
2) Sample 100 reasoning traces up to S=4096 new tokens per question, generated with temperature 0.6. For each sampled trace, generate the answer if we had stopped after W,2W,3W,... tokens. For each sampled trace, store a list of the W/S generated answers, a W/S length list of 0/1 indicating whether a given answer was correct, and the proportion of answers that were correct.  
3) Generate a pandas dataframe with 100 rows per question in GSM8K. The columns should be the GSM8K question id, the question text itself, whether the question is from the train or test split, the hidden state from the model from the first forward pass of the question (before any reasoning/answer tokens are generated), the reasoning trace id, the generated reasoning trace, the list of W/S generated answers if we had stopped early after any point, inserted the answer suffix and generated and extracted a numerical answer, the list of W/S extracted answers, and the corresponding list of W/S proportions across all 100 traces for this query that were correct. Note that within the group of 100 rows corresponding to a single GSM8K question, the question id, question text itself, hidden state, and list of proportions after early stopping after each point will all be shared. Save the computed data to a csv.      
4) Write separate python code that loads the pandas dataframe from csv after it has been computed.  The new python code should construct an MLP that takes in the 1024 dimension hidden states as input, has one hidden layer with dimension 128, and outputs a W/S dimensional vector. It should be trained on a dataset with one entry per question in the GSM8K dataset, where the input is the hidden state and the output is the W/S dimensional list of proportions. The code should report the MSE on both the train and the test set, and it should also output data about, for each W/S possible early stopping position, the pearson correlation between the predicted value and the actual value.  It should report MSE both across the entire predicted and actual output and across each individual entry of the W/S output individually.  
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

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    raise ImportError("Transformers library not found. Please install via pip install transformers")


def load_gsm8k_dataset():
    """
    Load the GSM8K dataset using Hugging Face datasets.
    The dataset is loaded from "openai/gsm8k" with configuration "main".
    For each sample, the ground truth answer is extracted as the text following "#### " in the answer column.
    Returns two lists (for train and test) of dicts with keys: 'id', 'question', 'answer'.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install the datasets library (pip install datasets) to load GSM8K data.")
    
    ds_train = load_dataset("openai/gsm8k", "main", split="train")
    ds_test = load_dataset("openai/gsm8k", "main", split="test")
    
    train_data = []
    test_data = []
    
    for i, sample in enumerate(ds_train):
        question = sample["question"]
        answer_field = sample["answer"]
        if "#### " in answer_field:
            ground_truth = answer_field.split("#### ")[-1].strip()
        else:
            ground_truth = answer_field.strip()
        train_data.append({"id": f"train_{i}", "question": question, "answer": ground_truth})
    
    for i, sample in enumerate(ds_test):
        question = sample["question"]
        answer_field = sample["answer"]
        if "#### " in answer_field:
            ground_truth = answer_field.split("#### ")[-1].strip()
        else:
            ground_truth = answer_field.strip()
        test_data.append({"id": f"test_{i}", "question": question, "answer": ground_truth})
    
    return train_data, test_data


def get_model_and_tokenizer():
    """
    Load the distilled Qwen 1.5B model and its tokenizer.
    Throws an error if the model cannot be loaded.
    """
    model_name = "distilled-qwen-1.5b"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Error loading model {model_name}: {e}")
    return model, tokenizer


def extract_numerical_answer(forced_text):
    """
    Extracts the answer text from the last occurrence of \boxed{<answer>} in forced_text.
    Returns the extracted answer text as a string, or None if no match is found.
    """
    matches = re.findall(r'\\boxed\{([^}]*)\}', forced_text)
    if matches:
        return matches[-1].strip()
    return None


def generate_data(batch_idx, split='train', num_traces=100, W=16, S=4096, output_csv='gsm8k_results.csv', batch_size=100):
    """
    Generate GSM8K reasoning trace data for a specific batch of questions.
    Processes questions from index batch_idx*batch_size to (batch_idx+1)*batch_size
    from the specified split (train or test).
    
    Args:
        batch_idx: Which batch of questions to process
        split: Which split to process ('train' or 'test')
        num_traces: Number of reasoning traces to generate per question
        W: Window size for early stopping positions
        S: Maximum number of new tokens
        output_csv: Path to save/load results
        batch_size: Number of questions per batch
    """
    print(f"Starting data generation for batch {batch_idx} of split {split}...")
    train_data, test_data = load_gsm8k_dataset()
    
    # Select the appropriate split
    questions = train_data if split == 'train' else test_data
    
    # Calculate batch bounds
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, len(questions))
    
    if start_idx >= len(questions):
        raise ValueError(f"Batch index {batch_idx} is too large for split {split} with {len(questions)} questions")
    
    # Get questions for this batch
    batch_questions = questions[start_idx:end_idx]
    print(f"Processing questions {start_idx} to {end_idx-1} from split {split}")
    
    # Check for existing results
    completed_question_ids = set()
    all_data = []
    if os.path.exists(output_csv):
        print(f"Loading existing results from {output_csv}")
        existing_df = pd.read_csv(output_csv)
        # Count number of traces per question
        trace_counts = existing_df.groupby('question_id').size()
        # Get questions with all traces completed
        completed_question_ids = set(trace_counts[trace_counts >= num_traces].index)
        all_data = existing_df.to_dict('records')
        print(f"Found {len(completed_question_ids)} completed questions")

    # Load the Qwen model and tokenizer
    model, tokenizer = get_model_and_tokenizer()

    early_stopping_positions = list(range(W, S + 1, W))  # e.g., 16, 32, ... 4096

    for question in batch_questions:
        qid = question['id']
        
        # Skip if question is already completed
        if qid in completed_question_ids:
            print(f"Skipping completed question {qid}")
            continue
            
        q_text = question['question']
        q_answer = question['answer']
        print(f"Processing question {qid} from {split} split...")
        
        # Obtain hidden state from the model's first forward pass
        inputs = tokenizer(q_text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hidden_state_tensor = outputs.hidden_states[-1][0, -1, :]
        hidden_state = hidden_state_tensor.detach().cpu().numpy().tolist()
        
        # Assert hidden state has 1024 dimensions
        assert len(hidden_state) == 1024, f"Hidden state dimension is {len(hidden_state)}, expected 1024"
        
        trace_results = []
        early_correct_matrix = []  # aggregate correctness flags per early stopping position
        
        for trace_id in range(num_traces):
            # Generate reasoning trace using model generation
            inputs_trace = tokenizer(q_text, return_tensors="pt")
            with torch.no_grad():
                generated_ids = model.generate(inputs_trace['input_ids'], do_sample=True, temperature=0.6, max_new_tokens=S)
            reasoning_trace = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            early_generated_answers = []
            early_extracted_answers = []
            early_correct_flags = []
            
            # For each early stopping position, force answer generation
            for pos in early_stopping_positions:
                # Tokenize the reasoning trace to get token ids
                trace_ids = tokenizer(reasoning_trace, return_tensors="pt").input_ids[0]
                effective_pos = pos if pos < len(trace_ids) else len(trace_ids)
                partial_ids = trace_ids[:effective_pos]
                partial_text = tokenizer.decode(partial_ids, skip_special_tokens=True)
                suffix = "...Oh, I suddenly got the answer to the whole problem, **Final Answer**:\n\n[\\boxed{"
                prompt = partial_text + suffix
                forced_ids = tokenizer(prompt, return_tensors="pt").input_ids
                with torch.no_grad():
                    forced_output = model.generate(forced_ids, max_new_tokens=50)
                forced_text = tokenizer.decode(forced_output[0], skip_special_tokens=True)
                forced_answer = forced_text
                
                early_generated_answers.append(forced_answer)
                extracted = extract_numerical_answer(forced_answer)
                early_extracted_answers.append(extracted)
                try:
                    if float(extracted) == float(q_answer):
                        early_correct_flags.append(1)
                    else:
                        early_correct_flags.append(0)
                except Exception:
                    early_correct_flags.append(0)
            
            early_correct_matrix.append(early_correct_flags)
            trace_results.append({
                "question_id": qid,
                "question_text": q_text,
                "split": split,
                "hidden_state": hidden_state,
                "trace_id": trace_id,
                "reasoning_trace": reasoning_trace,
                "early_generated_answers": early_generated_answers,
                "early_extracted_answers": early_extracted_answers
                # early_stop_correct_proportions will be added later
            })
        
        # Compute shared early stopping correctness proportions for this question
        early_correct_proportions = np.mean(early_correct_matrix, axis=0).tolist()
        for row in trace_results:
            row["early_stop_correct_proportions"] = early_correct_proportions
            all_data.append(row)
        
        # Print detailed information for the first question in the batch
        if question == batch_questions[0]:
            print("\nDetailed information for first question in batch:")
            print(f"Question ID: {qid}")
            print(f"Question text: {q_text}")
            print(f"Ground truth answer: {q_answer}")
            print(f"Hidden state dimension: {len(hidden_state)}")
            print(f"Number of traces generated: {len(trace_results)}")
            print(f"Early stopping positions: {early_stopping_positions}")
            print(f"Average correctness at each position: {early_correct_proportions}")
            print(f"Sample reasoning trace (first trace): {trace_results[0]['reasoning_trace'][:200]}...")
            print(f"Sample early extracted answers (first trace): {trace_results[0]['early_extracted_answers']}\n")
        
        # Save intermediate results after each question
        df = pd.DataFrame(all_data)
        df.to_csv(output_csv, index=False)
        print(f"Saved intermediate results to {output_csv}")
    
    # Final save (though this should be the same as the last intermediate save)
    df = pd.DataFrame(all_data)
    df.to_csv(output_csv, index=False)
    print(f"Data saved to {output_csv}")


def train_mlp(csv_file='gsm8k_results.csv', num_epochs=10, batch_size=4, learning_rate=1e-3):
    """
    Train an MLP to predict the early stopping correctness proportions from the hidden state.
    The input is a 1024-dim vector (hidden state) and the output is a vector of length (S/W).
    Training is performed on one entry per question (grouping the num_traces rows for each question).
    Reports overall MSE and per-position MSE and Pearson correlation on both train and test splits.
    """
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)

    import ast
    def parse_list(x):
        try:
            return ast.literal_eval(x)
        except Exception:
            return x

    df['hidden_state'] = df['hidden_state'].apply(parse_list)
    df['early_stop_correct_proportions'] = df['early_stop_correct_proportions'].apply(parse_list)

    # Group by question_id, split, question_text and take the first occurrence
    grouped = df.groupby(['question_id', 'split', 'question_text']).first().reset_index()

    X = np.vstack(grouped['hidden_state'].values)  # shape (num_questions, 1024)
    Y = np.vstack(grouped['early_stop_correct_proportions'].values)  # shape (num_questions, output_dim)

    train_mask = grouped['split'] == 'train'
    X_train, Y_train = X[train_mask], Y[train_mask]
    X_test, Y_test = X[~train_mask], Y[~train_mask]

    print(f"Training questions: {X_train.shape[0]}, Testing questions: {X_test.shape[0]}")

    class MLP(nn.Module):
        def __init__(self, input_dim=1024, hidden_dim=128, output_dim=256):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    model_mlp = MLP(input_dim=1024, hidden_dim=128, output_dim=Y.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_mlp.parameters(), lr=learning_rate)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, Y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print("Starting MLP training...")
    for epoch in range(num_epochs):
        model_mlp.train()
        running_loss = 0.0
        for batch_X, batch_Y in train_loader:
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
        train_pred = model_mlp(X_train_tensor).cpu().numpy()
        test_pred = model_mlp(X_test_tensor).cpu().numpy()

    mse_train_overall = np.mean((train_pred - Y_train)**2)
    mse_test_overall = np.mean((test_pred - Y_test)**2)
    print(f"Overall MSE on train: {mse_train_overall:.4f}")
    print(f"Overall MSE on test: {mse_test_overall:.4f}")

    mse_train_individual = np.mean((train_pred - Y_train)**2, axis=0)
    mse_test_individual = np.mean((test_pred - Y_test)**2, axis=0)

    pearson_train = [pearsonr(train_pred[:, i], Y_train[:, i])[0] for i in range(Y_train.shape[1])]
    pearson_test = [pearsonr(test_pred[:, i], Y_test[:, i])[0] for i in range(Y_test.shape[1])]

    print("\nEarly stopping position-wise MSE and Pearson correlation (Train):")
    for i, (mse_val, p_val) in enumerate(zip(mse_train_individual, pearson_train)):
        print(f"Position {i+1}: MSE: {mse_val:.4f}, Pearson: {p_val:.4f}")

    print("\nEarly stopping position-wise MSE and Pearson correlation (Test):")
    for i, (mse_val, p_val) in enumerate(zip(mse_test_individual, pearson_test)):
        print(f"Position {i+1}: MSE: {mse_val:.4f}, Pearson: {p_val:.4f}")


def main():
    parser = argparse.ArgumentParser(description="MLP test experiment for GSM8K reasoning traces")
    parser.add_argument("--generate", action="store_true", help="Run data generation experiment")
    parser.add_argument("--train", action="store_true", help="Train the MLP on generated data")
    parser.add_argument("--batch_idx", type=int, help="Batch index for data generation (required with --generate)")
    parser.add_argument("--split", type=str, choices=['train', 'test'], help="Which split to process (required with --generate)")
    parser.add_argument("--csv_file", type=str, default="gsm8k_results.csv", help="CSV file to load/save generated data")
    args = parser.parse_args()

    if args.generate:
        if args.batch_idx is None:
            parser.error("--batch_idx is required when using --generate")
        if args.split is None:
            parser.error("--split is required when using --generate")
        generate_data(batch_idx=args.batch_idx, split=args.split, output_csv=args.csv_file)
    elif args.train:
        train_mlp(csv_file=args.csv_file)
    else:
        print("Please specify --generate or --train.")


if __name__ == "__main__":
    main()