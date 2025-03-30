import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import argparse
from datetime import datetime
from collections import defaultdict

'''
/n/netscratch/dwork_lab/Lab/katrina/envs/reasoning/bin/python -u vis_adaptive.py     
    --dataset gsm8k     
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B     
    --adaptive-dir Dynasor/benchmark/TokenDeprivation/results/adaptive_deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B_gsm8k_mlpgsm8k_train_20250327183843     
    --nonadaptive-dir Dynasor/benchmark/TokenDeprivation/results/deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B_gsm8k_step32_max256_trials10_20250325152252     
    --oracle-dir Dynasor/benchmark/TokenDeprivation/results/oracle_deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B_gsm8k_mlpgsm8k_train_20250327174756
'''

# TODO: change accuracy to reflect accuracy across questions, not across trials, e.g. question is 
# correct if the answer is the mode 

# TODO: why is oracle accuracy LESS than non-adaptive accuracy?
# Check predicted vs actual accuracies
# Regenerate ground truth data for gsm8k first train batch (i think the prompting changed at some point)

def load_results(results_dir):
    """Load results from json files in the given directory"""
    # Dictionary to store results for each token budget
    token_results = {}
    
    # Find all json files
    json_files = glob(os.path.join(results_dir, "question_*.json"))
    
    for file_path in json_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            max_tokens = data['max_tokens']
            
            if max_tokens not in token_results:
                token_results[max_tokens] = {
                    'correct': [],
                    'total': 0
                }
            
            # Add results for this question
            token_results[max_tokens]['correct'].extend(data['is_corrects'])
            token_results[max_tokens]['total'] += len(data['is_corrects'])
    
    # Calculate accuracies
    x_tokens = sorted(token_results.keys())
    y_accuracies = [
        np.mean(token_results[tokens]['correct']) * 100
        for tokens in x_tokens
    ]
    
    return x_tokens, y_accuracies

def create_comparison_plot(adaptive_dir, nonadaptive_dir, oracle_dir, model_name, dataset_name):
    """Create and save comparison plot with optional oracle data"""
    # Load results
    adaptive_tokens, adaptive_accuracies = load_results(adaptive_dir)
    nonadaptive_tokens, nonadaptive_accuracies = load_results(nonadaptive_dir)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot both lines
    plt.plot(adaptive_tokens, adaptive_accuracies, 'b-', label='Adaptive (MLP)', marker='o')
    plt.plot(nonadaptive_tokens, nonadaptive_accuracies, 'r-', label='Non-adaptive', marker='o')
    
    # Add oracle data if available
    if oracle_dir and os.path.exists(oracle_dir):
        oracle_tokens, oracle_accuracies = load_results(oracle_dir)
        plt.plot(oracle_tokens, oracle_accuracies, 'g-', label='Adaptive (Oracle)', marker='o')
    
    # Customize plot
    plt.xlabel('Token Budget')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs Token Budget\n{model_name} on {dataset_name}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save figure
    os.makedirs('figures', exist_ok=True)
    plt.savefig(f'figures/token_accuracy_comparison_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create improvement plot
    plt.figure(figsize=(10, 6))
    
    # Interpolate if necessary to align token budgets
    common_tokens = sorted(set(adaptive_tokens) & set(nonadaptive_tokens))
    adaptive_interp = np.interp(common_tokens, adaptive_tokens, adaptive_accuracies)
    nonadaptive_interp = np.interp(common_tokens, nonadaptive_tokens, nonadaptive_accuracies)
    
    # Plot MLP improvement
    diff_mlp = adaptive_interp - nonadaptive_interp
    plt.plot(common_tokens, diff_mlp, 'b-', marker='o', label='MLP vs. Non-adaptive')
    
    # Add oracle improvement if available
    if oracle_dir and os.path.exists(oracle_dir):
        oracle_interp = np.interp(common_tokens, oracle_tokens, oracle_accuracies)
        diff_oracle = oracle_interp - nonadaptive_interp
        plt.plot(common_tokens, diff_oracle, 'g-', marker='o', label='Oracle vs. Non-adaptive')
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Customize plot
    plt.xlabel('Token Budget')
    plt.ylabel('Accuracy Improvement')
    plt.title(f'Accuracy Improvement over Non-adaptive\n{model_name} on {dataset_name}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save figure
    plt.savefig(f'figures/token_accuracy_improvement_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def load_adaptive_results(adaptive_dir):
    """Load results from adaptive runs with budget subdirectories."""
    token_budgets = []
    accuracies = []
    question_counts = []
    predictions_vs_actuals = {}  # New dict to store comparisons
    
    budget_dirs = sorted(glob(os.path.join(adaptive_dir, "budget_*")))
    
    for budget_dir in budget_dirs:
        token_budget = int(budget_dir.split("_")[-1])
        token_budgets.append(token_budget)
        predictions_vs_actuals[token_budget] = {}  # Dict for each budget level
        
        question_files = glob(os.path.join(budget_dir, "question_*_tokens_*.json"))
        correct_count = 0
        total_count = 0
        num_questions = 0
        
        for qfile in question_files:
            with open(qfile, 'r') as f:
                data = json.load(f)
                problem_id = data.get('problem_id')
                is_corrects = data.get('is_corrects', [])
                predicted_score = data.get('predicted_score', None)
                tokens_used_per_problem = int(qfile.split("_")[-1].split(".")[0])
                
                if is_corrects:
                    actual_proportion = sum(is_corrects) / len(is_corrects)
                    predictions_vs_actuals[token_budget][problem_id] = {
                        'predicted': predicted_score,
                        'actual': actual_proportion,
                        'num_trials': len(is_corrects),
                        'tokens_used': tokens_used_per_problem,
                    }
                    correct_count += sum(is_corrects)
                    total_count += len(is_corrects)
                    num_questions += 1
            tokens_used = [predictions_vs_actuals[token_budget][qid]['tokens_used'] for qid in predictions_vs_actuals[token_budget]]
            predictions = [predictions_vs_actuals[token_budget][qid]['predicted'] for qid in predictions_vs_actuals[token_budget]]
            actuals = [predictions_vs_actuals[token_budget][qid]['actual'] for qid in predictions_vs_actuals[token_budget]]
            #print(f"Token budget: {token_budget}, num questions: {num_questions}")
            #print(f"Tokens used: {tokens_used}")
            #print(f"Predictions: {predictions}")
            #print(f"Actuals: {actuals}")
            #print("\n")
        
        question_counts.append(num_questions)
        if total_count > 0:
            accuracies.append(correct_count / total_count * 100)
        else:
            accuracies.append(0)
    return token_budgets, accuracies, question_counts, predictions_vs_actuals

def load_results(results_dir):
    """Load results from non-adaptive runs."""
    token_budgets = []
    accuracies = []
    question_counts = []  # New list to track question counts
    
    # Group files by token budget
    budget_files = defaultdict(list)
    for qfile in glob(os.path.join(results_dir, "question_*_tokens_*.json")):
        with open(qfile, 'r') as f:
            data = json.load(f)
            budget = data.get('max_tokens')
            budget_files[budget].append(data)
    
    # Process each budget level
    for budget in sorted(budget_files.keys()):
        token_budgets.append(budget)
        correct_count = 0
        total_count = 0
        num_questions = len(budget_files[budget])  # Count questions for this budget
        
        for data in budget_files[budget]:
            is_corrects = data.get('is_corrects', [])
            if is_corrects:
                correct_count += sum(is_corrects)
                total_count += len(is_corrects)
        
        question_counts.append(num_questions)  # Store count for this budget
        if total_count > 0:
            accuracies.append(correct_count / total_count * 100)
        else:
            accuracies.append(0)
    
    return token_budgets, accuracies, question_counts

def print_prediction_analysis(predictions_vs_actuals, run_type="Adaptive"):
    print(f"\n{run_type} Run Analysis - Predicted vs Actual Proportions:")
    print("Token Budget | Question ID | Predicted | Actual | Trials | Difference")
    print("-" * 75)
    
    for budget in sorted(predictions_vs_actuals.keys()):
        # print(f"\nBudget: {budget} tokens")
        total_abs_diff = 0
        num_questions = len(predictions_vs_actuals[budget])
        
        for qid in sorted(predictions_vs_actuals[budget].keys()):
            data = predictions_vs_actuals[budget][qid]
            pred = data['predicted']
            act = data['actual']
            
            # Handle None predictions
            if pred is None:
                # print(f"{budget:11d} | {qid:10d} | {'N/A':>8s} | {act:6.3f} | {data['num_trials']:6d} | {'N/A':>9s}")
                continue
                
            diff = act - pred
            total_abs_diff += abs(diff)
            
            # print(f"{budget:11d} | {qid:10d} | {pred:8.3f} | {act:6.3f} | {data['num_trials']:6d} | {diff:+9.3f}")
        
        # Only calculate average for questions with predictions
        valid_predictions = sum(1 for qid in predictions_vs_actuals[budget] 
                              if predictions_vs_actuals[budget][qid]['predicted'] is not None)
        if valid_predictions > 0:
            avg_abs_diff = total_abs_diff / valid_predictions
            print(f"Average absolute difference for budget {budget}: {avg_abs_diff:.3f}")
        else:
            print(f"No valid predictions for budget {budget}")

def create_prediction_scatter(adaptive_predictions, oracle_predictions=None):
    """Create scatter plot of predicted vs actual proportions."""
    plt.figure(figsize=(10, 6))
    
    # Collect all predictions and actuals for adaptive
    adaptive_pred = []
    adaptive_act = []
    for budget in adaptive_predictions:
        for qid in adaptive_predictions[budget]:
            data = adaptive_predictions[budget][qid]
            if data['predicted'] is not None:
                adaptive_pred.append(data['predicted'])
                adaptive_act.append(data['actual'])
    
    # Plot adaptive data
    plt.scatter(adaptive_pred, adaptive_act, alpha=0.5, label='Adaptive')
    adaptive_corr = np.corrcoef(adaptive_pred, adaptive_act)[0,1]
    print(f"\nAdaptive Correlation: {adaptive_corr:.3f}")
    
    # Plot oracle data if available
    if oracle_predictions:
        oracle_pred = []
        oracle_act = []
        for budget in oracle_predictions:
            for qid in oracle_predictions[budget]:
                data = oracle_predictions[budget][qid]
                if data['predicted'] is not None:
                    oracle_pred.append(data['predicted'])
                    oracle_act.append(data['actual'])
        
        plt.scatter(oracle_pred, oracle_act, alpha=0.5, label='Oracle')
        oracle_corr = np.corrcoef(oracle_pred, oracle_act)[0,1]
        print(f"Oracle Correlation: {oracle_corr:.3f}")
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)  # Add y=x line
    plt.xlabel('Predicted Proportion Correct')
    plt.ylabel('Actual Proportion Correct')
    plt.title('Predicted vs Actual Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'figures/prediction_scatter.png', dpi=300, bbox_inches='tight')

def get_directory_paths(dataset, model, split, start, end):
    """Construct directory paths based on input parameters"""
    base_dir = f"data/{dataset}_results"
    
    # Construct file names based on the pattern
    adaptive_file = f"adaptive_{model}_{dataset}_mlp{dataset}_{split}_{start}_{end}"
    non_adaptive_file = f"{model}_{dataset}_step32_max256_trials10_{start}_{end}"
    oracle_file = f"oracle_{model}_{dataset}_mlp{dataset}_{split}_{start}_{end}"
    
    # Construct full paths
    adaptive_dir = os.path.join(base_dir, "adaptive", adaptive_file)
    non_adaptive_dir = os.path.join(base_dir, "baseline", non_adaptive_file)
    oracle_dir = os.path.join(base_dir, "adaptive_oracle", oracle_file)
    
    return adaptive_dir, non_adaptive_dir, oracle_dir

def plot_results(adaptive_dir, non_adaptive_dir, oracle_dir, dataset, model, split, start, end):
    """Original plotting function with any existing plotting logic"""
    # Load adaptive results
    adaptive_tokens, adaptive_accuracies, adaptive_counts, adaptive_predictions = load_adaptive_results(adaptive_dir)
    print_prediction_analysis(adaptive_predictions, "Adaptive (Non-Oracle)")
    
    # Create prediction scatter plot
    if oracle_dir:
        oracle_tokens, oracle_accuracies, oracle_counts, oracle_predictions = load_adaptive_results(oracle_dir)
        create_prediction_scatter(adaptive_predictions, oracle_predictions)
    else:
        create_prediction_scatter(adaptive_predictions)
    
    # Load oracle results if provided
    if oracle_dir:
        print_prediction_analysis(oracle_predictions, "Oracle")
        
        print("\nOracle Results:")
        print("Token Budget | Accuracy | Questions")
        print("-" * 40)
        for t, a, c in zip(oracle_tokens, oracle_accuracies, oracle_counts):
            print(f"{t:11d} | {a:7.2f}% | {c:9d}")
    
    print("\nAdaptive Results:")
    print("Token Budget | Accuracy | Questions")
    print("-" * 40)
    for t, a, c in zip(adaptive_tokens, adaptive_accuracies, adaptive_counts):
        print(f"{t:11d} | {a:7.2f}% | {c:9d}")
    
    nonadaptive_tokens, nonadaptive_accuracies, nonadaptive_counts = load_results(non_adaptive_dir)
    print("\nNon-adaptive Results:")
    print("Token Budget | Accuracy | Questions")
    print("-" * 40)
    for t, a, c in zip(nonadaptive_tokens, nonadaptive_accuracies, nonadaptive_counts):
        print(f"{t:11d} | {a:7.2f}% | {c:9d}")
    
    plt.figure(figsize=(10, 6))
    # sort adaptive_tokens and adaptive_accuracies in ascending order of adaptive_tokens
    adaptive_tokens, adaptive_accuracies = zip(*sorted(zip(adaptive_tokens, adaptive_accuracies)))
    nonadaptive_tokens, nonadaptive_accuracies = zip(*sorted(zip(nonadaptive_tokens, nonadaptive_accuracies)))
    
    predicted_accuracies_adaptive = []
    for token_budget in adaptive_tokens:
        predicted_accuracies_adaptive.append(100.0*np.mean([float(adaptive_predictions[token_budget][qid]['predicted']) for qid in adaptive_predictions[token_budget]]))
    
    plt.plot(adaptive_tokens, adaptive_accuracies, 'b-', marker='o', label='Adaptive')
    # plt.plot(adaptive_tokens, predicted_accuracies_adaptive, 'p-', marker='o', label='Adaptive Predicted')
    plt.plot(nonadaptive_tokens, nonadaptive_accuracies, 'r-', marker='s', label='Non-adaptive')
    
    if oracle_dir:
        oracle_tokens, oracle_accuracies = zip(*sorted(zip(oracle_tokens, oracle_accuracies)))
        # plt.plot(oracle_tokens, oracle_accuracies, 'g-', marker='^', label='Oracle')
        # NOTE: for "Oracle" line, plot ground truth recorded prediction proportions rather than 10 new sampled reasoning traces proportion
        # sidesteps any data issues
        predicted_accuracies = []
        for token_budget in oracle_tokens:
            predicted_accuracies.append(100.0*np.mean([float(oracle_predictions[token_budget][qid]['predicted']) for qid in oracle_predictions[token_budget]]))
        print("Oracle predicted accuracies: ", predicted_accuracies)
        # plt.plot(oracle_tokens, predicted_accuracies, 'g-', marker='^', label='Oracle')
        # plots actual accuracies from 10 resampled reasoning traces
        plt.plot(oracle_tokens, oracle_accuracies, 'p-', label='Oracle', marker='o')
        
    plt.xlabel('Token Budget')
    plt.ylabel('Average Accuracy (%)')
    plt.title('Accuracy vs Token Budget')
    plt.legend()
    plt.grid(True)
    
    # Create figures directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)
    
    # Save plot with timestamp
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f'figures/accuracy_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("Adaptive:")
    print(f"  Max accuracy: {max(adaptive_accuracies):.2f}%")
    print(f"  Min accuracy: {min(adaptive_accuracies):.2f}%")
    print(f"  Mean accuracy: {sum(adaptive_accuracies)/len(adaptive_accuracies):.2f}%")
    
    print("\nNon-adaptive:")
    print(f"  Max accuracy: {max(nonadaptive_accuracies):.2f}%")
    print(f"  Min accuracy: {min(nonadaptive_accuracies):.2f}%")
    print(f"  Mean accuracy: {sum(nonadaptive_accuracies)/len(nonadaptive_accuracies):.2f}%")
    
    if oracle_dir:
        print("\nOracle:")
        print(f"  Max accuracy: {max(oracle_accuracies):.2f}%")
        print(f"  Min accuracy: {min(oracle_accuracies):.2f}%")
        print(f"  Mean accuracy: {sum(oracle_accuracies)/len(oracle_accuracies):.2f}%")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize adaptive and non-adaptive results")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., gsm8k)")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B)")
    parser.add_argument("--split", type=str, required=True, help="Data split (e.g., train, test)")
    parser.add_argument("--start", type=int, required=True, help="Start index")
    parser.add_argument("--end", type=int, required=True, help="End index")
    
    args = parser.parse_args()
    
    # Get directory paths based on arguments
    adaptive_dir, non_adaptive_dir, oracle_dir = get_directory_paths(
        args.dataset,
        args.model,
        args.split,
        args.start,
        args.end
    )
    
    # Call plotting function with all arguments
    plot_results(
        adaptive_dir,
        non_adaptive_dir,
        oracle_dir,
        args.dataset,
        args.model,
        args.split,
        args.start,
        args.end
    )

if __name__ == "__main__":
    main() 