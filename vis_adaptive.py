import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import argparse
from datetime import datetime
from collections import defaultdict

# TODO: change accuracy to reflect accuracy across questions, not across trials, e.g. question is 
# correct if the answer is the mode 

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
    
    budget_dirs = sorted(glob.glob(os.path.join(adaptive_dir, "budget_*")))
    
    for budget_dir in budget_dirs:
        token_budget = int(budget_dir.split("_")[-1])
        token_budgets.append(token_budget)
        predictions_vs_actuals[token_budget] = {}  # Dict for each budget level
        
        question_files = glob.glob(os.path.join(budget_dir, "question_*_tokens_*.json"))
        correct_count = 0
        total_count = 0
        num_questions = 0
        
        for qfile in question_files:
            with open(qfile, 'r') as f:
                data = json.load(f)
                problem_id = data.get('problem_id')
                is_corrects = data.get('is_corrects', [])
                predicted_score = data.get('predicted_score', None)
                
                if is_corrects:
                    actual_proportion = sum(is_corrects) / len(is_corrects)
                    predictions_vs_actuals[token_budget][problem_id] = {
                        'predicted': predicted_score,
                        'actual': actual_proportion,
                        'num_trials': len(is_corrects)
                    }
                    correct_count += sum(is_corrects)
                    total_count += len(is_corrects)
                    num_questions += 1
        
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
        print(f"\nBudget: {budget} tokens")
        total_abs_diff = 0
        num_questions = len(predictions_vs_actuals[budget])
        
        for qid in sorted(predictions_vs_actuals[budget].keys()):
            data = predictions_vs_actuals[budget][qid]
            pred = data['predicted']
            act = data['actual']
            diff = act - pred if pred is not None else float('nan')
            total_abs_diff += abs(diff) if pred is not None else 0
            
            print(f"{budget:11d} | {qid:10d} | {pred:8.3f} | {act:6.3f} | {data['num_trials']:6d} | {diff:+9.3f}")
        
        avg_abs_diff = total_abs_diff / num_questions if num_questions > 0 else 0
        print(f"Average absolute difference for budget {budget}: {avg_abs_diff:.3f}")

def plot_results(adaptive_dir, nonadaptive_dir, oracle_dir=None):
    # Load adaptive results
    adaptive_tokens, adaptive_accuracies, adaptive_counts, adaptive_predictions = load_adaptive_results(adaptive_dir)
    print_prediction_analysis(adaptive_predictions, "Adaptive (Non-Oracle)")
    
    # Load oracle results if provided
    if oracle_dir:
        oracle_tokens, oracle_accuracies, oracle_counts, oracle_predictions = load_adaptive_results(oracle_dir)
        print_prediction_analysis(oracle_predictions, "Oracle")
    
    # Load results with question counts
    adaptive_tokens, adaptive_accuracies, adaptive_counts = load_adaptive_results(adaptive_dir)
    print("\nAdaptive Results:")
    print("Token Budget | Accuracy | Questions")
    print("-" * 40)
    for t, a, c in zip(adaptive_tokens, adaptive_accuracies, adaptive_counts):
        print(f"{t:11d} | {a:7.2f}% | {c:9d}")
    
    nonadaptive_tokens, nonadaptive_accuracies, nonadaptive_counts = load_results(nonadaptive_dir)
    print("\nNon-adaptive Results:")
    print("Token Budget | Accuracy | Questions")
    print("-" * 40)
    for t, a, c in zip(nonadaptive_tokens, nonadaptive_accuracies, nonadaptive_counts):
        print(f"{t:11d} | {a:7.2f}% | {c:9d}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(adaptive_tokens, adaptive_accuracies, 'b-', marker='o', label='Adaptive')
    plt.plot(nonadaptive_tokens, nonadaptive_accuracies, 'r-', marker='s', label='Non-adaptive')
    
    if oracle_dir:
        plt.plot(oracle_tokens, oracle_accuracies, 'g-', marker='^', label='Oracle')
    
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--adaptive-dir', type=str, required=True)
    parser.add_argument('--nonadaptive-dir', type=str, required=True)
    parser.add_argument('--oracle-dir', type=str)
    args = parser.parse_args()
    
    plot_results(args.adaptive_dir, args.nonadaptive_dir, args.oracle_dir)

if __name__ == '__main__':
    main() 