import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import argparse

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
        np.mean(token_results[tokens]['correct'])
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
    total_questions = 0
    total_files = 0
    
    print(f"\nLoading adaptive results from root directory: {adaptive_dir}")
    
    # Find all budget subdirectories
    budget_dirs = sorted(glob(os.path.join(adaptive_dir, "budget_*")))
    print(f"Found {len(budget_dirs)} budget subdirectories")
    
    for budget_dir in budget_dirs:
        # Extract token budget from directory name
        token_budget = int(budget_dir.split("_")[-1])
        token_budgets.append(token_budget)
        
        # Load all question results from this budget directory
        question_files = glob(os.path.join(budget_dir, "question_*_tokens_*.json"))
        total_files += len(question_files)
        correct_count = 0
        total_count = 0
        
        print(f"\nProcessing budget directory: {budget_dir}")
        print(f"Found {len(question_files)} question files")
        
        for qfile in question_files:
            with open(qfile, 'r') as f:
                data = json.load(f)
                # Average across trials for this question
                is_corrects = data.get('is_corrects', [])
                if is_corrects:
                    correct_count += sum(is_corrects)
                    total_count += len(is_corrects)
                    total_questions += 1
        
        if total_count > 0:
            accuracies.append(correct_count / total_count * 100)
        else:
            accuracies.append(0)
    
    print(f"\nSummary:")
    print(f"Total question records processed: {total_questions}")
    print(f"Total files loaded: {total_files}")
    print(f"Average questions per budget: {total_questions/len(budget_dirs) if budget_dirs else 0:.2f}")
    
    return token_budgets, accuracies

def plot_results(adaptive_dir, nonadaptive_dir, oracle_dir=None):
    # Load adaptive results with new directory structure
    adaptive_tokens, adaptive_accuracies = load_adaptive_results(adaptive_dir)
    
    # Load non-adaptive results (unchanged)
    nonadaptive_tokens, nonadaptive_accuracies = load_results(nonadaptive_dir)
    
    plt.figure(figsize=(10, 6))
    plt.plot(adaptive_tokens, adaptive_accuracies, 'b-', label='Adaptive')
    plt.plot(nonadaptive_tokens, nonadaptive_accuracies, 'r-', label='Non-adaptive')
    
    if oracle_dir:
        oracle_tokens, oracle_accuracies = load_adaptive_results(oracle_dir)
        plt.plot(oracle_tokens, oracle_accuracies, 'g-', label='Oracle')
    
    plt.xlabel('Token Budget')
    plt.ylabel('Average Accuracy (%)')
    plt.title('Accuracy vs Token Budget')
    plt.legend()
    plt.grid(True)
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