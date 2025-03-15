import os
import json
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

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

def create_comparison_plot(adaptive_dir, nonadaptive_dir, model_name, dataset_name):
    """Create and save comparison plot"""
    # Load results
    adaptive_tokens, adaptive_accuracies = load_results(adaptive_dir)
    nonadaptive_tokens, nonadaptive_accuracies = load_results(nonadaptive_dir)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot both lines
    plt.plot(adaptive_tokens, adaptive_accuracies, 'b-', label='Adaptive', marker='o')
    plt.plot(nonadaptive_tokens, nonadaptive_accuracies, 'r-', label='Non-adaptive', marker='o')
    
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
    
    # Create difference plot
    plt.figure(figsize=(10, 6))
    
    # Interpolate if necessary to align token budgets
    common_tokens = sorted(set(adaptive_tokens) & set(nonadaptive_tokens))
    adaptive_interp = np.interp(common_tokens, adaptive_tokens, adaptive_accuracies)
    nonadaptive_interp = np.interp(common_tokens, nonadaptive_tokens, nonadaptive_accuracies)
    
    # Plot accuracy difference
    diff = adaptive_interp - nonadaptive_interp
    plt.plot(common_tokens, diff, 'g-', marker='o')
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Customize plot
    plt.xlabel('Token Budget')
    plt.ylabel('Accuracy Improvement (Adaptive - Non-adaptive)')
    plt.title(f'Accuracy Improvement from Adaptive Allocation\n{model_name} on {dataset_name}')
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.savefig(f'figures/token_accuracy_improvement_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Visualize adaptive vs non-adaptive results')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--adaptive-dir', type=str, required=True, help='Directory with adaptive results')
    parser.add_argument('--nonadaptive-dir', type=str, required=True, help='Directory with non-adaptive results')
    
    args = parser.parse_args()
    
    create_comparison_plot(
        args.adaptive_dir,
        args.nonadaptive_dir,
        args.model,
        args.dataset
    )
    
    print(f"Generated visualizations in figures/")

if __name__ == '__main__':
    main() 