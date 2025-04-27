import os
import pandas as pd
import numpy as np
from constants import Y_STEM
import ast
import seaborn as sns
import matplotlib.pyplot as plt

def load_gsm8k_train_y_data():
    # Construct the directory path for GSM8K Y data
    data_dir = os.path.join(Y_STEM, "gsm8k_results")
    
    # Initialize empty list to store all dataframes
    all_data = []
    batch_idx = 0
    
    while True:
        # Construct file path for each batch
        y_file = os.path.join(data_dir, f"gsm8k_Y_train_{batch_idx}.csv")
        
        # Break if file doesn't exist (we've processed all batches)
        if not os.path.exists(y_file):
            if batch_idx == 0:
                raise ValueError(f"No data files found at {y_file}")
            break
            
        print(f"Loading Y data from {y_file}")
        y_data = pd.read_csv(y_file)
        
        # Parse the early stopping proportions from string to list
        y_data['early_stop_correct_proportions'] = y_data['early_stop_correct_proportions'].apply(ast.literal_eval)
        
        all_data.append(y_data)
        batch_idx += 1
    
    # Concatenate all batches
    return pd.concat(all_data, ignore_index=True)

def expand_early_stop_data(df):
    """
    Expand the DataFrame to create a row for each position's probability.
    Also converts position to token budget.
    """
    expanded_rows = []
    
    for _, row in df.iterrows():
        probabilities = ast.literal_eval(row['early_stop_correct_proportions'])
        for pos, prob in enumerate(probabilities):
            new_row = row.copy()
            # Convert position to token budget: (position + 1) * 16
            token_budget = (pos + 1) * 16
            new_row['token_budget'] = token_budget
            new_row['probability'] = prob
            expanded_rows.append(new_row)
    
    expanded_df = pd.DataFrame(expanded_rows)
    expanded_df = expanded_df.drop('early_stop_correct_proportions', axis=1)
    
    print(f"Original shape: {df.shape}")
    print(f"Expanded shape: {expanded_df.shape}")
    print(f"Number of token budgets per question: {len(probabilities)}")
    
    return expanded_df

def create_kde_plots(expanded_df, output_dir='figures'):
    """
    Create and save two KDE plots:
    1. Distribution by token budget
    2. Aggregate distribution across all token budgets
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('default')
    
    # 1. Token budget-wise KDE plot
    plt.figure(figsize=(15, 10))
    
    # Create a color map for token budgets
    unique_budgets = sorted(expanded_df['token_budget'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_budgets)))
    
    # Plot each token budget's distribution
    for budget, color in zip(unique_budgets, colors):
        budget_data = expanded_df[expanded_df['token_budget'] == budget]['probability']
        sns.kdeplot(
            data=budget_data,
            label=f'Token Budget {budget}',
            color=color,
            alpha=0.7,
            linewidth=2
        )
    
    plt.title('Distribution of Early Stopping Probabilities by Token Budget', fontsize=16, pad=20)
    plt.xlabel('Probability of Correct Answer', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    
    # Place legend inside the plot in the upper right
    plt.legend(title='Token Budget',
              title_fontsize=12,
              fontsize=10,
              loc='upper right',
              ncol=2,
              framealpha=0.9)
    
    plt.grid(True, alpha=0.3)
    
    # Save token budget-wise plot
    budget_wise_path = os.path.join(output_dir, 'early_stopping_kde_by_budget.png')
    plt.savefig(budget_wise_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Token budget-wise KDE plot saved to {budget_wise_path}")
    
    # 2. Aggregate KDE plot
    plt.figure(figsize=(12, 8))
    
    # Create aggregate KDE plot
    sns.kdeplot(
        data=expanded_df['probability'],
        color='blue',
        fill=True,
        alpha=0.3,
        linewidth=2,
        label='Density'
    )
    
    # Add mean line
    mean_prob = expanded_df['probability'].mean()
    plt.axvline(x=mean_prob, color='red', linestyle='--', 
                label=f'Mean = {mean_prob:.3f}')
    
    # Add median line
    median_prob = expanded_df['probability'].median()
    plt.axvline(x=median_prob, color='green', linestyle='--', 
                label=f'Median = {median_prob:.3f}')
    
    plt.title('Aggregate Distribution of Early Stopping Probabilities\nAcross All Token Budgets', 
              fontsize=16, pad=20)
    plt.xlabel('Probability of Correct Answer', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    
    # Place legend inside the plot in the upper right
    plt.legend(fontsize=12, loc='upper right', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    
    # Add summary statistics as text
    stats_text = (
        f'Summary Statistics:\n'
        f'Mean: {mean_prob:.3f}\n'
        f'Median: {median_prob:.3f}\n'
        f'Std: {expanded_df["probability"].std():.3f}\n'
        f'Min: {expanded_df["probability"].min():.3f}\n'
        f'Max: {expanded_df["probability"].max():.3f}'
    )
    plt.text(0.02, 0.98, stats_text, 
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save aggregate plot
    aggregate_path = os.path.join(output_dir, 'early_stopping_kde_aggregate.png')
    plt.savefig(aggregate_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Aggregate KDE plot saved to {aggregate_path}")

def main():
    # Load the original data
    print("Loading GSM8K train Y data...")
    original_df = load_gsm8k_train_y_data()
    
    # Expand the data
    print("\nExpanding data to create position-wise rows...")
    expanded_df = expand_early_stop_data(original_df)
    
    # Save the expanded data
    output_file = "gsm8k_train_y_expanded.csv"
    print(f"\nSaving expanded data to {output_file}")
    expanded_df.to_csv(output_file, index=False)
    
    # Create and save both KDE plots
    print("\nCreating KDE plots...")
    create_kde_plots(expanded_df)
    
    # Print some statistics
    print("\nData Statistics:")
    print(f"Total number of questions: {len(original_df)}")
    print(f"Total number of token budgets: {len(expanded_df)}")
    print("\nProbability distribution by token budget:")
    budget_stats = expanded_df.groupby('token_budget')['probability'].agg(['mean', 'std', 'min', 'max'])
    print(budget_stats)

if __name__ == "__main__":
    main()
