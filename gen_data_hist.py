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
    Convert the dataframe with early_stop_correct_proportions lists into a long format
    where each position gets its own row.
    
    Args:
        df: DataFrame with columns including 'early_stop_correct_proportions'
        
    Returns:
        DataFrame with columns:
        - All original columns except 'early_stop_correct_proportions'
        - position: The index in the early_stop_correct_proportions list (0-15)
        - probability: The actual probability value
    """
    # Create list to store expanded rows
    expanded_rows = []
    
    # Iterate through original dataframe
    for _, row in df.iterrows():
        # Get the probabilities list
        probs = row['early_stop_correct_proportions']
        
        # Create a row for each position
        for pos, prob in enumerate(probs):
            # Create new row with all original data except early_stop_correct_proportions
            new_row = {
                'dataset': row['dataset'],
                'question_id': row['question_id'],
                'question_text': row['question_text'],
                'split': row['split'],
                'position': pos,
                'probability': prob
            }
            expanded_rows.append(new_row)
    
    # Convert to DataFrame
    expanded_df = pd.DataFrame(expanded_rows)
    
    # Reset index
    expanded_df = expanded_df.reset_index(drop=True)
    
    print(f"Original shape: {df.shape}")
    print(f"Expanded shape: {expanded_df.shape}")
    print(f"Number of positions per question: {len(df['early_stop_correct_proportions'].iloc[0])}")
    
    return expanded_df

def create_kde_plots(expanded_df, output_dir='figures'):
    """
    Create and save two KDE plots:
    1. Distribution by position
    2. Aggregate distribution across all positions
    
    Args:
        expanded_df: DataFrame with 'position' and 'probability' columns
        output_dir: Directory to save the plots
    """
    # Create figures directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the style
    #plt.style.use('seaborn')
    
    # 1. Position-wise KDE plot
    plt.figure(figsize=(15, 10))
    
    # Create a color map for positions
    unique_positions = sorted(expanded_df['position'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_positions)))
    
    # Plot each position's distribution
    for pos, color in zip(unique_positions, colors):
        pos_data = expanded_df[expanded_df['position'] == pos]['probability']
        sns.kdeplot(
            data=pos_data,
            label=f'Position {pos}',
            color=color,
            alpha=0.7,
            linewidth=2
        )
    
    plt.title('Distribution of Early Stopping Probabilities by Position', fontsize=16, pad=20)
    plt.xlabel('Probability of Correct Answer', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    
    # Place legend inside the plot in the upper right
    plt.legend(title='Position',
              title_fontsize=12,
              fontsize=10,
              loc='upper right',
              ncol=2,  # Use 2 columns to make legend more compact
              framealpha=0.9)  # Make legend background slightly transparent
    
    plt.grid(True, alpha=0.3)
    
    # Save position-wise plot
    position_wise_path = os.path.join(output_dir, 'early_stopping_kde_by_position.png')
    plt.savefig(position_wise_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Position-wise KDE plot saved to {position_wise_path}")
    
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
    
    plt.title('Aggregate Distribution of Early Stopping Probabilities\nAcross All Positions', 
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
    print(f"Total number of position-probability pairs: {len(expanded_df)}")
    print("\nProbability distribution by position:")
    position_stats = expanded_df.groupby('position')['probability'].agg(['mean', 'std', 'min', 'max'])
    print(position_stats)

if __name__ == "__main__":
    main()
