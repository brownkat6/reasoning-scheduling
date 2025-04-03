import argparse
import re
import numpy as np
import pandas as pd
import os
import wandb

import torch
import torch.nn as nn
import torch.optim as optim

from scipy.stats import pearsonr
from datasets import load_dataset
from Dynasor.benchmark.TokenDeprivation import utils
from Dynasor.benchmark.TokenDeprivation.run import execute_question_reuse
from Dynasor.benchmark.TokenDeprivation import run

from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
from scipy import stats

# Script to train an MLP and report metrics on a train and test set

def train_mlp(train_data_dir_X='', train_data_dir_Y='', train_split='train', train_dataset='gsm8k',
              test_data_dir_X='', test_data_dir_Y='', test_split='test', test_dataset='gsm8k',
              num_epochs=20, batch_size=4, learning_rate=1e-3,
              X_key='hidden_state', use_wandb=True, project_name="early-stopping-mlp"):
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
        X_key: Key to use for input features
        use_wandb: Whether to log to wandb
        project_name: wandb project name
    """
    # Initialize wandb
    if use_wandb:
        run_name = f"{train_dataset}_{train_split}_to_{test_dataset}_{test_split}"
        wandb.init(project=project_name, name=run_name, config={
            "train_dataset": train_dataset,
            "train_split": train_split,
            "test_dataset": test_dataset,
            "test_split": test_split,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "X_key": X_key
        })
    
    # Validate input directories
    if not os.path.exists(train_data_dir_X):
        raise ValueError(f"Training X data directory {train_data_dir_X} not found.")
    if not os.path.exists(train_data_dir_Y):
        raise ValueError(f"Training Y data directory {train_data_dir_Y} not found.")
    if not os.path.exists(test_data_dir_X):
        raise ValueError(f"Test X data directory {test_data_dir_X} not found.")
    if not os.path.exists(test_data_dir_Y):
        raise ValueError(f"Test Y data directory {test_data_dir_Y} not found.")
    
    train_dataset_name=train_dataset
    test_dataset_name=test_dataset
    
    def load_and_merge_data(data_dir_X, data_dir_Y, dataset, split):
        # Check if grouped file already exists
        grouped_file = f"{data_dir_Y}/{dataset}_grouped_{split}.csv"
        if os.path.exists(grouped_file):
            data = pd.read_csv(grouped_file)
            if use_wandb:
                wandb.log({f"{dataset}_{split}_data_loaded": True, 
                          f"{dataset}_{split}_shape": data.shape})
            return data
            
        # If not, load and merge X and Y files
        all_merged_data = []
        batch_idx = 0
        while True:
            x_file = f"{data_dir_X}/{dataset}_X_{split}_{batch_idx}.csv"
            y_file = f"{data_dir_Y}/{dataset}_Y_{split}_{batch_idx}.csv"
            
            # Break if we've processed all batches
            if not (os.path.exists(x_file) and os.path.exists(y_file)):
                break
                
            # Load X and Y data
            x_data = pd.read_csv(x_file)
            y_data = pd.read_csv(y_file)
            
            # Merge on common keys
            merged_data = pd.merge(
                x_data, 
                y_data,
                on=['question_id', 'dataset', 'split', 'question_text'],
                how='inner'
            )
            # print(merged_data.columns)
            merged_data["batch_idx"] = batch_idx
            
            all_merged_data.append(merged_data)
            batch_idx += 1
            
        if not all_merged_data:
            raise ValueError(f"No data files found in {data_dir_X} and {data_dir_Y} for {dataset} {split}")
            
        # Concatenate all batches
        final_data = pd.concat(all_merged_data, ignore_index=True)
        
        # Save grouped data
        print(f"Saving grouped data to {grouped_file} with shape {final_data.shape} and columns {final_data.columns}")
        final_data.to_csv(grouped_file, index=False)
        
        if use_wandb:
            wandb.log({f"{dataset}_{split}_data_created": True, 
                      f"{dataset}_{split}_shape": final_data.shape,
                      f"{dataset}_{split}_batch_count": batch_idx})
        
        return final_data
    
    # Load and merge training data
    print(f"Loading and merging training data for {train_dataset} {train_split}")
    print(f"X data from: {train_data_dir_X}")
    print(f"Y data from: {train_data_dir_Y}")
    train_data = load_and_merge_data(train_data_dir_X, train_data_dir_Y, train_dataset, train_split)
    
    # Load and merge test data
    print(f"Loading and merging test data for {test_dataset} {test_split}")
    print(f"X data from: {test_data_dir_X}")
    print(f"Y data from: {test_data_dir_Y}")
    test_data = load_and_merge_data(test_data_dir_X, test_data_dir_Y, test_dataset, test_split)
    
    # Parse lists from string representation
    import ast
    def parse_list(x):
        try:
            return ast.literal_eval(x)
        except Exception:
            return x

    train_data['hidden_state'] = train_data['hidden_state'].apply(parse_list)
    train_data['early_stop_correct_proportions'] = train_data['early_stop_correct_proportions'].apply(parse_list)
    test_data['hidden_state'] = test_data['hidden_state'].apply(parse_list)
    test_data['early_stop_correct_proportions'] = test_data['early_stop_correct_proportions'].apply(parse_list)

    # Prepare training data
    X_train = np.vstack(train_data[X_key].values)
    Y_train = np.vstack(train_data['early_stop_correct_proportions'].values)
    
    # Prepare test data
    X_test = np.vstack(test_data[X_key].values)
    Y_test = np.vstack(test_data['early_stop_correct_proportions'].values)

    print(f"Training questions: {X_train.shape[0]}, Testing questions: {X_test.shape[0]}")
    if use_wandb:
        wandb.log({
            "train_questions": X_train.shape[0],
            "test_questions": X_test.shape[0],
            "feature_dim": X_train.shape[1],
            "output_dim": Y_train.shape[1]
        })

    from mlp import MLP

    model_mlp = MLP(input_dim=1536, hidden_dim=256, output_dim=Y_train.shape[1]).to('cuda')  # Move model to CUDA
    if use_wandb:
        wandb.watch(model_mlp)  # Track gradients and parameters
    
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
        if use_wandb:
            wandb.log({"test_output_truncated": True, 
                      "truncated_test_output_dim": Y_test_tensor.shape[1]})

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
        
        # Evaluate on test set for this epoch
        model_mlp.eval()
        with torch.no_grad():
            test_outputs = model_mlp(X_test_tensor)
            test_loss = criterion(test_outputs, Y_test_tensor).item()
        
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": epoch_loss,
                "test_loss": test_loss
            })

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
    
    # Log overall metrics to wandb
    if use_wandb:
        wandb.log({
            "mse_train": mse_train_overall,
            "mse_test": mse_test_overall,
            "baseline_mse_train": baseline_mse_train,
            "baseline_mse_test": baseline_mse_test,
            "baseline_mse_train_test_means": baseline_mse_train_test_means,
            "baseline_mse_test_test_means": baseline_mse_test_test_means,
            "train_improvement_pct": train_improvement,
            "test_improvement_pct": test_improvement,
            "train_improvement_test_means_pct": train_improvement_test_means,
            "test_improvement_test_means_pct": test_improvement_test_means,
        })

    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save the model
    model_filename = f'models/mlp_{train_dataset_name}_{train_split}.pt'
    torch.save({
        'model_state_dict': model_mlp.state_dict(),
        'train_means': train_means,  # Save training means for future reference
        'config': {
            'input_dim': 1536,
            'hidden_dim': 256,
            'output_dim': Y_train.shape[1],
            'train_dataset': train_dataset,
            'train_split': train_split,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }
    }, model_filename)
    print(f"\nModel saved to {model_filename}")
    
    # Log to wandb that model was saved
    if use_wandb:
        wandb.save(model_filename)
        wandb.log({"model_saved": True})

    # Also print out the means themselves for comparison
    print("\nMean Values:")
    print("Position  Train_Mean  Test_Mean   Diff")
    print("-" * 45)
    
    # Create a table for wandb
    if use_wandb:
        wandb_table_means = wandb.Table(columns=["Position", "Train_Mean", "Test_Mean", "Diff"])
    
    for i, (train_mean, test_mean) in enumerate(zip(train_means, test_means)):
        diff = abs(train_mean - test_mean)
        print(f"{i+1:8d}  {train_mean:.4f}     {test_mean:.4f}     {diff:.4f}")
        
        # Add to wandb table
        if use_wandb:
            wandb_table_means.add_data(i+1, train_mean, test_mean, diff)
    
    # Log table to wandb
    if use_wandb:
        wandb.log({"mean_values_table": wandb_table_means})

    mse_train_individual = np.mean((train_pred - Y_train_np)**2, axis=0)
    mse_test_individual = np.mean((test_pred - Y_test_np)**2, axis=0)

    pearson_train = [pearsonr(train_pred[:, i], Y_train_np[:, i])[0] for i in range(Y_train_np.shape[1])]
    pearson_test = [pearsonr(test_pred[:, i], Y_test_np[:, i])[0] for i in range(Y_test_np.shape[1])]

    print("\nEarly stopping position-wise MSE and Pearson correlation (Train):")
    # Create wandb tables for position-wise metrics
    if use_wandb:
        wandb_table_train = wandb.Table(columns=["Position", "MSE", "Pearson"])
        wandb_table_test = wandb.Table(columns=["Position", "MSE", "Pearson"])
    
    for i, (mse_val, p_val) in enumerate(zip(mse_train_individual, pearson_train)):
        print(f"Position {i+1}: MSE: {mse_val:.4f}, Pearson: {p_val:.4f}")
        if use_wandb:
            wandb_table_train.add_data(i+1, mse_val, p_val)
            wandb.log({f"train_position_{i+1}_mse": mse_val, f"train_position_{i+1}_pearson": p_val})

    print("\nEarly stopping position-wise MSE and Pearson correlation (Test):")
    for i, (mse_val, p_val) in enumerate(zip(mse_test_individual, pearson_test)):
        print(f"Position {i+1}: MSE: {mse_val:.4f}, Pearson: {p_val:.4f}")
        if use_wandb:
            wandb_table_test.add_data(i+1, mse_val, p_val)
            wandb.log({f"test_position_{i+1}_mse": mse_val, f"test_position_{i+1}_pearson": p_val})
    
    # Log tables to wandb
    if use_wandb:
        wandb.log({
            "train_positions_metrics": wandb_table_train,
            "test_positions_metrics": wandb_table_test
        })

    # Create figures directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)
    
    # Calculate number of subplots needed (all positions + 1 for aggregate)
    n_positions = Y_train_np.shape[1]
    n_rows = (n_positions + 2) // 3  # 3 plots per row, +2 for ceiling division
    
    plt.figure(figsize=(15, 5*n_rows))
    
    from scipy import stats
    
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
    
    # Log the figure to wandb
    if use_wandb:
        wandb.log({"prediction_correlation_plots": wandb.Image(os.path.join('figures', figure_name))})
    
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
        
        # Log skewness metrics to wandb
        if use_wandb:
            wandb.log({
                f"train_position_{i+1}_residual_skew": stats.skew(train_residuals),
                f"test_position_{i+1}_residual_skew": stats.skew(test_residuals)
            })
    
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
    
    # Log aggregate skewness metrics to wandb
    if use_wandb:
        wandb.log({
            "train_all_positions_residual_skew": stats.skew(train_residuals_flat),
            "test_all_positions_residual_skew": stats.skew(test_residuals_flat)
        })
    
    plt.tight_layout()
    residuals_figure_name = f'residuals_normality_plots_{train_dataset_name}_{train_split}_to_{test_dataset_name}_{test_split}.png'
    plt.savefig(os.path.join('figures', residuals_figure_name), dpi=300, bbox_inches='tight')
    
    # Log the residuals figure to wandb
    if use_wandb:
        wandb.log({"residuals_normality_plots": wandb.Image(os.path.join('figures', residuals_figure_name))})
    
    plt.close()
    
    print(f"\nVisualization saved to figures/{residuals_figure_name}")
    
    # Create histograms of predictions vs actuals
    plt.figure(figsize=(15, 5*n_rows))
    
    for i in range(n_positions):
        plt.subplot(n_rows, 3, i+1)
        
        plt.hist(Y_train_np[:, i], bins=20, alpha=0.5, label='Train Actual', color='blue')
        plt.hist(train_pred[:, i], bins=20, alpha=0.5, label='Train Pred', color='lightblue')
        plt.hist(Y_test_np[:, i], bins=20, alpha=0.5, label='Test Actual', color='red')
        plt.hist(test_pred[:, i], bins=20, alpha=0.5, label='Test Pred', color='lightcoral')
        
        plt.xlabel('Probability')
        plt.ylabel('Count')
        plt.title(f'Position {i+1} Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Plot for aggregate
    plt.subplot(n_rows, 3, n_positions+1)
    plt.hist(Y_train_flat, bins=20, alpha=0.5, label='Train Actual', color='blue')
    plt.hist(train_pred_flat, bins=20, alpha=0.5, label='Train Pred', color='lightblue')
    plt.hist(Y_test_flat, bins=20, alpha=0.5, label='Test Actual', color='red')
    plt.hist(test_pred_flat, bins=20, alpha=0.5, label='Test Pred', color='lightcoral')
    
    plt.xlabel('Probability')
    plt.ylabel('Count')
    plt.title('All Positions Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    histogram_figure_name = f'distribution_histograms_{train_dataset_name}_{train_split}_to_{test_dataset_name}_{test_split}.png'
    plt.savefig(os.path.join('figures', histogram_figure_name), dpi=300, bbox_inches='tight')
    
    # Log the histograms to wandb
    if use_wandb:
        wandb.log({"distribution_histograms": wandb.Image(os.path.join('figures', histogram_figure_name))})
    
    plt.close()
    
    print(f"\nHistogram visualization saved to figures/{histogram_figure_name}")
    
    # Create position-wise error plots
    plt.figure(figsize=(10, 6))
    positions = np.arange(1, len(mse_train_individual) + 1)
    
    plt.plot(positions, mse_train_individual, 'bo-', label='Train MSE')
    plt.plot(positions, mse_test_individual, 'ro-', label='Test MSE')
    
    plt.xlabel('Position')
    plt.ylabel('Mean Squared Error')
    plt.title('Position-wise MSE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(positions)
    
    mse_position_plot = f'position_wise_mse_{train_dataset_name}_{train_split}_to_{test_dataset_name}_{test_split}.png'
    plt.savefig(os.path.join('figures', mse_position_plot), dpi=300, bbox_inches='tight')
    
    if use_wandb:
        wandb.log({"position_wise_mse": wandb.Image(os.path.join('figures', mse_position_plot))})
    
    plt.close()
    
    # Create position-wise Pearson correlation plots
    plt.figure(figsize=(10, 6))
    
    plt.plot(positions, pearson_train, 'bs-', label='Train Pearson r')
    plt.plot(positions, pearson_test, 'rs-', label='Test Pearson r')
    
    plt.xlabel('Position')
    plt.ylabel('Pearson Correlation')
    plt.title('Position-wise Pearson Correlation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(positions)
    plt.ylim(-0.1, 1.1)
    
    pearson_position_plot = f'position_wise_pearson_{train_dataset_name}_{train_split}_to_{test_dataset_name}_{test_split}.png'
    plt.savefig(os.path.join('figures', pearson_position_plot), dpi=300, bbox_inches='tight')
    
    if use_wandb:
        wandb.log({"position_wise_pearson": wandb.Image(os.path.join('figures', pearson_position_plot))})
    
    plt.close()
    
    # Create a heatmap of predicted vs actual values for each position
    plt.figure(figsize=(15, 5))
    
    # Train data heatmap
    plt.subplot(1, 2, 1)
    heatmap_train = np.zeros((10, len(positions)))
    for i in range(len(positions)):
        hist, _ = np.histogram(Y_train_np[:, i] - train_pred[:, i], bins=10, range=(-0.5, 0.5))
        heatmap_train[:, i] = hist / np.sum(hist)  # Normalize by column
    
    plt.imshow(heatmap_train, aspect='auto', cmap='viridis')
    plt.colorbar(label='Normalized Frequency')
    plt.xlabel('Position')
    plt.ylabel('Error Bin')
    plt.title('Train Error Distribution by Position')
    plt.xticks(range(len(positions)), positions)
    plt.yticks(range(10), [f"{i/10-0.5:.1f}" for i in range(10)])
    
    # Test data heatmap
    plt.subplot(1, 2, 2)
    heatmap_test = np.zeros((10, len(positions)))
    for i in range(len(positions)):
        hist, _ = np.histogram(Y_test_np[:, i] - test_pred[:, i], bins=10, range=(-0.5, 0.5))
        heatmap_test[:, i] = hist / np.sum(hist)  # Normalize by column
    
    plt.imshow(heatmap_test, aspect='auto', cmap='viridis')
    plt.colorbar(label='Normalized Frequency')
    plt.xlabel('Position')
    plt.ylabel('Error Bin')
    plt.title('Test Error Distribution by Position')
    plt.xticks(range(len(positions)), positions)
    plt.yticks(range(10), [f"{i/10-0.5:.1f}" for i in range(10)])
    
    plt.tight_layout()
    heatmap_plot = f'error_heatmap_{train_dataset_name}_{train_split}_to_{test_dataset_name}_{test_split}.png'
    plt.savefig(os.path.join('figures', heatmap_plot), dpi=300, bbox_inches='tight')
    
    if use_wandb:
        wandb.log({"error_heatmap": wandb.Image(os.path.join('figures', heatmap_plot))})
    
    plt.close()
    
    # Create a confusion matrix style visualization for binary classification
    # Threshold the probabilities at 0.5 to create binary predictions
    binary_threshold = 0.5
    
    plt.figure(figsize=(15, 5*n_rows))
    confusion_metrics = {}
    
    for i in range(n_positions):
        plt.subplot(n_rows, 3, i+1)
        
        # Convert to binary
        train_actual_bin = (Y_train_np[:, i] >= binary_threshold).astype(int)
        train_pred_bin = (train_pred[:, i] >= binary_threshold).astype(int)
        test_actual_bin = (Y_test_np[:, i] >= binary_threshold).astype(int)
        test_pred_bin = (test_pred[:, i] >= binary_threshold).astype(int)
        
        # Calculate metrics
        train_acc = np.mean(train_actual_bin == train_pred_bin)
        test_acc = np.mean(test_actual_bin == test_pred_bin)
        
        # True Positives, False Positives, etc.
        train_tp = np.sum((train_actual_bin == 1) & (train_pred_bin == 1))
        train_fp = np.sum((train_actual_bin == 0) & (train_pred_bin == 1))
        train_fn = np.sum((train_actual_bin == 1) & (train_pred_bin == 0))
        train_tn = np.sum((train_actual_bin == 0) & (train_pred_bin == 0))
        
        test_tp = np.sum((test_actual_bin == 1) & (test_pred_bin == 1))
        test_fp = np.sum((test_actual_bin == 0) & (test_pred_bin == 1))
        test_fn = np.sum((test_actual_bin == 1) & (test_pred_bin == 0))
        test_tn = np.sum((test_actual_bin == 0) & (test_pred_bin == 0))
        
        # Calculate F1 score, precision, recall
        train_precision = train_tp / (train_tp + train_fp) if (train_tp + train_fp) > 0 else 0
        train_recall = train_tp / (train_tp + train_fn) if (train_tp + train_fn) > 0 else 0
        train_f1 = 2 * train_precision * train_recall / (train_precision + train_recall) if (train_precision + train_recall) > 0 else 0
        
        test_precision = test_tp / (test_tp + test_fp) if (test_tp + test_fp) > 0 else 0
        test_recall = test_tp / (test_tp + test_fn) if (test_tp + test_fn) > 0 else 0
        test_f1 = 2 * test_precision * test_recall / (test_precision + test_recall) if (test_precision + test_recall) > 0 else 0
        
        # Store metrics
        confusion_metrics[f"position_{i+1}"] = {
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "train_precision": train_precision,
            "train_recall": train_recall,
            "train_f1": train_f1,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1
        }
        
        if use_wandb:
            wandb.log({
                f"train_position_{i+1}_accuracy": train_acc,
                f"test_position_{i+1}_accuracy": test_acc,
                f"train_position_{i+1}_precision": train_precision,
                f"train_position_{i+1}_recall": train_recall,
                f"train_position_{i+1}_f1": train_f1,
                f"test_position_{i+1}_precision": test_precision,
                f"test_position_{i+1}_recall": test_recall,
                f"test_position_{i+1}_f1": test_f1
            })
        
        # Create "confusion matrix" style plot
        plt.text(0.3, 0.7, f"Train TP: {train_tp}", fontsize=10, color='blue')
        plt.text(0.3, 0.65, f"Train FP: {train_fp}", fontsize=10, color='blue')
        plt.text(0.3, 0.6, f"Train FN: {train_fn}", fontsize=10, color='blue')
        plt.text(0.3, 0.55, f"Train TN: {train_tn}", fontsize=10, color='blue')
        
        plt.text(0.6, 0.7, f"Test TP: {test_tp}", fontsize=10, color='red')
        plt.text(0.6, 0.65, f"Test FP: {test_fp}", fontsize=10, color='red')
        plt.text(0.6, 0.6, f"Test FN: {test_fn}", fontsize=10, color='red')
        plt.text(0.6, 0.55, f"Test TN: {test_tn}", fontsize=10, color='red')
        
        plt.text(0.3, 0.45, f"Train Acc: {train_acc:.3f}", fontsize=10, color='blue')
        plt.text(0.3, 0.4, f"Train Prec: {train_precision:.3f}", fontsize=10, color='blue')
        plt.text(0.3, 0.35, f"Train Rec: {train_recall:.3f}", fontsize=10, color='blue')
        plt.text(0.3, 0.3, f"Train F1: {train_f1:.3f}", fontsize=10, color='blue')
        
        plt.text(0.6, 0.45, f"Test Acc: {test_acc:.3f}", fontsize=10, color='red')
        plt.text(0.6, 0.4, f"Test Prec: {test_precision:.3f}", fontsize=10, color='red')
        plt.text(0.6, 0.35, f"Test Rec: {test_recall:.3f}", fontsize=10, color='red')
        plt.text(0.6, 0.3, f"Test F1: {test_f1:.3f}", fontsize=10, color='red')
        
        plt.axis('off')
        plt.title(f'Position {i+1} Binary Classification Metrics')
    
    plt.tight_layout()
    binary_metrics_plot = f'binary_metrics_{train_dataset_name}_{train_split}_to_{test_dataset_name}_{test_split}.png'
    plt.savefig(os.path.join('figures', binary_metrics_plot), dpi=300, bbox_inches='tight')
    
    if use_wandb:
        wandb.log({"binary_metrics": wandb.Image(os.path.join('figures', binary_metrics_plot))})
    
    plt.close()
    
    # Log summary tables to wandb
    if use_wandb:
        # Create a summary table of all key metrics
        summary_table = wandb.Table(columns=["Metric", "Train", "Test"])
        summary_table.add_data("MSE (Model)", mse_train_overall, mse_test_overall)
        summary_table.add_data("MSE (Train Means Baseline)", baseline_mse_train, baseline_mse_test)
        summary_table.add_data("MSE (Test Means Baseline)", baseline_mse_train_test_means, baseline_mse_test_test_means)
        summary_table.add_data("Improvement over Train Means (%)", train_improvement, test_improvement)
        summary_table.add_data("Improvement over Test Means (%)", train_improvement_test_means, test_improvement_test_means)
        summary_table.add_data("Pearson r (All Positions)", r_train, r_test)
        
        wandb.log({"summary_metrics": summary_table})
        
        # Create a final run summary
        wandb.run.summary.update({
            "final_train_mse": mse_train_overall,
            "final_test_mse": mse_test_overall,
            "final_train_pearson": r_train,
            "final_test_pearson": r_test,
            "model_improvement_pct": test_improvement
        })
        
        # Finish the wandb run
        wandb.finish()
    
    print("\nTraining and evaluation complete!")
    return model_mlp


def main():
    from constants import X_STEM, Y_STEM
    parser = argparse.ArgumentParser(description="MLP test experiment for reasoning traces with wandb logging")
    # New arguments for refactored train_mlp
    parser.add_argument("--train_dataset", type=str, default='gsm8k', choices=['gsm8k', 'math500', 'numina'], help="Dataset for training")
    parser.add_argument("--train_split", type=str, default='train', choices=['train', 'test'], help="Split to use for training")
    parser.add_argument("--test_dataset", type=str, default='gsm8k', choices=['gsm8k', 'math500', 'numina'], help="Dataset for testing")
    parser.add_argument("--test_split", type=str, default='test', choices=['train', 'test'], help="Split to use for testing")
    parser.add_argument("--X-STEM", type=str, default=X_STEM, help="Base directory for X data")
    parser.add_argument("--Y-STEM", type=str, default=Y_STEM, help="Base directory for Y data")
    parser.add_argument("--X-key", type=str, default='hidden_state', choices=['hidden_state'], help="Key to use for input to MLP")
    parser.add_argument("--num-epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for training")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="early-stopping-mlp", help="wandb project name")
    
    args = parser.parse_args()
    
    # Construct full paths for X and Y data
    
    train_data_dir_X = os.path.join(X_STEM, f"{args.train_dataset}_results")
    train_data_dir_Y = os.path.join(Y_STEM, f"{args.train_dataset}_results")
    test_data_dir_X = os.path.join(X_STEM, f"{args.test_dataset}_results")
    test_data_dir_Y = os.path.join(Y_STEM, f"{args.test_dataset}_results")
    
    print(f"X data directories:\n Train: {train_data_dir_X}\n Test: {test_data_dir_X}")
    print(f"Y data directories:\n Train: {train_data_dir_Y}\n Test: {test_data_dir_Y}")
    
    train_mlp(
            train_data_dir_X=train_data_dir_X,
            train_data_dir_Y=train_data_dir_Y,
            train_split=args.train_split,
            train_dataset=args.train_dataset,
            test_data_dir_X=test_data_dir_X,
            test_data_dir_Y=test_data_dir_Y,
            test_split=args.test_split,
            test_dataset=args.test_dataset,
            X_key=args.X_key,
            use_wandb=not args.no_wandb,
            project_name=args.wandb_project,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
    )


if __name__ == "__main__":
    main()