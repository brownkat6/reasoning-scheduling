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

from mlp import MLP
# Updated MLP class to support different architectures


def train_mlp_sweep(config=None):
    """
    Run a single training with the given wandb config.
    This function is called by wandb.agent.
    """
    with wandb.init(config=config):
        # Access all hyperparameters through wandb.config
        config = wandb.config
        
        # Get paths
        from constants import X_STEM, Y_STEM
        #X_STEM = "/n/netscratch/gershman_lab/Lab/amuppidi/reasoning_scheduling_new_orig/data/"
        #Y_STEM = "/n/netscratch/gershman_lab/Lab/amuppidi/reasoning_scheduling_new/data/"
        
        train_data_dir_X = os.path.join(X_STEM, f"{config.train_dataset}_results")
        train_data_dir_Y = os.path.join(Y_STEM, f"{config.train_dataset}_results")
        test_data_dir_X = os.path.join(X_STEM, f"{config.test_dataset}_results")
        test_data_dir_Y = os.path.join(Y_STEM, f"{config.test_dataset}_results")
        
        # Set up our parameters from the sweep config
        hidden_layer = config.hidden_layer
        hidden_dims = config.hidden_dims
        activation = config.activation
        learning_rate = config.learning_rate
        weight_decay = config.weight_decay
        dropout = config.dropout
        batch_size = config.batch_size
        
        # Run training with these parameters
        train_mlp(
            train_data_dir_X=train_data_dir_X,
            train_data_dir_Y=train_data_dir_Y,
            train_split=config.train_split,
            train_dataset=config.train_dataset,
            test_data_dir_X=test_data_dir_X,
            test_data_dir_Y=test_data_dir_Y,
            test_split=config.test_split,
            test_dataset=config.test_dataset,
            num_epochs=config.num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            X_key=config.X_key,
            hidden_layer=hidden_layer,
            hidden_dims=hidden_dims,
            activation=activation,
            weight_decay=weight_decay,
            dropout=dropout,
            use_wandb=True  # Always use wandb in sweep mode
        )


def train_mlp(train_data_dir_X='', train_data_dir_Y='', train_split='train', train_dataset='gsm8k',
              test_data_dir_X='', test_data_dir_Y='', test_split='test', test_dataset='gsm8k',
              num_epochs=20, batch_size=4, learning_rate=1e-3, weight_decay=0,
              X_key='hidden_state', hidden_layer=None, hidden_dims=[256], activation='relu', 
              dropout=0.0, use_wandb=True, project_name="reasoning-scheduling"):
    """
    Train an MLP to predict the early stopping correctness proportions from the hidden state.
    Allows training on one dataset/split and testing on another.
    
    Args:
        train_data_dir_X: Directory containing the training X data files
        train_data_dir_Y: Directory containing the training Y data files
        train_split: Split to use for training ('train' or 'test')
        train_dataset: Dataset name for training data
        test_data_dir_X: Directory containing the test X data files
        test_data_dir_Y: Directory containing the test Y data files
        test_split: Split to use for testing ('train' or 'test')
        test_dataset: Dataset name for test data
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        weight_decay: L2 regularization weight decay parameter
        X_key: Key to use for input features to MLP
        hidden_layer: Specific hidden layer to use, if needed
        hidden_dims: List of hidden layer dimensions
        activation: Activation function to use ('relu', 'gelu', 'sigmoid')
        dropout: Dropout rate to use
        use_wandb: Whether to log to Weights & Biases
        project_name: WandB project name
    """
    # Validate input directories
    for dir_path in [train_data_dir_X, train_data_dir_Y, test_data_dir_X, test_data_dir_Y]:
        if not os.path.exists(dir_path):
            raise ValueError(f"Data directory {dir_path} not found.")
    
    train_dataset_name = train_dataset
    test_dataset_name = test_dataset
    
    # Initialize wandb if requested (and not already initialized)
    if use_wandb and not wandb.run:
        run_name = f"{train_dataset_name}_{train_split}_to_{test_dataset_name}_{test_split}"
        if hidden_layer:
            run_name += f"_layer_{hidden_layer}"
        
        wandb.init(
            project=project_name,
            name=run_name,
            config={
                "train_dataset": train_dataset_name,
                "train_split": train_split,
                "test_dataset": test_dataset_name,
                "test_split": test_split,
                "hidden_layer": hidden_layer,
                "hidden_dims": hidden_dims,
                "activation": activation,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "dropout": dropout,
                "X_key": X_key
            }
        )
        print(f"WandB initialized with run name: {run_name}")
    
    def load_and_merge_data(data_dir_X, data_dir_Y, dataset, split, hidden_layer=None):
        # Adjust X directory if hidden_layer is specified
        if hidden_layer and os.path.exists(os.path.join(data_dir_X, f"layer_{hidden_layer}")):
            data_dir_X = os.path.join(data_dir_X, f"layer_{hidden_layer}")
            print(f"Using hidden layer '{hidden_layer}' data from {data_dir_X}")
        
        # Check if grouped file already exists
        grouped_file = f"{data_dir_Y}/{dataset}_grouped_{split}.csv"
        if os.path.exists(grouped_file):
            return pd.read_csv(grouped_file)
            
        # If not, load and merge X and Y files
        all_merged_data = []
        batch_idx = 0
        
        while True:
            x_file = f"{data_dir_X}/{dataset}_X_{split}_{batch_idx}.csv"
            y_file = f"{data_dir_Y}/{dataset}_Y_{split}_{batch_idx}.csv"
            
            # Break if we've processed all batches
            if not (os.path.exists(x_file) and os.path.exists(y_file)):
                if batch_idx == 0:
                    print(f"No data files found at batch 0. X path: {x_file}, Y path: {y_file}")
                break
                
            # Load X and Y data
            print(f"Loading X data from {x_file}")
            x_data = pd.read_csv(x_file)
            print(f"Loading Y data from {y_file}")
            y_data = pd.read_csv(y_file)
            
            # Merge on common keys
            merged_data = pd.merge(
                x_data, 
                y_data,
                on=['question_id', 'dataset', 'split', 'question_text'],
                how='inner'
            )
            print(f"Merged batch {batch_idx}: {merged_data.shape[0]} rows")
            merged_data["batch_idx"] = batch_idx
            
            all_merged_data.append(merged_data)
            batch_idx += 1
            
        if not all_merged_data:
            raise ValueError(f"No matching data files found for {dataset} {split} in X dir: {data_dir_X} and Y dir: {data_dir_Y}")
            
        # Concatenate all batches
        final_data = pd.concat(all_merged_data, ignore_index=True)
        
        # Save grouped data
        print(f"Saving grouped data to {grouped_file} with shape {final_data.shape} and columns {final_data.columns}")
        final_data.to_csv(grouped_file, index=False)
        
        return final_data
    
    # Load and merge training data
    print(f"Loading and merging training data from X: {train_data_dir_X} and Y: {train_data_dir_Y} for {train_dataset} {train_split}")
    train_data = load_and_merge_data(train_data_dir_X, train_data_dir_Y, train_dataset, train_split, hidden_layer)
    
    # Load and merge test data
    print(f"Loading and merging test data from X: {test_data_dir_X} and Y: {test_data_dir_Y} for {test_dataset} {test_split}")
    test_data = load_and_merge_data(test_data_dir_X, test_data_dir_Y, test_dataset, test_split, hidden_layer)
    
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

    # Create model with the specified architecture parameters
    
    model_mlp = MLP(input_dim=1536, hidden_dim=256, output_dim=Y_train.shape[1]).to('cuda')  # Move model to CUDA

    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_mlp.parameters(), lr=learning_rate, weight_decay=weight_decay)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to('cuda')
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).to('cuda')
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to('cuda')
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32).to('cuda')
    
    print(f"Training data shape: {X_train_tensor.shape}, {Y_train_tensor.shape}")
    print(f"Testing data shape: {X_test_tensor.shape}, {Y_test_tensor.shape}")
    
    # Log dataset details to wandb
    if use_wandb:
        wandb.run.summary["train_samples"] = X_train_tensor.shape[0]
        wandb.run.summary["test_samples"] = X_test_tensor.shape[0]
        wandb.run.summary["input_dim"] = X_train_tensor.shape[1]
        wandb.run.summary["output_dim"] = Y_train_tensor.shape[1]
    
    # if Y_train has shape (N, D) and Y_test has shape (N, F) where F>D, truncate Y_test to have shape (N,D)
    if Y_test_tensor.shape[1] > Y_train_tensor.shape[1]:
        Y_test_tensor = Y_test_tensor[:, :Y_train_tensor.shape[1]]
        print(f"Truncated testing data shape: {Y_test_tensor.shape}")
        if use_wandb:
            wandb.run.summary["truncated_test_output_dim"] = Y_test_tensor.shape[1]

    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, Y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print("Starting MLP training...")
    # Track best metrics for early stopping & model saving
    best_test_mse = float('inf')
    
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
        
        # Calculate validation loss as well
        model_mlp.eval()
        with torch.no_grad():
            val_outputs = model_mlp(X_test_tensor)
            val_loss = criterion(val_outputs, Y_test_tensor).item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Log metrics to wandb
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": epoch_loss,
                "val_loss": val_loss
            })
            
        # Check if this is the best model so far
        if val_loss < best_test_mse:
            best_test_mse = val_loss
            if use_wandb:
                wandb.run.summary["best_val_loss"] = best_test_mse

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
    
    # Add layer and model info to names
    layer_str = f"_layer_{hidden_layer}" if hidden_layer is not None else ""
    arch_str = f"_arch_{hidden_dims[0]}" if isinstance(hidden_dims, (list, tuple)) else f"_arch_{hidden_dims}"
    model_suffix = f"{layer_str}{arch_str}_act_{activation}_drop_{dropout:.2f}"
    
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
    
    # Log model performance metrics to wandb
    if use_wandb:
        wandb.run.summary.update({
            # Model performance
            "final_train_mse": mse_train_overall,
            "final_test_mse": mse_test_overall,
            
            # Baseline performance (train means)
            "baseline_train_mse": baseline_mse_train,
            "baseline_test_mse": baseline_mse_test,
            
            # Baseline performance (test means)
            "baseline_train_mse_test_means": baseline_mse_train_test_means,
            "baseline_test_mse_test_means": baseline_mse_test_test_means,
            
            # Relative improvements
            "train_improvement_pct": train_improvement,
            "test_improvement_pct": test_improvement,
            "train_improvement_test_means_pct": train_improvement_test_means,
            "test_improvement_test_means_pct": test_improvement_test_means
        })

    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Modify model saving to match run_adaptive.py's expected format
    # Extract layer info for filename
    layer_str = f"_layer_{hidden_layer}" if hidden_layer is not None else ""
    
    # Extract architecture info
    arch_str = f"_arch_{hidden_dims[0]}" if isinstance(hidden_dims, (list, tuple)) else f"_arch_{hidden_dims}"
    
    # Construct model filename
    #model_filename = f'models/mlp_{train_dataset_name}_{train_split}{model_suffix}.pt'
    model_filename = f'models/mlp_{train_dataset_name}_{train_split}{layer_str}{arch_str}_act_{activation}_drop_{dropout:.2f}.pt'
    
    # Save the model with all necessary information
    torch.save({
        'model_state_dict': model_mlp.state_dict(),
        'train_means': train_means,  # Save training means for future reference
        'config': {
            'input_dim': X_train.shape[1],
            'hidden_dims': hidden_dims,
            'output_dim': Y_train.shape[1],
            'activation': activation,
            'dropout': dropout,
            'train_dataset': train_dataset,
            'train_split': train_split,
            'hidden_layer': hidden_layer,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay
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
    figure_name = f'prediction_correlation_plots_{train_dataset_name}_{train_split}_to_{test_dataset_name}_{test_split}{model_suffix}.png'
    plt.savefig(os.path.join('figures', figure_name), dpi=300, bbox_inches='tight')
    
    # Log figure to wandb
    if use_wandb:
        wandb.log({"aggregate_prediction_plot": wandb.Image(plt)})
        # Also log the correlation coefficients
        wandb.run.summary.update({
            "train_r_overall": r_train,
            "test_r_overall": r_test
        })
    
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
    residuals_figure_name = f'residuals_normality_plots_{train_dataset_name}_{train_split}_to_{test_dataset_name}_{test_split}{model_suffix}.png'
    plt.savefig(os.path.join('figures', residuals_figure_name), dpi=300, bbox_inches='tight')
    
    # Log residuals figure to wandb
    if use_wandb:
        wandb.log({"residuals_qq_plot": wandb.Image(plt)})
    
    plt.close()
    
    print(f"\nVisualization saved to figures/{residuals_figure_name}")
    
    # Create and log position-wise metrics plots
    if use_wandb:
        # Create a plot for position-wise MSE
        plt.figure(figsize=(10, 6))
        positions = list(range(1, n_positions + 1))
        plt.plot(positions, mse_train_individual, 'bo-', label='Train MSE')
        plt.plot(positions, mse_test_individual, 'ro-', label='Test MSE')
        plt.xlabel('Position')
        plt.ylabel('MSE')
        plt.title('Position-wise MSE')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(positions)
        wandb.log({"position_wise_mse": wandb.Image(plt)})
        plt.close()
        
        # Create a plot for position-wise Pearson correlation
        plt.figure(figsize=(10, 6))
        plt.plot(positions, pearson_train, 'bo-', label='Train Pearson r')
        plt.plot(positions, pearson_test, 'ro-', label='Test Pearson r')
        plt.xlabel('Position')
        plt.ylabel('Pearson Correlation')
        plt.title('Position-wise Pearson Correlation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(positions)
        wandb.log({"position_wise_pearson": wandb.Image(plt)})
        plt.close()
        
        # Log position-wise metrics as individual values
        for i in range(n_positions):
            wandb.run.summary[f"train_mse_pos_{i+1}"] = mse_train_individual[i]
            wandb.run.summary[f"test_mse_pos_{i+1}"] = mse_test_individual[i]
            wandb.run.summary[f"train_pearson_pos_{i+1}"] = pearson_train[i]
            wandb.run.summary[f"test_pearson_pos_{i+1}"] = pearson_test[i]
        
    # Return key metrics for sweep optimization
    return {
        "test_improvement_pct": test_improvement,  # Main metric for sweep optimization
        "test_mse": mse_test_overall,
        "train_mse": mse_train_overall
    }


def main():
    parser = argparse.ArgumentParser(description="MLP test experiment for reasoning traces")
    # Core data parameters
    parser.add_argument("--train_split", type=str, default='train', choices=['train', 'test'], help="Split to use for training")
    parser.add_argument("--train_dataset", type=str, default='gsm8k', choices=['gsm8k', 'math500', 'numina', 'amc23', 'aime24', 'GPQADiamond'], help="Dataset for training")
    parser.add_argument("--test_split", type=str, default='test', choices=['train', 'test'], help="Split to use for testing")
    parser.add_argument("--test_dataset", type=str, default='gsm8k', choices=['gsm8k', 'math500', 'numina', 'amc23', 'aime24', 'GPQADiamond'], help="Dataset for testing")
    parser.add_argument("--X-key", type=str, default='hidden_state', choices=['hidden_state'], help="Key to use for input to MLP")
    
    # Model architecture parameters
    parser.add_argument("--hidden-layer", type=str, default="first", help="Which hidden layer to use (first, middle, last, or specific index)")
    parser.add_argument("--hidden-dims", type=str, default="256", help="Comma-separated list of hidden layer dimensions")
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "gelu", "sigmoid"], help="Activation function")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    
    # Training parameters
    parser.add_argument("--num-epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for optimizer")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay (L2 regularization)")
    
    # Logging parameters
    parser.add_argument("--use-wandb", action="store_true", help="Enable logging with Weights & Biases")
    parser.add_argument("--wandb-project", type=str, default="reasoning-scheduling", help="WandB project name")
    
    # Sweep parameters
    parser.add_argument("--sweep-id", type=str, default=None, help="WandB sweep ID to run")
    parser.add_argument("--sweep-count", type=int, default=1, help="Number of sweep runs to execute")
    
    # Deprecated parameters (kept for compatibility)
    parser.add_argument("--train_data_dir", type=str, help="Directory containing training data files (deprecated)")
    parser.add_argument("--test_data_dir", type=str, help="Directory containing test data files (deprecated)")
    
    args = parser.parse_args()
    
    # Check if we're running a sweep
    if args.sweep_id:
        print(f"Running WandB sweep with ID: {args.sweep_id}")
        wandb.agent(args.sweep_id, function=train_mlp_sweep, count=args.sweep_count, project=args.wandb_project)
        return
    
    # Define the root directories for X and Y data
    from constants import X_STEM, Y_STEM
    #X_STEM = "/n/netscratch/gershman_lab/Lab/amuppidi/reasoning_scheduling_new_orig/data/"
    #Y_STEM = "/n/netscratch/gershman_lab/Lab/amuppidi/reasoning_scheduling_new/data/"
    
    # Construct full paths for X and Y data
    train_data_dir_X = os.path.join(X_STEM, f"{args.train_dataset}_results")
    train_data_dir_Y = os.path.join(Y_STEM, f"{args.train_dataset}_results")
    test_data_dir_X = os.path.join(X_STEM, f"{args.test_dataset}_results")
    test_data_dir_Y = os.path.join(Y_STEM, f"{args.test_dataset}_results")
    
    print(f"X data directories:\n  Train: {train_data_dir_X}\n  Test: {test_data_dir_X}")
    print(f"Y data directories:\n  Train: {train_data_dir_Y}\n  Test: {test_data_dir_Y}")
    
    # Parse hidden_dims from string to list of integers
    hidden_dims = [int(dim) for dim in args.hidden_dims.split(',')]
    
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
        hidden_layer=args.hidden_layer,
        hidden_dims=hidden_dims,
        activation=args.activation,
        dropout=args.dropout,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_wandb=args.use_wandb,
        project_name=args.wandb_project
    )


if __name__ == "__main__":
    main()