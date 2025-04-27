import argparse
import numpy as np
import pandas as pd
import os
import wandb

import torch
import torch.nn as nn
import torch.optim as optim

from scipy.stats import pearsonr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from constants import X_STEM, Y_STEM, MODELS_STEM
import ast

class ModifiedTransformerModel(nn.Module):
    def __init__(self, base_model, output_dim=16, target_layer=16):
        super().__init__()
        self.base_model = base_model
        self.target_layer = target_layer
        hidden_size = base_model.config.hidden_size
        
        # Add an MLP head instead of just a linear layer
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, output_dim)
        )
    
    def forward(self, hidden_states):
        outputs = self.base_model(inputs_embeds=hidden_states, output_hidden_states=True)
        target_hidden = outputs.hidden_states[self.target_layer]
        target_token = target_hidden[:, -1, :]
        return self.regression_head(target_token)

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
        
        if not (os.path.exists(x_file) and os.path.exists(y_file)):
            if batch_idx == 0:
                print(f"No data files found at batch 0. X path: {x_file}, Y path: {y_file}")
            break
            
        print(f"Loading X data from {x_file}")
        x_data = pd.read_csv(x_file)
        print(f"Loading Y data from {y_file}")
        y_data = pd.read_csv(y_file)
        
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
        raise ValueError(f"No matching data files found for {dataset} {split}")
        
    final_data = pd.concat(all_merged_data, ignore_index=True)
    
    print(f"Saving grouped data to {grouped_file}")
    final_data.to_csv(grouped_file, index=False)
    
    return final_data

def train_model(
    model_path,
    train_data_dir_X='', train_data_dir_Y='', train_split='train', train_dataset='gsm8k',
    test_data_dir_X='', test_data_dir_Y='', test_split='test', test_dataset='gsm8k',
    num_epochs=20, batch_size=4, learning_rate=1e-3, weight_decay=0, hidden_layer=None,
    use_wandb=True, project_name="reasoning-scheduling"):
    """
    Train a modified transformer model with LoRA to predict early stopping correctness.
    """
    # Validate input directories
    for dir_path in [train_data_dir_X, train_data_dir_Y, test_data_dir_X, test_data_dir_Y]:
        if not os.path.exists(dir_path):
            raise ValueError(f"Data directory {dir_path} not found.")
    
    # Initialize wandb if requested
    if use_wandb:
        run_name = f"lora_{train_dataset}_{train_split}_to_{test_dataset}_{test_split}"
        if hidden_layer:
            run_name += f"_layer_{hidden_layer}"
        
        wandb.init(
            project=project_name,
            name=run_name,
            config={
                "model_path": model_path,
                "train_dataset": train_dataset,
                "train_split": train_split,
                "test_dataset": test_dataset,
                "test_split": test_split,
                "hidden_layer": hidden_layer,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay
            }
        )
    
    # Load base model
    print(f"Loading base model from {model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Create modified model
    model = ModifiedTransformerModel(base_model, output_dim=16, target_layer=16)
    
    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        # Target attention layers up to layer 16
        target_modules=[f"layers.{i}.self_attn.q_proj" for i in range(17)] + 
                       [f"layers.{i}.self_attn.v_proj" for i in range(17)],
        # Potentially higher learning rate for earlier layers
        lora_learning_rate=1e-3
    )
    
    # Create PEFT model
    print("Applying LoRA adaptation")
    model = get_peft_model(model, peft_config)
    model = model.to('cuda')
    
    # Load and process data
    print("Loading training data...")
    train_data = load_and_merge_data(train_data_dir_X, train_data_dir_Y, train_dataset, train_split, hidden_layer)
    print("Loading test data...")
    test_data = load_and_merge_data(test_data_dir_X, test_data_dir_Y, test_dataset, test_split, hidden_layer)
    
    # Parse lists from string representation
    def parse_list(x):
        try:
            return ast.literal_eval(x)
        except Exception:
            return x

    # Process data
    train_data['hidden_state'] = train_data['hidden_state'].apply(parse_list)
    train_data['early_stop_correct_proportions'] = train_data['early_stop_correct_proportions'].apply(parse_list)
    test_data['hidden_state'] = test_data['hidden_state'].apply(parse_list)
    test_data['early_stop_correct_proportions'] = test_data['early_stop_correct_proportions'].apply(parse_list)

    # Prepare data tensors
    X_train = torch.tensor(np.vstack(train_data['hidden_state'].values), dtype=torch.float32).to('cuda')
    Y_train = torch.tensor(np.vstack(train_data['early_stop_correct_proportions'].values), dtype=torch.float32).to('cuda')
    X_test = torch.tensor(np.vstack(test_data['hidden_state'].values), dtype=torch.float32).to('cuda')
    Y_test = torch.tensor(np.vstack(test_data['early_stop_correct_proportions'].values), dtype=torch.float32).to('cuda')
    
    print(f"Training data shape: {X_train.shape}, {Y_train.shape}")
    print(f"Testing data shape: {X_test.shape}, {Y_test.shape}")
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_X.size(0)
            
        epoch_loss = running_loss / len(train_dataset)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            val_loss = criterion(val_outputs, Y_test).item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": epoch_loss,
                "val_loss": val_loss
            })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            output_dir = os.path.join(MODELS_STEM, f"lora_{train_dataset}_{train_split}")
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            print(f"Saved best model to {output_dir}")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        train_pred = model(X_train).cpu().numpy()
        test_pred = model(X_test).cpu().numpy()
    
    # Calculate metrics
    train_mse = np.mean((train_pred - Y_train.cpu().numpy())**2)
    test_mse = np.mean((test_pred - Y_test.cpu().numpy())**2)
    
    print("\nFinal Results:")
    print(f"Train MSE: {train_mse:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    
    # Calculate mean predictions from both training and test sets for each position
    train_means = np.mean(Y_train.cpu().numpy(), axis=0)
    test_means = np.mean(Y_test.cpu().numpy(), axis=0)
    
    # Create baseline predictions using train means
    baseline_train_pred = np.tile(train_means, (Y_train.shape[0], 1))
    baseline_test_pred = np.tile(train_means, (Y_test.shape[0], 1))
    
    # Calculate MSE for model and baseline
    mse_train_overall = np.mean((train_pred - Y_train.cpu().numpy())**2)
    mse_test_overall = np.mean((test_pred - Y_test.cpu().numpy())**2)
    baseline_mse_train = np.mean((baseline_train_pred - Y_train.cpu().numpy())**2)
    baseline_mse_test = np.mean((baseline_test_pred - Y_test.cpu().numpy())**2)
    
    # Calculate relative improvement over baseline
    train_improvement = ((baseline_mse_train - mse_train_overall) / baseline_mse_train) * 100
    test_improvement = ((baseline_mse_test - mse_test_overall) / baseline_mse_test) * 100
    
    print("\nModel Performance:")
    print(f"Overall MSE on train: {mse_train_overall:.4f}")
    print(f"Overall MSE on test: {mse_test_overall:.4f}")
    
    print("\nBaseline Performance (Predicting Train Means):")
    print(f"Overall MSE on train: {baseline_mse_train:.4f}")
    print(f"Overall MSE on test: {baseline_mse_test:.4f}")
    
    print("\nRelative Improvement Over Baseline:")
    print(f"Train improvement: {train_improvement:.1f}%")
    print(f"Test improvement: {test_improvement:.1f}%")
    
    # Position-wise analysis
    mse_train_individual = np.mean((train_pred - Y_train.cpu().numpy())**2, axis=0)
    mse_test_individual = np.mean((test_pred - Y_test.cpu().numpy())**2, axis=0)
    
    pearson_train = [pearsonr(train_pred[:, i], Y_train.cpu().numpy()[:, i])[0] for i in range(Y_train.shape[1])]
    pearson_test = [pearsonr(test_pred[:, i], Y_test.cpu().numpy()[:, i])[0] for i in range(Y_test.shape[1])]
    
    print("\nEarly stopping position-wise MSE and Pearson correlation (Train):")
    for i, (mse_val, p_val) in enumerate(zip(mse_train_individual, pearson_train)):
        print(f"Position {i+1}: MSE: {mse_val:.4f}, Pearson: {p_val:.4f}")
    
    print("\nEarly stopping position-wise MSE and Pearson correlation (Test):")
    for i, (mse_val, p_val) in enumerate(zip(mse_test_individual, pearson_test)):
        print(f"Position {i+1}: MSE: {mse_val:.4f}, Pearson: {p_val:.4f}")
    
    # Mean Values comparison
    print("\nMean Values:")
    print("Position  Train_Mean  Test_Mean   Diff")
    print("-" * 45)
    for i, (train_mean, test_mean) in enumerate(zip(train_means, test_means)):
        diff = abs(train_mean - test_mean)
        print(f"{i+1:8d}  {train_mean:.4f}     {test_mean:.4f}     {diff:.4f}")
    
    # Update wandb metrics
    if use_wandb:
        wandb.run.summary.update({
            "final_train_mse": mse_train_overall,
            "final_test_mse": mse_test_overall,
            "baseline_train_mse": baseline_mse_train,
            "baseline_test_mse": baseline_mse_test,
            "train_improvement_pct": train_improvement,
            "test_improvement_pct": test_improvement,
            "best_val_loss": best_val_loss
        })
        
        # Log position-wise metrics
        for i in range(Y_train.shape[1]):
            wandb.run.summary[f"train_mse_pos_{i+1}"] = mse_train_individual[i]
            wandb.run.summary[f"test_mse_pos_{i+1}"] = mse_test_individual[i]
            wandb.run.summary[f"train_pearson_pos_{i+1}"] = pearson_train[i]
            wandb.run.summary[f"test_pearson_pos_{i+1}"] = pearson_test[i]
    
    return {
        "train_mse": train_mse,
        "test_mse": test_mse,
        "best_val_loss": best_val_loss
    }

def main():
    parser = argparse.ArgumentParser(description="Train LoRA-adapted transformer for early stopping prediction")
    parser.add_argument("--model_path", type=str, required=True, help="HuggingFace model path")
    parser.add_argument("--train_split", type=str, default='train', choices=['train', 'test'])
    parser.add_argument("--train_dataset", type=str, default='gsm8k')
    parser.add_argument("--test_split", type=str, default='test', choices=['train', 'test'])
    parser.add_argument("--test_dataset", type=str, default='gsm8k')
    parser.add_argument("--hidden_layer", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="reasoning-scheduling")
    
    args = parser.parse_args()
    
    # Construct data directory paths
    train_data_dir_X = os.path.join(X_STEM, f"{args.train_dataset}_results")
    train_data_dir_Y = os.path.join(Y_STEM, f"{args.train_dataset}_results")
    test_data_dir_X = os.path.join(X_STEM, f"{args.test_dataset}_results")
    test_data_dir_Y = os.path.join(Y_STEM, f"{args.test_dataset}_results")
    
    train_model(
        model_path=args.model_path,
        train_data_dir_X=train_data_dir_X,
        train_data_dir_Y=train_data_dir_Y,
        train_split=args.train_split,
        train_dataset=args.train_dataset,
        test_data_dir_X=test_data_dir_X,
        test_data_dir_Y=test_data_dir_Y,
        test_split=args.test_split,
        test_dataset=args.test_dataset,
        hidden_layer=args.hidden_layer,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_wandb=args.use_wandb,
        project_name=args.wandb_project
    )

if __name__ == "__main__":
    main()
