#!/usr/bin/env python3
"""
Basic usage example for Predictive Scheduling framework.

This example demonstrates how to:
1. Load and configure the framework
2. Train MLP predictors on hidden states  
3. Perform greedy token allocation
4. Evaluate allocation performance

Run with: python examples/basic_usage.py
"""

import numpy as np
import torch
from pathlib import Path

# Import predictive scheduling components
from predictive_scheduling import Config, load_config
from predictive_scheduling.models import MLPPredictor, PredictorType
from predictive_scheduling.allocation import GreedyAllocator, UniformAllocator
from predictive_scheduling.training import MLPTrainer, TrainingConfig

def generate_synthetic_data(num_samples: int = 1000, input_dim: int = 1536, output_dim: int = 16):
    """Generate synthetic data for demonstration."""
    print("Generating synthetic data...")
    
    # Generate synthetic hidden states (normally distributed)
    X = np.random.randn(num_samples, input_dim).astype(np.float32)
    
    # Generate synthetic early stopping probabilities
    # Create realistic patterns: easier problems have higher probabilities at lower budgets
    difficulty = np.random.rand(num_samples)  # Random difficulty scores
    Y = np.zeros((num_samples, output_dim), dtype=np.float32)
    
    for i in range(num_samples):
        # Easier problems (lower difficulty) have higher success rates
        base_prob = 1.0 - difficulty[i]
        for j in range(output_dim):
            # Probability increases with token budget but saturates
            budget_factor = (j + 1) / output_dim
            Y[i, j] = base_prob * (1 - np.exp(-3 * budget_factor)) + np.random.normal(0, 0.1)
            Y[i, j] = np.clip(Y[i, j], 0, 1)  # Ensure probabilities are in [0, 1]
    
    return X, Y

def train_mlp_predictor():
    """Train an MLP predictor on synthetic data."""
    print("\n=== Training MLP Predictor ===")
    
    # Generate data
    X_train, Y_train = generate_synthetic_data(800, 1536, 16)
    X_val, Y_val = generate_synthetic_data(200, 1536, 16)
    
    # Create model configuration
    mlp_config = {
        'input_dim': 1536,
        'hidden_dims': [256],
        'output_dim': 16,
        'activation': 'relu',
        'dropout': 0.1
    }
    
    # Create MLP predictor
    model = MLPPredictor(PredictorType.EARLY_STOPPING_MLP, mlp_config)
    print(f"Created MLP with {model.get_num_parameters():,} parameters")
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), 
        torch.from_numpy(Y_train)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_val), 
        torch.from_numpy(Y_val)
    )
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)
    
    # Training configuration
    training_config = TrainingConfig(
        num_epochs=5,  # Short training for demo
        batch_size=32,
        learning_rate=1e-3,
        early_stopping_patience=3,
        use_wandb=False  # Disable wandb for demo
    )
    
    # Train model
    trainer = MLPTrainer(model, train_loader, val_loader, config=training_config)
    results = trainer.train()
    
    print(f"Training completed!")
    print(f"Final train loss: {results['final_metrics'].get('train_loss', 'N/A'):.4f}")
    print(f"Final val loss: {results['final_metrics'].get('val_loss', 'N/A'):.4f}")
    print(f"Final val correlation: {results['final_metrics'].get('val_pearson', 'N/A'):.4f}")
    
    return model

def demonstrate_allocation(predictor):
    """Demonstrate token allocation algorithms."""
    print("\n=== Token Allocation Demonstration ===")
    
    # Generate test queries
    num_queries = 10
    X_test, _ = generate_synthetic_data(num_queries, 1536, 16)
    
    # Get predictions from MLP
    predictions = predictor.predict(X_test)
    print(f"Generated predictions for {num_queries} queries")
    print(f"Predictions shape: {predictions.shape}")
    
    # Define budget parameters
    total_budget = 1600  # Total tokens available
    window_size = 16     # Allocation window size
    
    print(f"\nBudget allocation:")
    print(f"Total budget: {total_budget} tokens")
    print(f"Window size: {window_size} tokens")
    print(f"Average budget per query: {total_budget / num_queries:.1f} tokens")
    
    # 1. Uniform allocation (baseline)
    print("\n--- Uniform Allocation (Baseline) ---")
    uniform_allocator = UniformAllocator(window_size=window_size)
    uniform_result = uniform_allocator.allocate(predictions, total_budget)
    
    print(f"Uniform allocations: {uniform_result.allocations}")
    print(f"Allocation variance: {uniform_result.allocation_variance:.2f}")
    
    # 2. Greedy allocation (predictive scheduling)
    print("\n--- Greedy Allocation (Predictive Scheduling) ---")
    greedy_allocator = GreedyAllocator(window_size=window_size, min_allocation=16)
    greedy_result = greedy_allocator.allocate(predictions, total_budget)
    
    print(f"Greedy allocations: {greedy_result.allocations}")
    print(f"Allocation variance: {greedy_result.allocation_variance:.2f}")
    
    # Compare allocations
    print("\n--- Allocation Comparison ---")
    allocation_diff = np.array(greedy_result.allocations) - np.array(uniform_result.allocations)
    print(f"Allocation differences: {allocation_diff}")
    print(f"Queries with more tokens: {np.sum(allocation_diff > 0)}")
    print(f"Queries with fewer tokens: {np.sum(allocation_diff < 0)}")
    print(f"Max increase: {np.max(allocation_diff)} tokens")
    print(f"Max decrease: {np.min(allocation_diff)} tokens")
    
    return uniform_result, greedy_result

def demonstrate_configuration():
    """Demonstrate configuration management."""
    print("\n=== Configuration Management ===")
    
    # Load default configuration
    try:
        config = load_config()
        print("✓ Loaded default configuration")
        print(f"Model name: {config.model.model_name}")
        print(f"Hidden size: {config.model.hidden_size}")
        print(f"Training batch size: {config.training.batch_size}")
        print(f"Allocation window size: {config.allocation.window_size}")
    except Exception as e:
        print(f"⚠ Could not load configuration: {e}")
        print("Using manual configuration...")
        
        # Create manual configuration
        config = Config()
        print("✓ Created default configuration")
    
    # Show configuration validation
    try:
        config.validate()
        print("✓ Configuration validation passed")
    except Exception as e:
        print(f"⚠ Configuration validation failed: {e}")
    
    return config

def main():
    """Main demonstration function."""
    print("🚀 Predictive Scheduling Framework - Basic Usage Example")
    print("=" * 60)
    
    # 1. Configuration demonstration
    config = demonstrate_configuration()
    
    # 2. Train MLP predictor
    try:
        predictor = train_mlp_predictor()
    except Exception as e:
        print(f"⚠ Training failed: {e}")
        print("Continuing with dummy predictor...")
        
        # Create dummy predictor for demonstration
        mlp_config = {'input_dim': 1536, 'hidden_dims': [256], 'output_dim': 16}
        predictor = MLPPredictor(PredictorType.EARLY_STOPPING_MLP, mlp_config)
    
    # 3. Allocation demonstration
    try:
        uniform_result, greedy_result = demonstrate_allocation(predictor)
        
        print("\n=== Summary ===")
        print(f"✓ Successfully demonstrated MLP training")
        print(f"✓ Successfully demonstrated allocation algorithms")
        print(f"✓ Greedy allocation shows {greedy_result.allocation_variance:.2f} variance")
        print(f"  vs uniform allocation with {uniform_result.allocation_variance:.2f} variance")
        
        if greedy_result.allocation_variance > uniform_result.allocation_variance:
            print("✓ Greedy allocation is more adaptive (higher variance indicates differentiation)")
        
    except Exception as e:
        print(f"⚠ Allocation demonstration failed: {e}")
    
    print("\n🎉 Example completed successfully!")
    print("\nNext steps:")
    print("1. Try training on real data with scripts/train_mlp.py")
    print("2. Explore LoRA fine-tuning with scripts/train_difficulty.py")
    print("3. Run evaluation scripts to reproduce paper results")

if __name__ == "__main__":
    main()