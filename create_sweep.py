import argparse
import wandb
import yaml
import os

def create_sweep_configuration(
    train_dataset='gsm8k',
    train_split='train',
    test_dataset='gsm8k',
    test_split='test',
    sweep_name=None,
    sweep_count=1
):
    """
    Create a WandB sweep configuration for the MLP training script.
    
    Args:
        train_dataset: Dataset to use for training
        train_split: Split to use for training ('train' or 'test')
        test_dataset: Dataset to use for testing
        test_split: Split to use for testing ('train' or 'test')
        sweep_name: Optional name for the sweep
        sweep_count: Number of runs to perform in the sweep
        
    Returns:
        sweep_id: ID of the created sweep
    """
    # Define the sweep configuration
    sweep_config = {
        'method': 'bayes',  # Bayesian optimization method
        'metric': {
            'name': 'test_improvement_pct',  # Optimize based on test improvement percentage
            'goal': 'maximize'
        },
        'parameters': {
            # Hidden state layer selection
            'hidden_layer': {
                'values': ['first', 'middle', 'last']
            },
            
            # Model architecture parameters
            'hidden_dims': {
                'values': [
                    [128],
                    [256],
                    [512],
                    [128, 128],
                    [256, 256],
                    [128, 256],
                    [256, 128],
                    [128, 256, 128]
                ]
            },
            
            # Activation function
            'activation': {
                'values': ['relu', 'gelu', 'sigmoid']
            },
            
            # Learning rate
            'learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 1e-4,
                'max': 1e-2
            },
            
            # Regularization weight decay
            'weight_decay': {
                'distribution': 'log_uniform_values',
                'min': 1e-6,
                'max': 1e-3
            },
            
            # Dropout rate
            'dropout': {
                'distribution': 'uniform',
                'min': 0,
                'max': 0.5
            },
            
            # Batch size options
            'batch_size': {
                'values': [8, 16, 32]
            },
            
            # Fixed parameters
            'train_dataset': {'value': train_dataset},
            'train_split': {'value': train_split},
            'test_dataset': {'value': test_dataset},
            'test_split': {'value': test_split},
            'num_epochs': {'value': 25},
            'X_key': {'value': 'hidden_state'}
        }
    }
    
    # Initialize WandB
    wandb.login()
    
    # Create the sweep
    project_name = "reasoning-scheduling"

    sweep_id = wandb.sweep(
        sweep_config, 
        project=project_name,
    )


    print(f"Created sweep with ID: {sweep_id}")
    print(f"To run the sweep agent, use: python train_mlp.py --sweep-id {sweep_id}")
    
    # Save the sweep configuration to a file for reference
    os.makedirs('sweeps', exist_ok=True)
    with open(f'sweeps/sweep_config_{sweep_id.split("/")[-1]}.yaml', 'w') as f:
        yaml.dump(sweep_config, f, default_flow_style=False)
    
    return sweep_id

def main():
    parser = argparse.ArgumentParser(description="Create a WandB sweep for MLP training")
    parser.add_argument("--train-dataset", type=str, default='gsm8k', help="Dataset for training")
    parser.add_argument("--train-split", type=str, default='train', choices=['train', 'test'], help="Split for training")
    parser.add_argument("--test-dataset", type=str, default='gsm8k', help="Dataset for testing")
    parser.add_argument("--test-split", type=str, default='test', choices=['train', 'test'], help="Split for testing")
    parser.add_argument("--sweep-name", type=str, default=None, help="Name for the sweep (optional)")
    parser.add_argument("--sweep-count", type=int, default=10, help="Number of runs to perform in the sweep")
    
    args = parser.parse_args()
    
    sweep_id = create_sweep_configuration(
        train_dataset=args.train_dataset,
        train_split=args.train_split,
        test_dataset=args.test_dataset,
        test_split=args.test_split,
        sweep_name=args.sweep_name,
        sweep_count=args.sweep_count
    )
    
    # Print the command to start the sweep agent
    print("\nSample usage:")
    print(f"wandb agent {sweep_id}")
    print(f"python train_mlp.py --sweep-id {sweep_id}")

if __name__ == "__main__":
    main()