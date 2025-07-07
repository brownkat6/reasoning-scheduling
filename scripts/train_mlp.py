#!/usr/bin/env python3
"""
Train MLP predictors for early stopping probability estimation.

This script trains lightweight MLP models on transformer hidden states to predict
early stopping probabilities for the predictive scheduling framework.

Usage:
    python scripts/train_mlp.py --dataset gsm8k --hidden-layer 16
    python scripts/train_mlp.py --config configs/mlp_training.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from predictive_scheduling import Config, load_config
from predictive_scheduling.models import MLPPredictor, PredictorType
from predictive_scheduling.training import MLPTrainer, TrainingConfig, create_data_loaders
from predictive_scheduling.training.utils import set_seed

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train MLP predictors for early stopping probability estimation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration file"
    )
    
    # Data parameters
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="gsm8k",
        choices=["gsm8k", "math500", "numina", "amc23", "aime24"],
        help="Dataset to train on"
    )
    parser.add_argument(
        "--train-split", 
        type=str, 
        default="train",
        choices=["train", "test"],
        help="Data split for training"
    )
    parser.add_argument(
        "--test-split", 
        type=str, 
        default="test", 
        choices=["train", "test"],
        help="Data split for testing"
    )
    parser.add_argument(
        "--hidden-layer", 
        type=str, 
        default="16",
        help="Hidden layer to use (number, 'first', 'middle', 'last')"
    )
    
    # Model parameters
    parser.add_argument(
        "--hidden-dims", 
        type=str, 
        default="256",
        help="Comma-separated hidden layer dimensions"
    )
    parser.add_argument(
        "--activation", 
        type=str, 
        default="relu",
        choices=["relu", "gelu", "tanh", "sigmoid"],
        help="Activation function"
    )
    parser.add_argument(
        "--dropout", 
        type=float, 
        default=0.0,
        help="Dropout rate"
    )
    
    # Training parameters
    parser.add_argument(
        "--num-epochs", 
        type=int, 
        default=20,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=32,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight-decay", 
        type=float, 
        default=0.0,
        help="Weight decay (L2 regularization)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed"
    )
    
    # Output parameters
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="outputs/mlp_training",
        help="Output directory"
    )
    parser.add_argument(
        "--save-name", 
        type=str,
        help="Custom name for saved model (auto-generated if not provided)"
    )
    
    # Logging and tracking
    parser.add_argument(
        "--use-wandb", 
        action="store_true",
        help="Use Weights & Biases for experiment tracking"
    )
    parser.add_argument(
        "--wandb-project", 
        type=str, 
        default="predictive-scheduling-mlp",
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )


def create_model_config(args, base_config: Config):
    """Create model configuration from arguments."""
    hidden_dims = [int(dim) for dim in args.hidden_dims.split(',')]
    
    return {
        'input_dim': base_config.model.hidden_size,
        'hidden_dims': hidden_dims,
        'output_dim': 16,  # 16 early stopping positions
        'activation': args.activation,
        'dropout': args.dropout,
        'use_batch_norm': False,
        'bias': True
    }


def create_training_config(args):
    """Create training configuration from arguments."""
    return TrainingConfig(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=1.0,
        early_stopping_patience=5,
        save_every_n_epochs=5,
        eval_every_n_epochs=1,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=create_run_name(args),
        output_dir=args.output_dir,
        seed=args.seed
    )


def create_run_name(args):
    """Create descriptive run name for experiment tracking."""
    layer_str = f"layer_{args.hidden_layer}" if args.hidden_layer.isdigit() else args.hidden_layer
    return f"{args.dataset}_{args.train_split}_to_{args.test_split}_{layer_str}_mlp"


def generate_model_filename(args):
    """Generate filename for saved model."""
    if args.save_name:
        return f"{args.save_name}.pt"
    
    layer_str = f"_layer_{args.hidden_layer}" if args.hidden_layer.isdigit() else f"_{args.hidden_layer}"
    arch_str = f"_arch_{args.hidden_dims}"
    return f"mlp_{args.dataset}_{args.train_split}{layer_str}{arch_str}_act_{args.activation}_drop_{args.dropout:.2f}.pt"


def main():
    """Main training function."""
    args = parse_args()
    setup_logging(args.verbose)
    
    logger.info("Starting MLP training for predictive scheduling")
    logger.info(f"Arguments: {vars(args)}")
    
    # Load configuration
    try:
        if args.config:
            config = load_config(args.config)
            logger.info(f"Loaded configuration from {args.config}")
        else:
            config = load_config()
            logger.info("Using default configuration")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Set random seed
    set_seed(args.seed)
    
    # Create model configuration
    model_config = create_model_config(args, config)
    logger.info(f"Model configuration: {model_config}")
    
    # Create model
    try:
        model = MLPPredictor(PredictorType.EARLY_STOPPING_MLP, model_config)
        logger.info(f"Created MLP with {model.get_num_parameters():,} parameters")
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        sys.exit(1)
    
    # Load data
    try:
        logger.info("Loading training data...")
        train_loader, val_loader, test_loader = create_data_loaders(
            dataset=args.dataset,
            train_split=args.train_split,
            test_split=args.test_split,
            hidden_layer=args.hidden_layer,
            batch_size=args.batch_size,
            config=config
        )
        
        logger.info(f"Training batches: {len(train_loader)}")
        logger.info(f"Validation batches: {len(val_loader) if val_loader else 0}")
        logger.info(f"Test batches: {len(test_loader) if test_loader else 0}")
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)
    
    # Create training configuration
    training_config = create_training_config(args)
    
    # Create trainer
    try:
        trainer = MLPTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=training_config
        )
        logger.info("Created trainer")
    except Exception as e:
        logger.error(f"Failed to create trainer: {e}")
        sys.exit(1)
    
    # Train model
    try:
        logger.info("Starting training...")
        results = trainer.train()
        logger.info("Training completed successfully!")
        
        # Log final results
        final_metrics = results.get('final_metrics', {})
        logger.info("Final Results:")
        for metric_name, value in final_metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {metric_name}: {value:.4f}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)
    
    # Save model with custom name
    try:
        model_filename = generate_model_filename(args)
        model_path = Path(args.output_dir) / model_filename
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        model.set_training_metrics(results.get('final_metrics', {}))
        model.save_model(str(model_path))
        logger.info(f"Model saved to: {model_path}")
        
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        sys.exit(1)
    
    logger.info("MLP training completed successfully!")


if __name__ == "__main__":
    main()