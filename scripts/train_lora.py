#!/usr/bin/env python3
"""
Train LoRA fine-tuned models for early stopping prediction and difficulty classification.

This script trains LoRA-adapted language models for predictive scheduling tasks,
supporting both early stopping probability prediction and difficulty classification.

Usage:
    python scripts/train_lora.py --task early_stopping --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
    python scripts/train_lora.py --task difficulty --config configs/lora_training.yaml
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
from predictive_scheduling.models import EarlyStopFinetuner, DifficultyClassifier, PredictorType
from predictive_scheduling.training import LoRATrainer, TrainingConfig, create_data_loaders
from predictive_scheduling.training.utils import set_seed

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train LoRA fine-tuned models for predictive scheduling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration file"
    )
    
    # Task parameters
    parser.add_argument(
        "--task", 
        type=str, 
        required=True,
        choices=["early_stopping", "difficulty"],
        help="Task to train model for"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="gsm8k",
        choices=["gsm8k", "math500", "numina", "amc23", "aime24"],
        help="Dataset to train on"
    )
    
    # Model parameters
    parser.add_argument(
        "--model-name", 
        type=str, 
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        help="Base model name"
    )
    parser.add_argument(
        "--use-lora", 
        action="store_true",
        default=True,
        help="Use LoRA adaptation"
    )
    parser.add_argument(
        "--lora-rank", 
        type=int, 
        default=16,
        help="LoRA rank parameter"
    )
    parser.add_argument(
        "--lora-alpha", 
        type=int, 
        default=32,
        help="LoRA alpha parameter"
    )
    parser.add_argument(
        "--lora-dropout", 
        type=float, 
        default=0.05,
        help="LoRA dropout rate"
    )
    
    # Training parameters
    parser.add_argument(
        "--num-epochs", 
        type=int, 
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=32,
        help="Training batch size"
    )
    parser.add_argument(
        "--eval-batch-size", 
        type=int, 
        default=8,
        help="Evaluation batch size"
    )
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--max-length", 
        type=int, 
        default=512,
        help="Maximum sequence length"
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
        default="outputs/lora_training",
        help="Output directory"
    )
    parser.add_argument(
        "--cache-dir", 
        type=str,
        help="Model cache directory (defaults to output_dir/cache)"
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
        default="predictive-scheduling-lora",
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
    cache_dir = args.cache_dir or str(Path(args.output_dir) / "cache")
    
    config = {
        'model_name': args.model_name,
        'use_lora': args.use_lora,
        'lora_r': args.lora_rank,
        'lora_alpha': args.lora_alpha,
        'lora_dropout': args.lora_dropout,
        'target_modules': base_config.lora.target_modules,
        'output_dir': cache_dir,
        'device_map': base_config.model.device,
        'use_flash_attention': base_config.model.use_flash_attention,
    }
    
    if args.task == "early_stopping":
        config['num_positions'] = 16  # 16 early stopping positions
    else:  # difficulty classification
        config['num_classes'] = 3  # easy, medium, hard
        config['class_names'] = ['easy', 'medium', 'hard']
    
    return config


def create_training_config(args):
    """Create training configuration from arguments."""
    return TrainingConfig(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,  # Default for LoRA
        gradient_clip_norm=1.0,
        early_stopping_patience=3,
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
    model_short = args.model_name.split('/')[-1]  # Get model name without org
    return f"{args.task}_{args.dataset}_{model_short}_lora_r{args.lora_rank}"


def generate_model_filename(args):
    """Generate filename for saved model."""
    model_short = args.model_name.split('/')[-1].replace('-', '_')
    return f"lora_{args.task}_{args.dataset}_{model_short}_r{args.lora_rank}_a{args.lora_alpha}.pt"


def main():
    """Main training function."""
    args = parse_args()
    setup_logging(args.verbose)
    
    logger.info(f"Starting LoRA training for {args.task}")
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
        if args.task == "early_stopping":
            model = EarlyStopFinetuner(PredictorType.EARLY_STOPPING_LORA, model_config)
            logger.info("Created LoRA early stopping model")
        else:  # difficulty classification
            model = DifficultyClassifier(PredictorType.DIFFICULTY_CLASSIFICATION, model_config)
            logger.info("Created LoRA difficulty classifier")
        
        logger.info(f"Model has {model.get_num_parameters():,} total parameters")
        if hasattr(model, 'get_trainable_parameters'):
            logger.info(f"Trainable parameters: {model.get_trainable_parameters():,}")
        
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        sys.exit(1)
    
    # Load data
    try:
        logger.info("Loading training data...")
        train_loader, val_loader, test_loader = create_data_loaders(
            dataset=args.dataset,
            task=args.task,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            max_length=args.max_length,
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
        trainer = LoRATrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=training_config
        )
        logger.info("Created LoRA trainer")
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
    
    # Save model
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
    
    logger.info(f"LoRA {args.task} training completed successfully!")


if __name__ == "__main__":
    main()