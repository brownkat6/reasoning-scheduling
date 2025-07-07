"""
Training utilities for predictive scheduling models.

This module provides unified training interfaces for both MLP and LoRA-based models,
with proper experiment tracking, checkpointing, and evaluation.
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union, List
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report

from ..config import Config
from ..models.base import BasePredictor
from .metrics import EvaluationMetrics, compute_metrics
from .utils import set_seed, save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    num_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 5
    save_every_n_epochs: int = 1
    eval_every_n_epochs: int = 1
    use_wandb: bool = False
    wandb_project: str = "predictive-scheduling"
    wandb_run_name: Optional[str] = None
    output_dir: str = "./outputs"
    seed: int = 42


class BaseTrainer:
    """
    Base trainer class providing common training functionality.
    
    This class implements the core training loop, evaluation, checkpointing,
    and logging functionality that is shared across different model types.
    """
    
    def __init__(self, 
                 model: BasePredictor,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 test_loader: Optional[DataLoader] = None,
                 config: Optional[TrainingConfig] = None):
        """
        Initialize base trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            test_loader: Test data loader (optional)
            config: Training configuration
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config or TrainingConfig()
        
        # Set random seed
        set_seed(self.config.seed)
        
        # Setup output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize training state
        self.current_epoch = 0
        self.best_val_metric = float('inf')  # Assuming lower is better (e.g., loss)
        self.patience_counter = 0
        self.training_history = []
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Setup optimizer and loss function
        self._setup_training()
        
        # Setup logging
        self._setup_logging()
        
        logger.info(f"Initialized trainer with {self.model.get_num_parameters()} parameters")
        logger.info(f"Training on device: {self.device}")
    
    def _setup_training(self):
        """Setup optimizer and loss function."""
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Default loss function (can be overridden in subclasses)
        self.criterion = nn.MSELoss()
    
    def _setup_logging(self):
        """Setup experiment tracking."""
        if self.config.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=self.config.wandb_project,
                    name=self.config.wandb_run_name,
                    config=self.config.__dict__
                )
                self.use_wandb = True
                logger.info("Initialized Weights & Biases logging")
            except ImportError:
                logger.warning("wandb not available, skipping W&B logging")
                self.use_wandb = False
        else:
            self.use_wandb = False
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = self._batch_to_device(batch)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self._forward_pass(batch)
            loss = self._compute_loss(outputs, batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.gradient_clip_norm
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Log batch metrics occasionally
            if batch_idx % 100 == 0:
                logger.debug(f"Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}
    
    def evaluate(self, data_loader: DataLoader, split_name: str = "val") -> Dict[str, float]:
        """Evaluate model on given data loader."""
        self.model.eval()
        total_loss = 0.0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = self._batch_to_device(batch)
                outputs = self._forward_pass(batch)
                loss = self._compute_loss(outputs, batch)
                
                total_loss += loss.item()
                
                # Collect predictions and targets for metrics
                pred, target = self._extract_predictions_and_targets(outputs, batch)
                predictions.append(pred)
                targets.append(target)
        
        # Concatenate all predictions and targets
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        
        # Compute metrics
        metrics = self._compute_metrics(predictions, targets, split_name)
        metrics[f'{split_name}_loss'] = total_loss / len(data_loader)
        
        return metrics
    
    def train(self) -> Dict[str, Any]:
        """
        Run full training loop.
        
        Returns:
            Training history and final metrics
        """
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Training step
            train_metrics = self.train_epoch()
            
            # Validation step
            val_metrics = {}
            if self.val_loader is not None and epoch % self.config.eval_every_n_epochs == 0:
                val_metrics = self.evaluate(self.val_loader, "val")
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            epoch_metrics['epoch'] = epoch
            epoch_metrics['epoch_time'] = time.time() - epoch_start_time
            
            self.training_history.append(epoch_metrics)
            
            # Logging
            self._log_metrics(epoch_metrics)
            
            # Early stopping check
            if self._should_early_stop(val_metrics):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
            
            # Save checkpoint
            if epoch % self.config.save_every_n_epochs == 0:
                self._save_checkpoint(epoch, epoch_metrics)
        
        # Final evaluation
        final_metrics = self._final_evaluation()
        
        # Save final model
        self._save_final_model(final_metrics)
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        
        return {
            'training_history': self.training_history,
            'final_metrics': final_metrics,
            'total_time': total_time
        }
    
    def _batch_to_device(self, batch):
        """Move batch to training device."""
        if isinstance(batch, dict):
            return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return [item.to(self.device) if torch.is_tensor(item) else item for item in batch]
        else:
            return batch.to(self.device) if torch.is_tensor(batch) else batch
    
    def _forward_pass(self, batch):
        """Forward pass (to be implemented by subclasses)."""
        raise NotImplementedError
    
    def _compute_loss(self, outputs, batch):
        """Compute loss (to be implemented by subclasses)."""
        raise NotImplementedError
    
    def _extract_predictions_and_targets(self, outputs, batch):
        """Extract predictions and targets for metrics computation."""
        raise NotImplementedError
    
    def _compute_metrics(self, predictions: np.ndarray, targets: np.ndarray, split_name: str) -> Dict[str, float]:
        """Compute evaluation metrics (to be implemented by subclasses)."""
        return {}
    
    def _should_early_stop(self, val_metrics: Dict[str, float]) -> bool:
        """Check if early stopping should be triggered."""
        if not val_metrics or 'val_loss' not in val_metrics:
            return False
        
        current_metric = val_metrics['val_loss']
        
        if current_metric < self.best_val_metric:
            self.best_val_metric = current_metric
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.config.early_stopping_patience
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to console and wandb."""
        # Console logging
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float))])
        logger.info(f"Epoch {metrics['epoch']}: {metric_str}")
        
        # Wandb logging
        if self.use_wandb:
            import wandb
            wandb.log(metrics)
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save training checkpoint."""
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            metrics=metrics,
            path=checkpoint_path
        )
    
    def _final_evaluation(self) -> Dict[str, Any]:
        """Run final evaluation on all available datasets."""
        final_metrics = {}
        
        # Evaluate on all available datasets
        for loader_name, loader in [("train", self.train_loader), ("val", self.val_loader), ("test", self.test_loader)]:
            if loader is not None:
                metrics = self.evaluate(loader, loader_name)
                final_metrics.update(metrics)
        
        return final_metrics
    
    def _save_final_model(self, final_metrics: Dict[str, Any]):
        """Save final trained model."""
        model_path = self.output_dir / "final_model.pt"
        self.model.set_training_metrics(final_metrics)
        self.model.save_model(model_path)
        logger.info(f"Saved final model to {model_path}")


class MLPTrainer(BaseTrainer):
    """Trainer for MLP-based early stopping predictors."""
    
    def _forward_pass(self, batch):
        """Forward pass for MLP models."""
        if isinstance(batch, dict):
            x = batch['features']
        else:
            x, _ = batch  # Assume (features, targets) tuple
        return self.model(x)
    
    def _compute_loss(self, outputs, batch):
        """Compute MSE loss for MLP models."""
        if isinstance(batch, dict):
            targets = batch['targets']
        else:
            _, targets = batch
        return self.criterion(outputs, targets)
    
    def _extract_predictions_and_targets(self, outputs, batch):
        """Extract predictions and targets for MLP models."""
        predictions = outputs.cpu().numpy()
        if isinstance(batch, dict):
            targets = batch['targets'].cpu().numpy()
        else:
            _, targets = batch
            targets = targets.cpu().numpy()
        return predictions, targets
    
    def _compute_metrics(self, predictions: np.ndarray, targets: np.ndarray, split_name: str) -> Dict[str, float]:
        """Compute regression metrics for MLP models."""
        # Flatten for overall metrics
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()
        
        # Overall metrics
        mse = mean_squared_error(target_flat, pred_flat)
        pearson_r, _ = pearsonr(pred_flat, target_flat)
        
        metrics = {
            f'{split_name}_mse': mse,
            f'{split_name}_pearson': pearson_r
        }
        
        # Position-wise metrics if multi-dimensional output
        if predictions.ndim > 1 and predictions.shape[1] > 1:
            for i in range(predictions.shape[1]):
                pos_mse = mean_squared_error(targets[:, i], predictions[:, i])
                pos_pearson, _ = pearsonr(targets[:, i], predictions[:, i])
                metrics[f'{split_name}_mse_pos_{i+1}'] = pos_mse
                metrics[f'{split_name}_pearson_pos_{i+1}'] = pos_pearson
        
        return metrics


class LoRATrainer(BaseTrainer):
    """Trainer for LoRA-based models."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Use cross-entropy loss for classification tasks
        if hasattr(self.model, 'num_classes'):
            self.criterion = nn.CrossEntropyLoss()
    
    def _forward_pass(self, batch):
        """Forward pass for LoRA models."""
        return self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch.get('labels')
        )
    
    def _compute_loss(self, outputs, batch):
        """Compute loss for LoRA models."""
        return outputs.loss
    
    def _extract_predictions_and_targets(self, outputs, batch):
        """Extract predictions and targets for LoRA models."""
        logits = outputs.logits.cpu().numpy()
        
        if hasattr(self.model, 'num_classes'):
            # Classification task
            predictions = np.argmax(logits, axis=-1)
            targets = batch['labels'].cpu().numpy()
        else:
            # Regression task (early stopping)
            predictions = logits
            targets = batch['labels'].cpu().numpy()
        
        return predictions, targets
    
    def _compute_metrics(self, predictions: np.ndarray, targets: np.ndarray, split_name: str) -> Dict[str, float]:
        """Compute metrics for LoRA models."""
        if hasattr(self.model, 'num_classes'):
            # Classification metrics
            accuracy = accuracy_score(targets, predictions)
            return {f'{split_name}_accuracy': accuracy}
        else:
            # Regression metrics (same as MLP)
            pred_flat = predictions.flatten()
            target_flat = targets.flatten()
            
            mse = mean_squared_error(target_flat, pred_flat)
            pearson_r, _ = pearsonr(pred_flat, target_flat)
            
            return {
                f'{split_name}_mse': mse,
                f'{split_name}_pearson': pearson_r
            }


def create_trainer(model: BasePredictor, 
                   train_loader: DataLoader,
                   val_loader: Optional[DataLoader] = None,
                   test_loader: Optional[DataLoader] = None,
                   config: Optional[TrainingConfig] = None) -> BaseTrainer:
    """
    Create appropriate trainer for the given model type.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        config: Training configuration
        
    Returns:
        Appropriate trainer instance
    """
    if 'lora' in model.predictor_type.value.lower():
        return LoRATrainer(model, train_loader, val_loader, test_loader, config)
    else:
        return MLPTrainer(model, train_loader, val_loader, test_loader, config)