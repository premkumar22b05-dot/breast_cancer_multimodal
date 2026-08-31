"""Training loop for breast cancer detection models.

Handles training, validation, checkpointing, and logging.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from .early_stopping import EarlyStopping
from .losses import get_loss_function

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer class for breast cancer detection models.
    
    Handles:
    - Training loop with gradient updates
    - Validation with metrics computation
    - Checkpointing best models
    - Early stopping
    - Learning rate scheduling
    
    Args:
        model: PyTorch model to train
        config: Configuration dictionary
        device: Device (cpu/cuda)
    """

    def __init__(self, model: nn.Module, config: Dict, device: torch.device):
        """Initialize Trainer.
        
        Args:
            model: Model to train
            config: Configuration dictionary
            device: torch device
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Loss function
        self.criterion = get_loss_function(config)
        
        # Optimizer
        self.optimizer = self._get_optimizer()
        
        # Learning rate scheduler
        self.scheduler = self._get_scheduler()
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config['training']['early_stopping']['patience'],
            verbose=True,
        )
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config['paths']['checkpoints'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"Trainer initialized. Device: {device}, "
            f"Optimizer: {type(self.optimizer).__name__}"
        )

    def _get_optimizer(self) -> Optimizer:
        """Get optimizer based on configuration.
        
        Returns:
            Optimizer instance
        """
        opt_name = self.config['training']['optimizer']
        lr = self.config['training']['learning_rate']
        
        if opt_name == 'adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=self.config['training']['weight_decay'],
            )
        elif opt_name == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=self.config['training']['weight_decay'],
            )
        elif opt_name == 'sgd':
            return torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=self.config['training'].get('momentum', 0.9),
                weight_decay=self.config['training']['weight_decay'],
            )
        else:
            logger.warning(f"Unknown optimizer: {opt_name}. Using Adam.")
            return torch.optim.Adam(self.model.parameters(), lr=lr)

    def _get_scheduler(self) -> Optional[LRScheduler]:
        """Get learning rate scheduler.
        
        Returns:
            Scheduler or None
        """
        scheduler_name = self.config['training'].get('scheduler', None)
        
        if scheduler_name is None:
            return None
        
        if scheduler_name == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs'],
            )
        elif scheduler_name == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config['training']['scheduler_step_size'],
                gamma=self.config['training']['scheduler_gamma'],
            )
        elif scheduler_name == 'exponential':
            return torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=self.config['training']['scheduler_gamma'],
            )
        
        return None

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            images = batch['images'].to(self.device)
            targets = batch['target'].to(self.device)
            clinical = batch['clinical']
            if clinical is not None:
                clinical = clinical.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits, _ = self.model(images, clinical)
            logits = logits.squeeze(-1)  # (B,)
            
            # Compute loss
            loss = self.criterion(logits, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config['training'].get('gradient_clip', None):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip'],
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Logging
            if (batch_idx + 1) % self.config['training'].get('log_interval', 10) == 0:
                avg_loss = total_loss / num_batches
                logger.debug(
                    f"Batch {batch_idx + 1}/{len(train_loader)}: "
                    f"Loss = {loss.item():.4f} (avg: {avg_loss:.4f})"
                )
        
        avg_epoch_loss = total_loss / num_batches
        self.train_losses.append(avg_epoch_loss)
        
        return avg_epoch_loss

    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict]:
        """Validate on validation set.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (avg_loss, metrics_dict)
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                images = batch['images'].to(self.device)
                targets = batch['target'].to(self.device)
                clinical = batch['clinical']
                if clinical is not None:
                    clinical = clinical.to(self.device)
                
                # Forward pass
                logits, _ = self.model(images, clinical)
                logits = logits.squeeze(-1)  # (B,)
                
                # Compute loss
                loss = self.criterion(logits, targets)
                total_loss += loss.item()
                num_batches += 1
                
                # Predictions
                probs = torch.sigmoid(logits)
                all_preds.append(probs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        # Compute metrics
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        
        metrics = self._compute_metrics(all_preds, all_targets)
        self.val_metrics.append(metrics)
        
        return avg_loss, metrics

    def _compute_metrics(self, preds: np.ndarray, targets: np.ndarray) -> Dict:
        """Compute evaluation metrics.
        
        Args:
            preds: Predictions (B,) in [0, 1]
            targets: Ground truth (B,)
            
        Returns:
            Dictionary with metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        )
        
        # Threshold predictions
        threshold = self.config['training'].get('threshold', 0.5)
        pred_labels = (preds >= threshold).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(targets, pred_labels),
            'precision': precision_score(targets, pred_labels, zero_division=0),
            'recall': recall_score(targets, pred_labels, zero_division=0),
            'f1': f1_score(targets, pred_labels, zero_division=0),
            'auc': roc_auc_score(targets, preds),
        }
        
        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict:
        """Train model for specified number of epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Dictionary with training history
        """
        num_epochs = self.config['training']['epochs']
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_metrics = self.validate(val_loader)
            
            # Log
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs}: "
                f"Train Loss = {train_loss:.4f}, "
                f"Val Loss = {val_loss:.4f}, "
                f"AUC = {val_metrics['auc']:.4f}"
            )
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Check early stopping
            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
            
            # Save best model
            if self.early_stopping.best_loss == val_loss:
                self._save_checkpoint(epoch, train_loss, val_loss)
        
        # Load best model
        if self.early_stopping.best_model_path:
            logger.info(f"Loading best model from {self.early_stopping.best_model_path}")
            checkpoint = torch.load(self.early_stopping.best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics,
        }

    def _save_checkpoint(self, epoch: int, train_loss: float, val_loss: float) -> None:
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch
            train_loss: Training loss
            val_loss: Validation loss
        """
        checkpoint_path = self.checkpoint_dir / f"model_epoch_{epoch + 1}.pt"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, checkpoint_path)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
