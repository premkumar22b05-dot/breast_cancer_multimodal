"""Early stopping to prevent overfitting.

Monitors validation loss and stops training if no improvement.
"""

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping callback.
    
    Stops training if validation loss doesn't improve for patience epochs.
    Saves the best model.
    
    Args:
        patience: Number of epochs without improvement before stopping
        verbose: Whether to log messages
        delta: Minimum change to qualify as improvement
    """

    def __init__(self, patience: int = 5, verbose: bool = True, delta: float = 0.0):
        """Initialize Early Stopping.
        
        Args:
            patience: Epochs without improvement before stopping
            verbose: Whether to log
            delta: Minimum change to count as improvement
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_path = None

    def __call__(self, val_loss: float, model: torch.nn.Module) -> None:
        """Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: Model to save if best
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"EarlyStopping: {self.counter}/{self.patience}. "
                    f"Best loss: {self.best_loss:.4f}"
                )
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info("EarlyStopping: Training stopped")

    def save_checkpoint(self, model: torch.nn.Module) -> None:
        """Save model checkpoint.
        
        Args:
            model: Model to save
        """
        checkpoint_dir = Path('./results/checkpoints')
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_model_path = checkpoint_dir / 'best_model.pt'
        torch.save(model.state_dict(), self.best_model_path)
        
        if self.verbose:
            logger.debug(f"Saved best model to {self.best_model_path}")
