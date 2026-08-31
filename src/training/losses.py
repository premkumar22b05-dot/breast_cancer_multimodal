"""Loss functions for breast cancer detection.

Includes BCE, Focal Loss, and weighted variants for class imbalance.
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.
    
    Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Reduces weight of easy examples, focusing training on hard negatives/positives.
    
    Args:
        alpha: Weight for positive class
        gamma: Focusing parameter (higher = more focus on hard examples)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """Initialize Focal Loss.
        
        Args:
            alpha: Weight for positive class (default 0.25)
            gamma: Focusing parameter (default 2.0)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.
        
        Args:
            inputs: Model logits (B,)
            targets: Binary labels (B,)
        
        Returns:
            Scalar loss
        """
        # Get probabilities
        probs = torch.sigmoid(inputs)
        
        # Compute BCE
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Compute focal weight
        p_t = torch.where(targets > 0.5, probs, 1 - probs)
        focal_weight = (1 - p_t).pow(self.gamma)
        
        # Compute alpha-weighted focal loss
        alpha_t = torch.where(targets > 0.5, self.alpha, 1 - self.alpha)
        focal_loss = alpha_t * focal_weight * bce
        
        return focal_loss.mean()


class WeightedBCELoss(nn.Module):
    """Weighted Binary Cross-Entropy for class imbalance.
    
    Computes BCE with class weights based on dataset statistics.
    
    Args:
        pos_weight: Weight for positive class
    """

    def __init__(self, pos_weight: float = 1.0):
        """Initialize Weighted BCE Loss.
        
        Args:
            pos_weight: Weight for positive examples
        """
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted BCE loss.
        
        Args:
            inputs: Model logits (B,)
            targets: Binary labels (B,)
        
        Returns:
            Scalar loss
        """
        return F.binary_cross_entropy_with_logits(
            inputs,
            targets,
            pos_weight=torch.tensor(self.pos_weight, device=inputs.device),
            reduction='mean',
        )


def get_loss_function(config: Dict) -> nn.Module:
    """Get loss function based on configuration.
    
    Args:
        config: Configuration dictionary with loss settings
        
    Returns:
        Loss function module
    """
    loss_name = config['training']['loss']
    
    if loss_name == 'bce':
        logger.info("Using Binary Cross-Entropy loss")
        return nn.BCEWithLogitsLoss()
    
    elif loss_name == 'weighted_bce':
        pos_weight = config['training'].get('pos_weight', 1.0)
        logger.info(f"Using Weighted BCE loss (pos_weight={pos_weight})")
        return WeightedBCELoss(pos_weight=pos_weight)
    
    elif loss_name == 'focal':
        alpha = config['training'].get('focal_alpha', 0.25)
        gamma = config['training'].get('focal_gamma', 2.0)
        logger.info(f"Using Focal loss (alpha={alpha}, gamma={gamma})")
        return FocalLoss(alpha=alpha, gamma=gamma)
    
    else:
        logger.warning(f"Unknown loss: {loss_name}. Using BCE.")
        return nn.BCEWithLogitsLoss()
