"""Gated Multimodal Fusion module.

Implements learned weighted fusion of visual and clinical modalities.
"""

import logging
from typing import Dict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class GatedFusion(nn.Module):
    """Gated multimodal fusion combining visual and clinical features.
    
    Uses gating mechanism to learn optimal weighting of modalities.
    Supports sigmoid, GELU, and ReLU gates.
    
    Args:
        config: Configuration dictionary
    """

    def __init__(self, config: Dict):
        """Initialize Gated Fusion module.
        
        Args:
            config: Configuration dictionary with model settings
        """
        super().__init__()
        self.config = config
        
        self.d_model = config['model']['d_model']
        self.output_dim = config['model']['gated_fusion']['output_dim']
        self.gate_type = config['model']['gated_fusion']['gate_type']
        self.use_residual = config['model']['gated_fusion']['use_residual']
        
        # Project visual features
        self.visual_projection = nn.Linear(self.d_model, self.output_dim)
        
        # Project clinical features
        self.clinical_projection = nn.Linear(self.d_model, self.output_dim)
        
        # Gate networks (one for each modality)
        self.visual_gate = nn.Sequential(
            nn.Linear(self.output_dim * 2, self.output_dim),
            self._get_activation(self.gate_type),
            nn.Linear(self.output_dim, self.output_dim),
            nn.Sigmoid(),  # Gate output is always in [0, 1]
        )
        
        self.clinical_gate = nn.Sequential(
            nn.Linear(self.output_dim * 2, self.output_dim),
            self._get_activation(self.gate_type),
            nn.Linear(self.output_dim, self.output_dim),
            nn.Sigmoid(),  # Gate output is always in [0, 1]
        )
        
        # Fusion output projection
        self.fusion_projection = nn.Sequential(
            nn.Linear(self.output_dim * 2, self.output_dim),
            nn.LayerNorm(self.output_dim),
        )
        
        logger.info(
            f"GatedFusion: {self.gate_type} gates, "
            f"residual={self.use_residual}, output_dim={self.output_dim}"
        )

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function.
        
        Args:
            activation: Activation name (sigmoid, gelu, relu)
            
        Returns:
            Activation module
        """
        if activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'gelu':
            return nn.GELU()
        elif activation == 'relu':
            return nn.ReLU()
        else:
            return nn.ReLU()  # Default

    def forward(self, visual_features: torch.Tensor, clinical_features: torch.Tensor) -> torch.Tensor:
        """Fuse visual and clinical features using gating.
        
        Args:
            visual_features: Visual feature tensor (B, d_model) or (B, 1, d_model)
            clinical_features: Clinical feature tensor (B, d_model) or (B, 1, d_model)
        
        Returns:
            Fused features (B, output_dim)
        """
        # Ensure 2D tensors
        if visual_features.dim() == 3:
            visual_features = visual_features.squeeze(1)  # (B, d_model)
        if clinical_features.dim() == 3:
            clinical_features = clinical_features.squeeze(1)  # (B, d_model)
        
        # Project to output dimension
        visual_proj = self.visual_projection(visual_features)  # (B, output_dim)
        clinical_proj = self.clinical_projection(clinical_features)  # (B, output_dim)
        
        # Concatenate for gate input
        combined = torch.cat([visual_proj, clinical_proj], dim=1)  # (B, 2*output_dim)
        
        # Compute gates
        visual_gate = self.visual_gate(combined)  # (B, output_dim)
        clinical_gate = self.clinical_gate(combined)  # (B, output_dim)
        
        # Apply gates
        gated_visual = visual_proj * visual_gate  # (B, output_dim)
        gated_clinical = clinical_proj * clinical_gate  # (B, output_dim)
        
        # Fuse
        fused = torch.cat([gated_visual, gated_clinical], dim=1)  # (B, 2*output_dim)
        fused = self.fusion_projection(fused)  # (B, output_dim)
        
        # Residual connection (optional)
        if self.use_residual and visual_features.shape[-1] == self.output_dim:
            fused = fused + visual_features
        
        return fused
