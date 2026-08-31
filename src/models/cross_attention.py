"""Cross-Attention mechanism for multimodal fusion.

Implements bidirectional attention between visual and clinical modalities.
"""

import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CrossAttentionLayer(nn.Module):
    """Cross-Attention layer for multimodal fusion.
    
    Allows visual features to attend to clinical features and vice versa.
    Supports bidirectional, visual-to-clinical, or clinical-to-visual attention.
    
    Args:
        config: Configuration dictionary
    """

    def __init__(self, config: Dict):
        """Initialize Cross-Attention layer.
        
        Args:
            config: Configuration dictionary with model settings
        """
        super().__init__()
        self.config = config
        
        self.d_model = config['model']['d_model']
        self.num_heads = config['model']['cross_attention']['num_heads']
        self.mode = config['model']['cross_attention']['mode']  # bidirectional, visual_to_clinical, etc.
        
        # Multi-head attention mechanisms
        self.visual_to_clinical_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.num_heads,
            dropout=config['model']['cross_attention']['dropout'],
            batch_first=True,
        )
        
        self.clinical_to_visual_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.num_heads,
            dropout=config['model']['cross_attention']['dropout'],
            batch_first=True,
        )
        
        # Layer normalization
        self.use_layer_norm = config['model']['cross_attention']['use_layer_norm']
        if self.use_layer_norm:
            self.norm1 = nn.LayerNorm(self.d_model)
            self.norm2 = nn.LayerNorm(self.d_model)
            self.norm3 = nn.LayerNorm(self.d_model)
            self.norm4 = nn.LayerNorm(self.d_model)
        
        # Feed-forward networks for residual connections
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 4),
            nn.ReLU(),
            nn.Dropout(config['model']['cross_attention']['dropout']),
            nn.Linear(self.d_model * 4, self.d_model),
        )
        
        logger.info(
            f"CrossAttentionLayer: {self.mode} mode, {self.num_heads} heads"
        )

    def forward(
        self,
        visual_features: torch.Tensor,
        clinical_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply cross-attention.
        
        Args:
            visual_features: Visual tokens (B, seq_len_v, d_model)
            clinical_features: Clinical tokens (B, seq_len_c, d_model)
        
        Returns:
            Tuple of (attended_visual, attended_clinical)
                - attended_visual: (B, seq_len_v, d_model)
                - attended_clinical: (B, seq_len_c, d_model)
        """
        # Ensure proper dimensions
        if visual_features.dim() == 2:
            visual_features = visual_features.unsqueeze(1)  # (B, 1, d_model)
        if clinical_features.dim() == 2:
            clinical_features = clinical_features.unsqueeze(1)  # (B, 1, d_model)
        
        # Visual attends to clinical
        attended_visual_to_clinical, _ = self.visual_to_clinical_attn(
            query=visual_features,
            key=clinical_features,
            value=clinical_features,
        )
        
        # Residual connection
        if self.use_layer_norm:
            attended_visual_to_clinical = self.norm1(
                visual_features + attended_visual_to_clinical
            )
            attended_visual_to_clinical = self.norm2(
                attended_visual_to_clinical + self.ffn(attended_visual_to_clinical)
            )
        else:
            attended_visual_to_clinical = visual_features + attended_visual_to_clinical
        
        # Clinical attends to visual (for bidirectional mode)
        if self.mode in ['bidirectional', 'clinical_to_visual']:
            attended_clinical_to_visual, _ = self.clinical_to_visual_attn(
                query=clinical_features,
                key=visual_features,
                value=visual_features,
            )
            
            # Residual connection
            if self.use_layer_norm:
                attended_clinical_to_visual = self.norm3(
                    clinical_features + attended_clinical_to_visual
                )
                attended_clinical_to_visual = self.norm4(
                    attended_clinical_to_visual + self.ffn(attended_clinical_to_visual)
                )
            else:
                attended_clinical_to_visual = clinical_features + attended_clinical_to_visual
        else:
            attended_clinical_to_visual = clinical_features
        
        return attended_visual_to_clinical, attended_clinical_to_visual
