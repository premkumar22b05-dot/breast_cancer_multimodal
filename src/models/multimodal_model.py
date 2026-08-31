"""Complete multimodal model integrating all components.

Combines visual branches (Swin + DenseNet), clinical processor, cross-attention,
and gated fusion into a unified architecture.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .swin_branch import SwinBranch
from .densenet_branch import DenseNetBranch
from .tabular_transformer import TabularTransformer
from .cross_attention import CrossAttentionLayer
from .gated_fusion import GatedFusion

logger = logging.getLogger(__name__)


class MultimodalModel(nn.Module):
    """Complete multimodal breast cancer detection model.
    
    Architecture:
    1. Visual branches: Swin Transformer + DenseNet-121
    2. Clinical processor: Tabular Transformer
    3. Cross-Attention: Bidirectional attention between modalities
    4. Gated Fusion: Learned weighted combination
    5. Classification head: Binary classification (cancer yes/no)
    
    Args:
        config: Configuration dictionary
        has_clinical_data: Whether clinical data is available
    """

    def __init__(self, config: Dict, has_clinical_data: bool = True):
        """Initialize MultimodalModel.
        
        Args:
            config: Configuration dictionary
            has_clinical_data: Whether to include clinical modality
        """
        super().__init__()
        self.config = config
        self.has_clinical_data = has_clinical_data
        self.d_model = config['model']['d_model']
        
        # Visual branches
        logger.info("Initializing visual branches...")
        self.swin_branch = SwinBranch(config)
        self.densenet_branch = DenseNetBranch(config)
        
        # Project visual features to d_model
        self.swin_projection = nn.Sequential(
            nn.Linear(self.swin_branch.output_dim, self.d_model),
            nn.LayerNorm(self.d_model),
        )
        
        self.densenet_projection = nn.Sequential(
            nn.Linear(self.densenet_branch.output_dim, self.d_model),
            nn.LayerNorm(self.d_model),
        )
        
        # Visual feature fusion (early fusion of two branches)
        self.visual_fusion = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.ReLU(),
            nn.LayerNorm(self.d_model),
        )
        
        # Clinical processor (optional)
        if self.has_clinical_data:
            logger.info("Initializing clinical processor...")
            self.clinical_processor = TabularTransformer(config)
            
            # Cross-Attention
            logger.info("Initializing cross-attention...")
            self.cross_attention = CrossAttentionLayer(config)
            
            # Project clinical to d_model if needed
            if self.clinical_processor.output_dim != self.d_model:
                self.clinical_projection = nn.Sequential(
                    nn.Linear(self.clinical_processor.output_dim, self.d_model),
                    nn.LayerNorm(self.d_model),
                )
            else:
                self.clinical_projection = nn.Identity()
            
            # Gated Fusion
            logger.info("Initializing gated fusion...")
            self.gated_fusion = GatedFusion(config)
            
            final_dim = config['model']['gated_fusion']['output_dim']
        else:
            final_dim = self.d_model
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout']),
            nn.Linear(final_dim // 2, 1),
        )
        
        logger.info(f"MultimodalModel initialized. Final dimension: {final_dim}")

    def forward(
        self,
        images: torch.Tensor,
        clinical_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """Forward pass through multimodal model.
        
        Args:
            images: Mammography images (B, 4, H, W) for 4 views
            clinical_features: Clinical features (B, num_features) or None
        
        Returns:
            Tuple of (logits, intermediate_outputs)
                - logits: (B, 1) prediction logits
                - intermediate_outputs: Dict with intermediate representations
        """
        intermediate_outputs = {}
        
        # Visual processing
        swin_features = self.swin_branch(images)  # (B, swin_dim)
        densenet_features = self.densenet_branch(images)  # (B, densenet_dim)
        
        # Project to d_model
        swin_proj = self.swin_projection(swin_features)  # (B, d_model)
        densenet_proj = self.densenet_projection(densenet_features)  # (B, d_model)
        
        intermediate_outputs['swin_features'] = swin_proj
        intermediate_outputs['densenet_features'] = densenet_proj
        
        # Fuse visual branches
        visual_combined = torch.cat([swin_proj, densenet_proj], dim=1)  # (B, 2*d_model)
        visual_fused = self.visual_fusion(visual_combined)  # (B, d_model)
        
        intermediate_outputs['visual_fused'] = visual_fused
        
        # Clinical processing and multimodal fusion
        if self.has_clinical_data and clinical_features is not None:
            # Process clinical features
            clinical_tokens, clinical_pooled = self.clinical_processor(clinical_features)
            # (B, seq_len, d_model), (B, output_dim)
            
            # Project clinical
            clinical_proj = self.clinical_projection(clinical_pooled)  # (B, d_model)
            
            intermediate_outputs['clinical_features'] = clinical_proj
            
            # Cross-attention
            attended_visual, attended_clinical = self.cross_attention(
                visual_fused, clinical_proj
            )
            # (B, 1, d_model), (B, 1, d_model)
            
            intermediate_outputs['attended_visual'] = attended_visual
            intermediate_outputs['attended_clinical'] = attended_clinical
            
            # Gated fusion
            fused_features = self.gated_fusion(attended_visual, attended_clinical)
            # (B, output_dim)
        else:
            fused_features = visual_fused
        
        intermediate_outputs['fused_features'] = fused_features
        
        # Classification
        logits = self.classifier(fused_features)  # (B, 1)
        
        return logits, intermediate_outputs

    def get_model_info(self) -> Dict:
        """Get model architecture information.
        
        Returns:
            Dictionary with model details
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params,
            'has_clinical': self.has_clinical_data,
            'visual_branches': ['swin', 'densenet'],
        }
        
        return info
