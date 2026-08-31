"""Swin Transformer branch for global structure feature extraction.

Uses Swin Transformer as backbone for capturing global mammographic patterns.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from timm import create_model

logger = logging.getLogger(__name__)


class SwinBranch(nn.Module):
    """Swin Transformer feature extractor.
    
    Extracts global structural features from mammography images.
    Can optionally freeze backbone for transfer learning.
    
    Args:
        config: Configuration dictionary with model settings
    """

    def __init__(self, config: Dict):
        """Initialize Swin Transformer branch.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self.config = config
        self.model_name = config['model']['swin']['name']
        self.pretrained = config['model']['swin']['pretrained']
        self.freeze_backbone = config['model']['swin']['freeze_backbone']
        self.use_features_only = config['model']['swin'].get('use_features_only', True)
        
        # Load pretrained Swin model
        logger.info(f"Loading {self.model_name} (pretrained={self.pretrained})")
        
        try:
            self.backbone = create_model(
                self.model_name,
                pretrained=self.pretrained,
                num_classes=0,  # Remove classification head
                global_pool='',  # No pooling
            )
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
        
        # Get output dimension
        self.output_dim = self._get_output_dim()
        logger.info(f"Swin output dimension: {self.output_dim}")
        
        # Freeze backbone if needed
        if self.freeze_backbone:
            self._freeze_backbone()
            logger.info("Swin backbone frozen for transfer learning")

    def _get_output_dim(self) -> int:
        """Determine output dimension from backbone.
        
        Returns:
            Output feature dimension
        """
        # Try to infer from model structure
        if hasattr(self.backbone, 'num_features'):
            return self.backbone.num_features
        elif hasattr(self.backbone, 'embed_dim'):
            return self.backbone.embed_dim * 4  # Swin uses hierarchical features
        else:
            # Default for Swin Tiny
            return 768

    def _freeze_backbone(self) -> None:
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input images.
        
        Args:
            x: Input tensor (B, C, H, W) - expects 4 channels (4 views)
              or (B, C, H, W) where C=3 for single images
        
        Returns:
            Features (B, output_dim)
        """
        # Handle 4-channel input (4 mammographic views)
        if x.shape[1] == 4:
            # Process each view separately and average
            features_list = []
            for i in range(4):
                # Extract single channel, repeat to RGB
                view = x[:, i:i+1, :, :]  # (B, 1, H, W)
                view = view.repeat(1, 3, 1, 1)  # (B, 3, H, W)
                
                # Extract features
                feat = self.backbone(view)
                features_list.append(feat)
            
            # Average features across views
            features = torch.stack(features_list, dim=1)  # (B, 4, D)
            features = features.mean(dim=1)  # (B, D)
        else:
            # Single image input
            features = self.backbone(x)
        
        # Global pooling if needed
        if features.dim() > 2:
            # If output is (B, C, H, W), apply global average pooling
            features = torch.nn.functional.adaptive_avg_pool2d(features, 1)
            features = features.view(features.size(0), -1)
        
        return features

    def get_attention_maps(self, x: torch.Tensor) -> Dict:
        """Extract attention maps from Swin Transformer.
        
        Useful for interpretability (Grad-CAM, attention rollout).
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Dictionary with intermediate features
        """
        attention_maps = {}
        
        # Hook into intermediate layers
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    attention_maps[name] = output.detach()
            return hook
        
        # Register hooks on attention layers
        handles = []
        for name, module in self.backbone.named_modules():
            if 'attn' in name.lower():
                h = module.register_forward_hook(hook_fn(name))
                handles.append(h)
        
        # Forward pass
        with torch.no_grad():
            _ = self.forward(x)
        
        # Remove hooks
        for h in handles:
            h.remove()
        
        return attention_maps
