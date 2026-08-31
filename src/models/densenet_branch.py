"""DenseNet-121 branch for local texture feature extraction.

Uses DenseNet-121 as backbone for capturing fine-grained mammographic patterns.
"""

import logging
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from timm import create_model

logger = logging.getLogger(__name__)


class DenseNetBranch(nn.Module):
    """DenseNet-121 feature extractor.
    
    Extracts local texture features from mammography images.
    Can optionally freeze backbone for transfer learning.
    
    Args:
        config: Configuration dictionary with model settings
    """

    def __init__(self, config: Dict):
        """Initialize DenseNet-121 branch.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self.config = config
        self.model_name = config['model']['densenet']['name']
        self.pretrained = config['model']['densenet']['pretrained']
        self.freeze_backbone = config['model']['densenet']['freeze_backbone']
        self.use_features_only = config['model']['densenet'].get('use_features_only', True)
        
        # Load pretrained DenseNet model
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
        logger.info(f"DenseNet output dimension: {self.output_dim}")
        
        # Freeze backbone if needed
        if self.freeze_backbone:
            self._freeze_backbone()
            logger.info("DenseNet backbone frozen for transfer learning")

    def _get_output_dim(self) -> int:
        """Determine output dimension from backbone.
        
        Returns:
            Output feature dimension
        """
        # DenseNet121 has 1024 features
        if hasattr(self.backbone, 'num_features'):
            return self.backbone.num_features
        else:
            return 1024  # Default for DenseNet121

    def _freeze_backbone(self) -> None:
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input images.
        
        Args:
            x: Input tensor (B, C, H, W) - expects 4 channels (4 views)
              or (B, 3, H, W) for single RGB images
        
        Returns:
            Features (B, output_dim)
        """
        # Handle 4-channel input (4 mammographic views)
        if x.shape[1] == 4:
            # Process each view separately and concatenate
            features_list = []
            for i in range(4):
                # Extract single channel, repeat to RGB
                view = x[:, i:i+1, :, :]  # (B, 1, H, W)
                view = view.repeat(1, 3, 1, 1)  # (B, 3, H, W)
                
                # Extract features
                feat = self.backbone(view)
                features_list.append(feat)
            
            # Concatenate features across views
            features = torch.cat(features_list, dim=1)  # (B, 4*D)
        else:
            # Single image input
            features = self.backbone(x)
        
        # Global pooling if needed
        if features.dim() > 2:
            # If output is (B, C, H, W), apply global average pooling
            features = torch.nn.functional.adaptive_avg_pool2d(features, 1)
            features = features.view(features.size(0), -1)
        
        return features

    def get_gradcam_layer(self) -> nn.Module:
        """Return layer to use for Grad-CAM visualization.
        
        Returns:
            Module for Grad-CAM hook
        """
        # Use final conv layer of DenseNet
        return self.backbone.features

    def get_intermediate_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract intermediate features at different depths.
        
        Useful for multi-scale feature analysis.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Dictionary with features at different depths
        """
        intermediate_features = {}
        
        # Process through dense blocks
        if hasattr(self.backbone, 'features'):
            features = self.backbone.features
            
            x = features.conv0(x)
            x = features.norm0(x)
            x = features.relu0(x)
            x = features.pool0(x)
            
            intermediate_features['pool0'] = x.clone()
            
            # Dense blocks
            x = features.denseblock1(x)
            intermediate_features['denseblock1'] = x.clone()
            
            x = features.transition1(x)
            x = features.denseblock2(x)
            intermediate_features['denseblock2'] = x.clone()
            
            x = features.transition2(x)
            x = features.denseblock3(x)
            intermediate_features['denseblock3'] = x.clone()
            
            x = features.transition3(x)
            x = features.denseblock4(x)
            intermediate_features['denseblock4'] = x.clone()
        
        return intermediate_features
