"""Tabular Transformer for processing clinical features.

Transforms categorical and continuous clinical data into contextual tokens.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TabularTransformer(nn.Module):
    """Processes clinical/tabular features using Transformer.
    
    Handles both continuous and categorical features with embeddings.
    Outputs contextual tokens for fusion with visual features.
    
    Args:
        config: Configuration dictionary
    """

    def __init__(self, config: Dict):
        """Initialize Tabular Transformer.
        
        Args:
            config: Configuration dictionary with model settings
        """
        super().__init__()
        self.config = config
        
        # Dimensions
        self.cont_embedding_dim = config['model']['tabular_transformer']['continuous_embedding_dim']
        self.cat_embedding_dim = config['model']['tabular_transformer']['categorical_embedding_dim']
        self.d_model = config['model']['d_model']
        
        # Count features
        self.num_continuous = len(config['data']['continuous_features'])
        self.num_categorical = len(config['data']['categorical_features'])
        self.num_features = self.num_continuous + self.num_categorical
        
        # Continuous feature embeddings
        if self.num_continuous > 0:
            self.continuous_embeddings = nn.ModuleList([
                nn.Linear(1, self.cont_embedding_dim)
                for _ in range(self.num_continuous)
            ])
        
        # Categorical feature embeddings
        if self.num_categorical > 0:
            self.categorical_embeddings = nn.ModuleList([
                nn.Embedding(10, self.cat_embedding_dim)  # 10 categories per feature
                for _ in range(self.num_categorical)
            ])
        
        # Feature embedding projection
        total_embed_dim = (
            self.num_continuous * self.cont_embedding_dim +
            self.num_categorical * self.cat_embedding_dim
        )
        self.feature_projection = nn.Linear(total_embed_dim, self.d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config['model']['tabular_transformer']['num_heads'],
            dim_feedforward=config['model']['tabular_transformer']['ffn_dim'],
            dropout=config['model']['dropout'],
            activation=config['model']['tabular_transformer']['activation'],
            batch_first=True,
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config['model']['tabular_transformer']['num_layers'],
        )
        
        # Output projection
        self.output_dim = config['model']['tabular_transformer']['output_dim']
        self.output_projection = nn.Sequential(
            nn.Linear(self.d_model, self.output_dim),
            nn.LayerNorm(self.output_dim),
        )
        
        logger.info(
            f"TabularTransformer: {self.num_continuous} continuous, "
            f"{self.num_categorical} categorical features -> {self.output_dim}D"
        )

    def forward(self, clinical_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process clinical features.
        
        Args:
            clinical_features: Tensor of shape (B, num_features)
                              Contains continuous features first, then categorical
        
        Returns:
            Tuple of (tokens, pooled_output)
                - tokens: (B, num_features, d_model) contextual tokens
                - pooled_output: (B, output_dim) aggregated representation
        """
        embeddings = []
        
        # Embed continuous features
        if self.num_continuous > 0:
            cont_features = clinical_features[:, :self.num_continuous]  # (B, num_cont)
            for i, embed_layer in enumerate(self.continuous_embeddings):
                feat = cont_features[:, i:i+1]  # (B, 1)
                embedded = embed_layer(feat)  # (B, cont_embedding_dim)
                embeddings.append(embedded)
        
        # Embed categorical features
        if self.num_categorical > 0:
            cat_features = clinical_features[:, self.num_continuous:]  # (B, num_cat)
            for i, embed_layer in enumerate(self.categorical_embeddings):
                feat = (cat_features[:, i] * 9).long()  # Convert to integer index
                feat = torch.clamp(feat, 0, 9)  # Ensure within bounds
                embedded = embed_layer(feat)  # (B, cat_embedding_dim)
                embeddings.append(embedded)
        
        # Concatenate all embeddings
        if embeddings:
            embedded = torch.cat(embeddings, dim=1)  # (B, total_embed_dim)
        else:
            # If no features, create dummy embedding
            embedded = torch.zeros(
                clinical_features.size(0),
                self.d_model,
                device=clinical_features.device,
                dtype=clinical_features.dtype,
            )
            return embedded.unsqueeze(1), embedded  # Return dummy with shape (B, 1, d_model)
        
        # Project to d_model dimension
        projected = self.feature_projection(embedded)  # (B, d_model)
        
        # Add sequence dimension for transformer
        tokens = projected.unsqueeze(1)  # (B, 1, d_model)
        
        # Apply transformer encoder
        tokens = self.transformer_encoder(tokens)  # (B, 1, d_model)
        
        # Pool to get output
        pooled_output = self.output_projection(tokens.mean(dim=1))  # (B, output_dim)
        
        return tokens, pooled_output
