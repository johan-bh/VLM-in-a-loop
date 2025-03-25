import torch
import torch.nn as nn
from typing import Any

class FusionModule(nn.Module):
    """
    A fusion module that projects vision model features to a space compatible with
    the language model's hidden representation.
    """
    def __init__(self, vision_dim: int, language_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(vision_dim, language_dim)
    
    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Projects visual features into the language model's feature space.
        
        Args:
            visual_features (torch.Tensor): Tensor of shape (B, vision_dim)
            
        Returns:
            torch.Tensor: Fused features of shape (B, language_dim)
        """
        return self.projection(visual_features)
