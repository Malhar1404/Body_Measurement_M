"""
model.py - ResNet-18 model for Hip & Bust prediction

Architecture:
- 1-channel input (grayscale silhouettes)
- ResNet-18 backbone
- Dual-view fusion
- 2 outputs: hip, bust
"""

import torch
import torch.nn as nn
import torchvision.models as models

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from resnet18.config.config import Config


class HipBustModel(nn.Module):
    """
    Hip and Bust prediction model.
    
    Input: 2 grayscale silhouettes (front + side) + height
    Output: 2 measurements (hip, bust)
    """
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # ResNet-18 backbone
        self.backbone = models.resnet18(pretrained=config.model.PRETRAINED)
        
        # Modify first conv layer for 1-channel input
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=1,  # Grayscale!
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        
        # Initialize from pretrained RGB weights (average across channels)
        if config.model.PRETRAINED:
            with torch.no_grad():
                self.backbone.conv1.weight[:, 0, :, :] = original_conv1.weight.mean(dim=1)
            print("✓ Initialized 1-channel conv from pretrained RGB weights")
        
        # Remove final FC layer
        self.backbone.fc = nn.Identity()
        
        # Feature dimension for ResNet-18
        feature_dim = 512
        
        # Fusion layer (combine front + side views)
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Regression head
        input_features = 256
        if config.model.USE_HEIGHT:
            input_features += 1
        
        self.regression_head = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Only 2 outputs: hip, bust
        )
        
        print(f"✓ Created HipBustModel")
        print(f"  Backbone: ResNet-18")
        print(f"  Input channels: 1 (grayscale)")
        print(f"  Outputs: 2 (hip, bust)")
    
    def forward(self, mask_front: torch.Tensor, mask_side: torch.Tensor, 
                height: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            mask_front: Front view (B, 1, H, W)
            mask_side: Side view (B, 1, H, W)
            height: Height (B,)
        
        Returns:
            Predictions (B, 2) - [hip, bust]
        """
        # Extract features from both views
        features_front = self.backbone(mask_front)  # (B, 512)
        features_side = self.backbone(mask_side)    # (B, 512)
        
        # Concatenate
        combined = torch.cat([features_front, features_side], dim=1)  # (B, 1024)
        
        # Fuse
        fused = self.fusion(combined)  # (B, 256)
        
        # Add height
        if self.config.model.USE_HEIGHT:
            height = height.unsqueeze(1)  # (B, 1)
            fused = torch.cat([fused, height], dim=1)  # (B, 257)
        
        # Predict
        predictions = self.regression_head(fused)  # (B, 2)
        
        return predictions


def create_model(config: Config) -> nn.Module:
    """Factory function to create model."""
    return HipBustModel(config)
