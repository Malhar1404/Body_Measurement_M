"""
body_measurement_model.py - Multi-view ResNet Model for Body Measurement

This module implements a ResNet-50 based model that:
1. Processes front and side silhouette views
2. Fuses visual features with height metadata
3. Predicts 14 body measurements
"""

import torch
import torch.nn as nn
import timm
from typing import Dict
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config.config import Config


class BodyMeasurementModel(nn.Module):
    """
    Multi-view body measurement prediction model.
    
    Architecture:
    - Two ResNet-50 backbones (shared weights) for front/side views
    - Feature fusion layer combining visual + height features
    - Regression head for 14 measurements
    """
    
    def __init__(self, config: Config, pretrained: bool = True):
        """
        Initialize the model.
        
        Args:
            config: Configuration object
            pretrained: Whether to use ImageNet pretrained weights
        """
        super(BodyMeasurementModel, self).__init__()
        
        self.config = config
        self.num_measurements = config.model.NUM_MEASUREMENTS
        self.use_height = config.model.USE_HEIGHT
        
        # Load pretrained ResNet-50
        self.backbone = timm.create_model(
            'resnet50',
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool='avg'  # Global average pooling
        )
        
        # ResNet-50 outputs 2048 features after global pooling
        self.feature_dim = 2048
        
        # Feature fusion: 2 views × 2048 features + 1 height = 4097 features
        fusion_input_dim = self.feature_dim * 2  # Front + side views
        if self.use_height:
            fusion_input_dim += 1  # Add height feature
        
        # Fusion and regression layers
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        # Final regression head
        self.regression_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_measurements)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize fusion and regression layers."""
        for m in self.fusion.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Initialize regression head with smaller weights
        for m in self.regression_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, mask: torch.Tensor, mask_left: torch.Tensor, 
                height: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            mask: Front view images (B, 3, 683, 512)
            mask_left: Side view images (B, 3, 683, 512)
            height: Normalized height values (B,)
            
        Returns:
            Predicted measurements (B, 14)
        """
        batch_size = mask.size(0)
        
        # Extract features from both views using shared backbone
        features_front = self.backbone(mask)  # (B, 2048)
        features_side = self.backbone(mask_left)  # (B, 2048)
        
        # Concatenate features from both views
        combined_features = torch.cat([features_front, features_side], dim=1)  # (B, 4096)
        
        # Add height if used
        if self.use_height:
            height = height.view(batch_size, 1)  # (B, 1)
            combined_features = torch.cat([combined_features, height], dim=1)  # (B, 4097)
        
        # Fusion layers
        fused_features = self.fusion(combined_features)  # (B, 512)
        
        # Regression head
        measurements = self.regression_head(fused_features)  # (B, 14)
        
        return measurements
    
    def get_feature_maps(self, mask: torch.Tensor, mask_left: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract intermediate feature maps (for visualization/debugging).
        
        Args:
            mask: Front view images
            mask_left: Side view images
            
        Returns:
            Dictionary with feature maps
        """
        with torch.no_grad():
            features_front = self.backbone(mask)
            features_side = self.backbone(mask_left)
        
        return {
            'features_front': features_front,
            'features_side': features_side
        }


class LightweightBodyMeasurementModel(nn.Module):
    """
    Lightweight version using ResNet-18 (for faster training/inference).
    """
    
    def __init__(self, config: Config, pretrained: bool = True):
        """Initialize lightweight model."""
        super(LightweightBodyMeasurementModel, self).__init__()
        
        self.config = config
        self.num_measurements = config.model.NUM_MEASUREMENTS
        self.use_height = config.model.USE_HEIGHT
        
        # Load pretrained ResNet-18 (lighter than ResNet-50)
        self.backbone = timm.create_model(
            'resnet18',
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )
        
        # ResNet-18 outputs 512 features
        self.feature_dim = 512
        
        fusion_input_dim = self.feature_dim * 2
        if self.use_height:
            fusion_input_dim += 1
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.regression_head = nn.Linear(128, self.num_measurements)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights."""
        for m in self.fusion.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        nn.init.normal_(self.regression_head.weight, mean=0, std=0.01)
        nn.init.constant_(self.regression_head.bias, 0)
    
    def forward(self, mask: torch.Tensor, mask_left: torch.Tensor, 
                height: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch_size = mask.size(0)
        
        features_front = self.backbone(mask)
        features_side = self.backbone(mask_left)
        
        combined_features = torch.cat([features_front, features_side], dim=1)
        
        if self.use_height:
            height = height.view(batch_size, 1)
            combined_features = torch.cat([combined_features, height], dim=1)
        
        fused_features = self.fusion(combined_features)
        measurements = self.regression_head(fused_features)
        
        return measurements


def create_model(config: Config, model_type: str = 'resnet50', pretrained: bool = True) -> nn.Module:
    """
    Factory function to create model.
    
    Args:
        config: Configuration object
        model_type: 'resnet50' or 'resnet18'
        pretrained: Use ImageNet pretrained weights
        
    Returns:
        Model instance
    """
    if model_type == 'resnet50':
        model = BodyMeasurementModel(config, pretrained=pretrained)
    elif model_type == 'resnet18':
        model = LightweightBodyMeasurementModel(config, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


# Quick test
if __name__ == "__main__":
    from config.config import Config
    
    print("="*80)
    print("Testing Body Measurement Model")
    print("="*80)
    
    config = Config()
    
    # Create model
    print("\n✓ Creating ResNet-50 model...")
    model = create_model(config, model_type='resnet50', pretrained=True)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    print("\n✓ Testing forward pass...")
    batch_size = 4
    mask = torch.randn(batch_size, 3, 683, 512)
    mask_left = torch.randn(batch_size, 3, 683, 512)
    height = torch.randn(batch_size)
    
    model.eval()
    with torch.no_grad():
        output = model(mask, mask_left, height)
    
    print(f"✓ Input shapes:")
    print(f"  mask: {mask.shape}")
    print(f"  mask_left: {mask_left.shape}")
    print(f"  height: {height.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Expected: ({batch_size}, {config.model.NUM_MEASUREMENTS})")
    
    # Test lightweight model
    print("\n" + "="*80)
    print("Testing Lightweight Model (ResNet-18)")
    print("="*80)
    
    light_model = create_model(config, model_type='resnet18', pretrained=True)
    light_params = sum(p.numel() for p in light_model.parameters())
    
    print(f"✓ Lightweight model parameters: {light_params:,}")
    print(f"✓ Reduction: {(1 - light_params/total_params)*100:.1f}% fewer parameters")
    
    with torch.no_grad():
        light_output = light_model(mask, mask_left, height)
    print(f"✓ Lightweight output shape: {light_output.shape}")
    
    print("\n✅ Model test completed successfully!")
    print("\nModel Summary:")
    print(f"  ResNet-50: {total_params:,} parameters")
    print(f"  ResNet-18: {light_params:,} parameters")
    print(f"  Input: 2 views (512×683) + height")
    print(f"  Output: 14 measurements")
