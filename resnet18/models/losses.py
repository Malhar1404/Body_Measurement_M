"""
losses.py - Custom loss functions for Hip & Bust prediction
"""

import torch
import torch.nn as nn


class WeightedMSELoss(nn.Module):
    """
    MSE loss with per-measurement weights.
    Give equal weight to hip and bust (or customize).
    """
    
    def __init__(self, hip_weight: float = 1.0, bust_weight: float = 1.0):
        super().__init__()
        self.weights = torch.tensor([hip_weight, bust_weight], dtype=torch.float32)
        print(f"✓ WeightedMSELoss: hip={hip_weight}, bust={bust_weight}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, 2)
            target: (B, 2)
        
        Returns:
            Weighted MSE loss
        """
        errors = (pred - target) ** 2
        weighted_errors = errors * self.weights.to(pred.device)
        return weighted_errors.mean()


class AdaptiveLoss(nn.Module):
    """
    Combination of MSE and MAE (more robust to outliers).
    """
    
    def __init__(self, alpha: float = 0.7):
        super().__init__()
        self.alpha = alpha
        self.beta = 1 - alpha
        print(f"✓ AdaptiveLoss: MSE={alpha}, MAE={self.beta}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = ((pred - target) ** 2).mean()
        mae = torch.abs(pred - target).mean()
        return self.alpha * mse + self.beta * mae
