"""
losses.py - Loss Functions for Hip & Bust Prediction

Includes:
- Wing Loss (best for body measurements)
- Smooth L1 Loss
- Combined losses
- Per-measurement weighted losses
"""

import torch
import torch.nn as nn
import numpy as np


class WeightedMSELoss(nn.Module):
    """MSE loss with per-measurement weights."""
    
    def __init__(self, hip_weight: float = 1.0, bust_weight: float = 1.0):
        super().__init__()
        self.weights = torch.tensor([hip_weight, bust_weight], dtype=torch.float32)
        print(f"✓ WeightedMSELoss: hip={hip_weight}, bust={bust_weight}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        errors = (pred - target) ** 2
        weighted_errors = errors * self.weights.to(pred.device)
        return weighted_errors.mean()


class AdaptiveLoss(nn.Module):
    """Combination of MSE and MAE (more robust to outliers)."""
    
    def __init__(self, alpha: float = 0.7):
        super().__init__()
        self.alpha = alpha
        self.beta = 1 - alpha
        print(f"✓ AdaptiveLoss: MSE={alpha}, MAE={self.beta}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = ((pred - target) ** 2).mean()
        mae = torch.abs(pred - target).mean()
        return self.alpha * mse + self.beta * mae


class SmoothL1Loss(nn.Module):
    """
    Smooth L1 Loss (Huber Loss variant)
    Less sensitive to outliers than MSE.
    """
    
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta
        print(f"✓ SmoothL1Loss: beta={beta}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(pred - target)
        loss = torch.where(
            diff < self.beta,
            0.5 * (diff ** 2) / self.beta,
            diff - 0.5 * self.beta
        )
        return loss.mean()


class WingLoss(nn.Module):
    """
    Wing Loss - Designed for landmark/measurement prediction
    
    Better than MSE for body measurements:
    - More attention to small errors (hip/bust precision)
    - Less influenced by large outliers
    - State-of-the-art for body landmark prediction
    
    Reference: "Wing Loss for Robust Facial Landmark Localisation with CNNs"
    https://arxiv.org/abs/1711.06753
    
    Parameters:
        omega: Threshold for switching between linear and non-linear parts (default: 10.0)
        epsilon: Controls the curvature of the non-linear region (default: 2.0)
    """
    
    def __init__(self, omega: float = 10.0, epsilon: float = 2.0):
        super().__init__()
        self.omega = omega
        self.epsilon = epsilon
        # Constant C ensures continuity at |x| = omega
        self.C = self.omega - self.omega * np.log(1 + self.omega / self.epsilon)
        print(f"✓ WingLoss: omega={omega}, epsilon={epsilon}, C={self.C:.4f}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, 2) predictions [hip, bust]
            target: (B, 2) ground truth [hip, bust]
        
        Returns:
            Wing loss value
        """
        diff = torch.abs(pred - target)
        
        # Wing loss formula:
        # For small errors (< omega): logarithmic (amplifies attention)
        # For large errors (>= omega): linear (less penalty than MSE)
        loss = torch.where(
            diff < self.omega,
            self.omega * torch.log(1 + diff / self.epsilon),
            diff - self.C
        )
        
        return loss.mean()


class HipBustLoss(nn.Module):
    """
    Custom loss combining Wing Loss + Smooth L1
    Optimized specifically for hip and bust prediction
    """
    
    def __init__(
        self, 
        wing_weight: float = 0.7, 
        smooth_weight: float = 0.3,
        omega: float = 10.0,
        epsilon: float = 2.0,
        beta: float = 1.0
    ):
        super().__init__()
        self.wing = WingLoss(omega=omega, epsilon=epsilon)
        self.smooth = SmoothL1Loss(beta=beta)
        self.wing_weight = wing_weight
        self.smooth_weight = smooth_weight
        
        print(f"✓ HipBustLoss:")
        print(f"  Wing weight: {wing_weight}, Smooth weight: {smooth_weight}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        wing_loss = self.wing(pred, target)
        smooth_loss = self.smooth(pred, target)
        combined = self.wing_weight * wing_loss + self.smooth_weight * smooth_loss
        return combined


class PerMeasurementWingLoss(nn.Module):
    """Wing Loss with different weights for hip and bust."""
    
    def __init__(
        self, 
        hip_weight: float = 1.0, 
        bust_weight: float = 1.0,
        omega: float = 10.0,
        epsilon: float = 2.0
    ):
        super().__init__()
        self.weights = torch.tensor([hip_weight, bust_weight], dtype=torch.float32)
        self.omega = omega
        self.epsilon = epsilon
        self.C = self.omega - self.omega * np.log(1 + self.omega / self.epsilon)
        
        print(f"✓ PerMeasurementWingLoss:")
        print(f"  Hip weight: {hip_weight}, Bust weight: {bust_weight}")
        print(f"  Omega: {omega}, Epsilon: {epsilon}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(pred - target)
        
        loss = torch.where(
            diff < self.omega,
            self.omega * torch.log(1 + diff / self.epsilon),
            diff - self.C
        )
        
        # Apply per-measurement weights
        weighted_loss = loss * self.weights.to(pred.device)
        return weighted_loss.mean()


def create_loss(loss_type: str = "wing", **kwargs):
    """
    Factory function to create loss functions
    
    Args:
        loss_type: One of ["mse", "adaptive", "smooth_l1", "wing", "hip_bust", "per_measurement_wing"]
        **kwargs: Loss-specific parameters
    
    Returns:
        Loss function instance
    """
    loss_registry = {
        "mse": WeightedMSELoss,
        "adaptive": AdaptiveLoss,
        "smooth_l1": SmoothL1Loss,
        "wing": WingLoss,
        "hip_bust": HipBustLoss,
        "per_measurement_wing": PerMeasurementWingLoss
    }
    
    if loss_type not in loss_registry:
        raise ValueError(f"Unknown loss type: {loss_type}. Choose from {list(loss_registry.keys())}")
    
    return loss_registry[loss_type](**kwargs)


if __name__ == "__main__":
    """Test loss functions"""
    print("Testing loss functions...\n")
    
    # Create dummy data
    pred = torch.tensor([[100.0, 90.0], [105.0, 88.0], [98.0, 92.0]])
    target = torch.tensor([[102.0, 91.0], [103.0, 89.0], [100.0, 90.0]])
    
    print(f"Predictions: {pred}")
    print(f"Targets:     {target}\n")
    
    # Test all losses
    losses = {
        "MSE": WeightedMSELoss(),
        "Adaptive": AdaptiveLoss(),
        "Smooth L1": SmoothL1Loss(),
        "Wing": WingLoss(),
        "HipBust": HipBustLoss(),
        "PerMeasurementWing": PerMeasurementWingLoss()
    }
    
    for name, loss_fn in losses.items():
        loss_value = loss_fn(pred, target)
        print(f"{name:20s}: {loss_value.item():.6f}")
    
    print("\n✅ All loss functions working!")
