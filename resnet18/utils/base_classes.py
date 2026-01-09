"""
base_classes.py - Base classes for training

Provides abstract base classes for trainers and models.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Tuple, Dict


class BaseTrainer(ABC):
    """
    Abstract base trainer class.
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    @abstractmethod
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train one epoch. Returns (loss, mae)."""
        pass
    
    @abstractmethod
    def validate(self, epoch: int) -> Tuple[float, float]:
        """Validate. Returns (loss, mae)."""
        pass
    
    @abstractmethod
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool):
        """Save model checkpoint."""
        pass
    
    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        pass


class BaseModel(nn.Module, ABC):
    """
    Abstract base model class.
    """
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass."""
        pass
    
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        
        return {
            'total': total,
            'trainable': trainable,
            'frozen': frozen
        }
    
    def print_summary(self):
        """Print model summary."""
        params = self.count_parameters()
        print(f"\n{'='*60}")
        print("MODEL SUMMARY")
        print(f"{'='*60}")
        print(f"  Total parameters:     {params['total']:,}")
        print(f"  Trainable parameters: {params['trainable']:,}")
        print(f"  Frozen parameters:    {params['frozen']:,}")
        print(f"{'='*60}\n")
