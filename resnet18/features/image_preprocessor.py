"""
image_preprocessor.py - Grayscale silhouette preprocessing

Processes silhouette images to 1-channel grayscale.
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union
import pickle

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from resnet18.config.config import Config


class ImagePreprocessor:
    """
    Preprocess silhouette images to grayscale.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.target_size = config.image.TARGET_SIZE  # (W, H)
        self.channels = config.image.CHANNELS
        self.mean = np.array(config.image.PIXEL_MEAN, dtype=np.float32)
        self.std = np.array(config.image.PIXEL_STD, dtype=np.float32)
        self.fitted = False
    
    def fit(self):
        """Mark as fitted."""
        self.fitted = True
        print(f"✓ ImagePreprocessor fitted")
        print(f"  Target size: {self.target_size}")
        print(f"  Channels: {self.channels}")
        return self
    
    def transform(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Transform silhouette image to grayscale.
        
        Args:
            image_path: Path to image
            
        Returns:
            Preprocessed image (H, W, 1) in range [-1, 1]
        """
        # Load image
        img = Image.open(image_path)
        
        # Convert to grayscale
        if img.mode != 'L':
            img = img.convert('L')
        
        # Resize
        img = img.resize(self.target_size, Image.LANCZOS)
        
        # Convert to numpy
        img_array = np.array(img, dtype=np.float32)
        
        # Normalize to [0, 1]
        img_array = img_array / 255.0
        
        # Add channel dimension (H, W) -> (H, W, 1)
        img_array = np.expand_dims(img_array, axis=-1)
        
        # Standardize with mean/std
        if self.config.image.NORMALIZE:
            img_array = (img_array - self.mean) / self.std
        
        return img_array
    
    def inverse_transform(self, image_array: np.ndarray) -> np.ndarray:
        """Denormalize image."""
        if self.config.image.NORMALIZE:
            image_array = (image_array * self.std) + self.mean
        
        image_array = np.clip(image_array * 255.0, 0, 255).astype(np.uint8)
        return image_array
    
    def save_params(self, filepath: str):
        """Save preprocessor parameters."""
        params = {
            'target_size': self.target_size,
            'channels': self.channels,
            'mean': self.mean,
            'std': self.std,
            'fitted': self.fitted
        }
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)
        print(f"✓ Saved image preprocessor params to {filepath}")
    
    def load_params(self, filepath: str):
        """Load preprocessor parameters."""
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        
        self.target_size = params['target_size']
        self.channels = params['channels']
        self.mean = params['mean']
        self.std = params['std']
        self.fitted = params['fitted']
        
        print(f"✓ Loaded image preprocessor params from {filepath}")
