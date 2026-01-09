"""
measurement_preprocessor.py - Hip & Bust measurement preprocessing

Only processes hip and chest (bust) measurements.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union
import pickle

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from resnet18.config.config import Config


class MeasurementPreprocessor:
    """
    Standardize hip and bust measurements.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.measurement_cols = config.measurement.MEASUREMENT_COLUMNS
        
        # Stats for measurements
        self.measurement_mean = None
        self.measurement_std = None
        
        # Stats for height
        self.height_mean = None
        self.height_std = None
        
        self.fitted = False
    
    def fit(self, df: pd.DataFrame):
        """
        Fit preprocessor on training data.
        
        Args:
            df: DataFrame with measurements
        """
        # Compute measurement statistics
        measurements = df[self.measurement_cols].values
        self.measurement_mean = measurements.mean(axis=0)
        self.measurement_std = measurements.std(axis=0)
        
        # Compute height statistics
        self.height_mean = df['height_cm'].mean()
        self.height_std = df['height_cm'].std()
        
        self.fitted = True
        
        print("✓ MeasurementPreprocessor fitted")
        print(f"  Hip   - Mean: {self.measurement_mean[0]:.2f}, Std: {self.measurement_std[0]:.2f}")
        print(f"  Bust  - Mean: {self.measurement_mean[1]:.2f}, Std: {self.measurement_std[1]:.2f}")
        print(f"  Height - Mean: {self.height_mean:.2f}, Std: {self.height_std:.2f}")
        
        return self
    
    def transform(self, measurements: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Standardize measurements."""
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted first")
        
        if isinstance(measurements, pd.DataFrame):
            measurements = measurements[self.measurement_cols].values
        
        standardized = (measurements - self.measurement_mean) / self.measurement_std
        return standardized.astype(np.float32)
    
    def inverse_transform(self, standardized: np.ndarray) -> np.ndarray:
        """Denormalize measurements back to centimeters."""
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted first")
        
        original = (standardized * self.measurement_std) + self.measurement_mean
        return original
    
    def transform_height(self, height: float) -> float:
        """Standardize height."""
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted first")
        
        return (height - self.height_mean) / self.height_std
    
    def inverse_transform_height(self, standardized_height: float) -> float:
        """Denormalize height."""
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted first")
        
        return (standardized_height * self.height_std) + self.height_mean
    
    def save_params(self, filepath: str):
        """Save fitted parameters."""
        if not self.fitted:
            raise ValueError("Cannot save unfitted preprocessor")
        
        params = {
            'measurement_mean': self.measurement_mean,
            'measurement_std': self.measurement_std,
            'height_mean': self.height_mean,
            'height_std': self.height_std,
            'measurement_cols': self.measurement_cols,
            'fitted': self.fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)
        
        print(f"✓ Saved measurement preprocessor params to {filepath}")
    
    def load_params(self, filepath: str):
        """Load fitted parameters."""
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        
        self.measurement_mean = params['measurement_mean']
        self.measurement_std = params['measurement_std']
        self.height_mean = params['height_mean']
        self.height_std = params['height_std']
        self.measurement_cols = params['measurement_cols']
        self.fitted = params['fitted']
        
        print(f"✓ Loaded measurement preprocessor params from {filepath}")
