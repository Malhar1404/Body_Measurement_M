"""
measurement_preprocessor.py - Preprocessing for Body Measurements

This module handles normalization and denormalization of body measurement values
for training and inference.
"""

import numpy as np
import pandas as pd
from typing import Dict, Union
import pickle
import logging

from pathlib import Path
from src.utils.base_classes import BasePreprocessor

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class MeasurementPreprocessor(BasePreprocessor):
    """
    Preprocessor for body measurements and metadata.
    
    Handles:
    - Z-score normalization: (x - mean) / std
    - Separate height normalization for model input
    - Inverse transformation to convert predictions back to cm
    - Parameter persistence for inference
    """
    
    def __init__(self, config):
        """Initialize measurement preprocessor."""
        super().__init__(config)
        self.logger = logger
        self.measurement_cols = config.measurement.MEASUREMENT_COLUMNS
        self.normalization_method = config.measurement.NORMALIZATION_METHOD
        
        # Storage for fitted parameters
        self.measurement_mean_ = None
        self.measurement_std_ = None
        self.height_mean_ = None
        self.height_std_ = None
    
    def fit(self, data: pd.DataFrame) -> 'MeasurementPreprocessor':
        """
        Learn normalization parameters from training data.
        
        IMPORTANT: Only fit on training data, never on validation/test!
        
        Args:
            data: Training DataFrame with measurement columns
            
        Returns:
            Self (for method chaining)
        """
        self.logger.info("Fitting measurement preprocessor...")
        
        # Fit measurement normalization (all 14 measurements)
        measurements = data[self.measurement_cols]
        self.measurement_mean_ = measurements.mean().values
        self.measurement_std_ = measurements.std().values
        
        # Fit height normalization (if using height as input)
        if self.config.model.USE_HEIGHT and 'height_cm' in data.columns:
            self.height_mean_ = data['height_cm'].mean()
            self.height_std_ = data['height_cm'].std()
        
        # Store parameters
        self._params = {
            'measurement_mean': self.measurement_mean_,
            'measurement_std': self.measurement_std_,
            'height_mean': self.height_mean_,
            'height_std': self.height_std_,
            'measurement_cols': self.measurement_cols,
            'normalization_method': self.normalization_method
        }
        
        self.is_fitted = True
        
        # Log statistics
        self.logger.info("✓ Measurement statistics:")
        for i, col in enumerate(self.measurement_cols):
            self.logger.info(
                f"  {col:20s}: mean={self.measurement_mean_[i]:6.2f}, "
                f"std={self.measurement_std_[i]:6.2f}"
            )
        
        if self.height_mean_ is not None:
            self.logger.info(
                f"✓ Height normalization: mean={self.height_mean_:.2f}, "
                f"std={self.height_std_:.2f}"
            )
        
        return self
    
    def transform(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Transform measurements to normalized values.
        
        Formula: z = (x - mean) / std
        
        Args:
            data: DataFrame or array with measurements
            
        Returns:
            Normalized measurements (same type as input)
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        
        is_dataframe = isinstance(data, pd.DataFrame)
        
        if is_dataframe:
            normalized = data.copy()
            measurements = data[self.measurement_cols].values
        else:
            measurements = data.copy()
        
        # Apply Z-score normalization
        normalized_measurements = (measurements - self.measurement_mean_) / self.measurement_std_
        
        if is_dataframe:
            normalized[self.measurement_cols] = normalized_measurements
            return normalized
        else:
            return normalized_measurements
    
    def inverse_transform(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Convert normalized measurements back to original scale (cm).
        
        CRITICAL: Use this to interpret model predictions!
        
        Formula: x = (z * std) + mean
        
        Args:
            data: Normalized measurements
            
        Returns:
            Original-scale measurements in centimeters
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before inverse_transform")
        
        is_dataframe = isinstance(data, pd.DataFrame)
        
        if is_dataframe:
            denormalized = data.copy()
            normalized_measurements = data[self.measurement_cols].values
        else:
            normalized_measurements = data.copy()
        
        # Reverse z-score normalization
        original_measurements = (normalized_measurements * self.measurement_std_) + self.measurement_mean_
        
        if is_dataframe:
            denormalized[self.measurement_cols] = original_measurements
            return denormalized
        else:
            return original_measurements
    
    def transform_height(self, height: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Normalize height value(s) for model input.
        
        Args:
            height: Height in cm (scalar or array)
            
        Returns:
            Normalized height
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform_height")
        
        if self.height_mean_ is None:
            raise ValueError("Height normalization not available")
        
        return (height - self.height_mean_) / self.height_std_
    
    def inverse_transform_height(self, normalized_height: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Convert normalized height back to centimeters.
        
        Args:
            normalized_height: Normalized height value(s)
            
        Returns:
            Height in cm
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before inverse_transform_height")
        
        if self.height_mean_ is None:
            raise ValueError("Height normalization not available")
        
        return (normalized_height * self.height_std_) + self.height_mean_
    
    def get_measurement_statistics(self) -> pd.DataFrame:
        """
        Get summary statistics for measurements.
        
        Returns:
            DataFrame with mean and std for each measurement
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor not fitted yet")
        
        stats = pd.DataFrame({
            'measurement': self.measurement_cols,
            'mean_cm': self.measurement_mean_,
            'std_cm': self.measurement_std_
        })
        
        return stats
    def load_params(self, filepath: str):
        """Load fitted parameters from disk."""
        import pickle
        with open(filepath, 'rb') as f:
            self._params = pickle.load(f)
        
        # Restore parameters to class attributes
        self.measurement_mean_ = self._params['measurement_mean']
        self.measurement_std_ = self._params['measurement_std']
        self.height_mean_ = self._params.get('height_mean')
        self.height_std_ = self._params.get('height_std')
        
        self.is_fitted = True



class MetadataPreprocessor:
    """
    Simple preprocessor for categorical metadata (gender).
    """
    
    def __init__(self):
        """Initialize metadata preprocessor."""
        self.gender_mapping = {'female': 0, 'male': 1}
        self.inverse_gender_mapping = {0: 'female', 1: 'male'}
    
    def encode_gender(self, gender: Union[str, pd.Series]) -> Union[int, np.ndarray]:
        """
        Encode gender to numeric values.
        
        Args:
            gender: Gender string or Series
            
        Returns:
            Encoded gender (0=female, 1=male)
        """
        if isinstance(gender, pd.Series):
            return gender.map(self.gender_mapping).values
        else:
            return self.gender_mapping.get(gender.lower(), -1)
    
    def decode_gender(self, encoded: Union[int, np.ndarray]) -> Union[str, np.ndarray]:
        """
        Decode numeric gender back to string.
        
        Args:
            encoded: Numeric gender code
            
        Returns:
            Gender string
        """
        if isinstance(encoded, (int, np.integer)):
            return self.inverse_gender_mapping.get(int(encoded), 'unknown')
        else:
            return np.array([self.inverse_gender_mapping.get(int(e), 'unknown') for e in encoded])


# Quick test
if __name__ == "__main__":
    from src.config.config import Config
    
    print("="*80)
    print("Testing MeasurementPreprocessor")
    print("="*80)
    
    config = Config()
    preprocessor = MeasurementPreprocessor(config)
    
    # Create dummy training data
    np.random.seed(42)
    n_samples = 100
    dummy_data = pd.DataFrame({
        col: np.random.normal(100, 15, n_samples)
        for col in config.measurement.MEASUREMENT_COLUMNS
    })
    dummy_data['height_cm'] = np.random.normal(165, 7, n_samples)
    
    print(f"\n📊 Created dummy data: {dummy_data.shape}")
    print(f"   Original chest mean: {dummy_data['chest'].mean():.2f} cm")
    
    # Fit on training data
    preprocessor.fit(dummy_data)
    
    # Transform (normalize)
    normalized = preprocessor.transform(dummy_data)
    print(f"\n✓ Normalized chest mean: {normalized['chest'].mean():.6f} (should be ~0)")
    print(f"✓ Normalized chest std: {normalized['chest'].std():.6f} (should be ~1)")
    
    # Inverse transform (denormalize)
    denormalized = preprocessor.inverse_transform(normalized)
    print(f"\n✓ Denormalized chest mean: {denormalized['chest'].mean():.2f} cm")
    
    # Check accuracy
    accuracy = np.allclose(dummy_data[config.measurement.MEASUREMENT_COLUMNS].values, 
                          denormalized[config.measurement.MEASUREMENT_COLUMNS].values)
    print(f"✓ Inverse transform accuracy: {accuracy}")
    
    # Test height normalization
    sample_height = 170.0
    normalized_height = preprocessor.transform_height(sample_height)
    denormalized_height = preprocessor.inverse_transform_height(normalized_height)
    print(f"\n✓ Height test:")
    print(f"  Original: {sample_height} cm")
    print(f"  Normalized: {normalized_height:.4f}")
    print(f"  Denormalized: {denormalized_height:.2f} cm")
    
    # Test metadata preprocessor
    print("\n" + "="*80)
    print("Testing MetadataPreprocessor")
    print("="*80)
    
    meta_preprocessor = MetadataPreprocessor()
    print(f"✓ 'female' → {meta_preprocessor.encode_gender('female')}")
    print(f"✓ 'male' → {meta_preprocessor.encode_gender('male')}")
    print(f"✓ 0 → '{meta_preprocessor.decode_gender(0)}'")
    print(f"✓ 1 → '{meta_preprocessor.decode_gender(1)}'")
    
    print("\n✅ Measurement preprocessor test completed successfully!")
