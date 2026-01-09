"""
base_classes.py - Abstract Base Classes for BMnet Preprocessing Pipeline

This module defines the abstract interfaces that all concrete implementations
must follow. This ensures consistency and allows easy extension.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
import pandas as pd


class BaseDataLoader(ABC):
    """
    Abstract base class for data loading operations.
    
    All data loaders must implement methods to load CSV files,
    validate data integrity, and merge datasets.
    """
    
    def __init__(self, config):
        """
        Initialize the data loader with configuration.
        
        Args:
            config: Configuration object containing paths and parameters
        """
        self.config = config
        self._data = {}  # Internal storage for loaded data
    
    @abstractmethod
    def load_csv_files(self) -> Dict[str, pd.DataFrame]:
        """Load all required CSV files."""
        pass
    
    @abstractmethod
    def validate_data(self) -> bool:
        """Validate data integrity."""
        pass
    
    @abstractmethod
    def merge_datasets(self) -> pd.DataFrame:
        """Merge all datasets into a single DataFrame."""
        pass
    
    def get_data(self, key: str) -> pd.DataFrame:
        """Retrieve loaded data by key."""
        if key not in self._data:
            raise KeyError(f"Dataset '{key}' not found. Available: {list(self._data.keys())}")
        return self._data[key]


class BasePreprocessor(ABC):
    """
    Abstract base class for preprocessing operations.
    
    All preprocessors must implement fit, transform, and fit_transform methods
    following scikit-learn conventions.
    """
    
    def __init__(self, config):
        """
        Initialize the preprocessor with configuration.
        
        Args:
            config: Configuration object containing preprocessing parameters
        """
        self.config = config
        self.is_fitted = False
        self._params = {}  # Store fitted parameters (means, stds, etc.)
    
    @abstractmethod
    def fit(self, data: Any) -> 'BasePreprocessor':
        """
        Learn preprocessing parameters from training data.
        
        Args:
            data: Training data to learn parameters from
            
        Returns:
            Self (for method chaining)
        """
        pass
    
    @abstractmethod
    def transform(self, data: Any) -> Any:
        """
        Apply preprocessing transformation to data.
        
        Args:
            data: Data to transform
            
        Returns:
            Transformed data
        """
        pass
    
    def fit_transform(self, data: Any) -> Any:
        """
        Fit and transform in one step (convenience method).
        
        Args:
            data: Data to fit and transform
            
        Returns:
            Transformed data
        """
        return self.fit(data).transform(data)
    
    def get_params(self) -> Dict[str, Any]:
        """Get fitted parameters."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor has not been fitted yet.")
        return self._params.copy()
    
    def save_params(self, filepath: str):
        """Save fitted parameters to disk."""
        import pickle
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted preprocessor.")
        with open(filepath, 'wb') as f:
            pickle.dump(self._params, f)
    
    def load_params(self, filepath: str):
        """Load fitted parameters from disk."""
        import pickle
        with open(filepath, 'rb') as f:
            self._params = pickle.load(f)
        self.is_fitted = True


class BaseDatasetBuilder(ABC):
    """
    Abstract base class for building train/val/test datasets.
    
    Handles splitting, stratification, and dataset creation.
    """
    
    def __init__(self, config):
        """
        Initialize the dataset builder with configuration.
        
        Args:
            config: Configuration object with split parameters
        """
        self.config = config
        self._splits = {}  # Store train/val/test splits
    
    @abstractmethod
    def create_splits(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create train/validation/test splits."""
        pass
    
    @abstractmethod
    def validate_splits(self) -> bool:
        """Validate that splits are correctly formed."""
        pass
    
    def get_split(self, split_name: str) -> pd.DataFrame:
        """Retrieve a specific split."""
        if split_name not in self._splits:
            raise KeyError(f"Split '{split_name}' not found. Available: {list(self._splits.keys())}")
        return self._splits[split_name]
    
    def get_split_statistics(self) -> Dict[str, Any]:
        """Get statistics about the splits."""
        stats = {}
        for name, df in self._splits.items():
            stats[name] = {
                'size': len(df),
                'percentage': len(df) / sum(len(s) for s in self._splits.values()) * 100
            }
        return stats


# Quick test
if __name__ == "__main__":
    print("✅ Base classes defined successfully!")
    print("\nAvailable abstract classes:")
    print("  - BaseDataLoader: For loading CSV and image data")
    print("  - BasePreprocessor: For data transformation (fit/transform)")
    print("  - BaseDatasetBuilder: For creating train/val/test splits")
