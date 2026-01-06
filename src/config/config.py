"""
config.py - Configuration Management for BMnet Body Measurement Model

This module defines all configuration parameters for the preprocessing pipeline.
Uses a class-based approach for easy modification and version control.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List


@dataclass
class DataConfig:
    """Configuration for data paths and file locations"""
    
    # Base paths
    BASE_DIR: Path = Path(".")
    DATA_DIR: Path = BASE_DIR / "data"
    
    # Raw data paths
    RAW_DIR: Path = DATA_DIR / "raw"
    MASK_DIR: Path = RAW_DIR / "train" / "mask"
    MASK_LEFT_DIR: Path = RAW_DIR / "train" / "mask_left"
    
    # CSV file paths
    HWG_METADATA_PATH: Path = RAW_DIR / "train" / "hwg_metadata.csv"
    MEASUREMENTS_PATH: Path = RAW_DIR / "train" / "measurements.csv"
    SUBJECT_PHOTO_MAP_PATH: Path = RAW_DIR / "train" / "subject_to_photo_map.csv"
    
    # Processed data paths
    INTERIM_DIR: Path = DATA_DIR / "interim"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    
    # Test data paths
    TEST_A_DIR: Path = RAW_DIR / "testA"
    TEST_B_DIR: Path = RAW_DIR / "testB"


@dataclass
class ImageConfig:
    """Configuration for image preprocessing"""
    
    # Original dimensions (720 x 960)
    ORIGINAL_WIDTH: int = 720
    ORIGINAL_HEIGHT: int = 960
    
    # Target dimensions (maintaining 3:4 aspect ratio)
    TARGET_WIDTH: int = 512
    TARGET_HEIGHT: int = 683
    TARGET_SIZE: Tuple[int, int] = (512, 683)
    
    # Image properties
    CHANNELS: int = 3  # RGB
    PIXEL_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)  # ImageNet means
    PIXEL_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)   # ImageNet stds
    
    # Processing options
    NORMALIZE: bool = True
    RESIZE_METHOD: str = "lanczos"
    
    # Image file format
    IMAGE_FORMAT: str = ".png"


@dataclass
class MeasurementConfig:
    """Configuration for body measurements"""
    
    # All 14 measurement columns (in cm)
    MEASUREMENT_COLUMNS: List[str] = None
    
    # Metadata columns
    METADATA_COLUMNS: List[str] = None
    
    # Normalization strategy
    NORMALIZATION_METHOD: str = "standard"  # Z-score normalization
    
    def __post_init__(self):
        """Initialize list attributes"""
        if self.MEASUREMENT_COLUMNS is None:
            self.MEASUREMENT_COLUMNS = [
                'ankle', 'arm-length', 'bicep', 'calf', 'chest', 
                'forearm', 'height', 'hip', 'leg-length', 
                'shoulder-breadth', 'shoulder-to-crotch', 
                'thigh', 'waist', 'wrist'
            ]
        
        if self.METADATA_COLUMNS is None:
            self.METADATA_COLUMNS = ['gender', 'height_cm', 'weight_kg']


@dataclass
class TrainingConfig:
    """Configuration for training/validation/test splits"""
    
    # Split ratios
    TRAIN_RATIO: float = 0.8
    VAL_RATIO: float = 0.2
    TEST_RATIO: float = 0.0  # Using separate testA and testB
    
    # Random seed for reproducibility
    RANDOM_SEED: int = 42
    
    # Stratification
    STRATIFY_BY: str = "gender"
    
    # Batch processing
    BATCH_SIZE: int = 8
    NUM_WORKERS: int = 1


@dataclass
class ModelConfig:
    """Configuration for model architecture"""
    
    # Input configuration
    NUM_VIEWS: int = 2  # Front (mask) + Side (mask_left)
    USE_HEIGHT: bool = True  # Use height_cm as additional input
    USE_WEIGHT: bool = False  # Weight not available during inference
    
    # Output configuration
    NUM_MEASUREMENTS: int = 14  # All 14 body measurements
    
    # Model type
    BACKBONE: str = "resnet50"  # ResNet-50 pretrained on ImageNet
    PRETRAINED: bool = True  # Use ImageNet pretrained weights


class Config:
    """
    Master configuration class that aggregates all config dataclasses.
    
    Usage:
        config = Config()
        print(config.image.TARGET_SIZE)
        print(config.data.MASK_DIR)
    """
    
    def __init__(self):
        self.data = DataConfig()
        self.image = ImageConfig()
        self.measurement = MeasurementConfig()
        self.training = TrainingConfig()
        self.model = ModelConfig()
    
    def __repr__(self):
        return (
            f"Config(\n"
            f"  Image Size: {self.image.TARGET_SIZE}\n"
            f"  Measurements: {self.model.NUM_MEASUREMENTS}\n"
            f"  Train/Val: {self.training.TRAIN_RATIO}/{self.training.VAL_RATIO}\n"
            f"  Backbone: {self.model.BACKBONE} (Pretrained: {self.model.PRETRAINED})\n"
            f")"
        )


# Quick test
if __name__ == "__main__":
    config = Config()
    print("✅ Configuration loaded successfully!")
    print(config)
    print(f"\n📐 Target image size: {config.image.TARGET_SIZE}")
    print(f"📊 Number of measurements: {len(config.measurement.MEASUREMENT_COLUMNS)}")
    print(f"🔀 Train/Val split: {config.training.TRAIN_RATIO}/{config.training.VAL_RATIO}")
    print(f"🧠 Backbone: {config.model.BACKBONE} (ImageNet pretrained: {config.model.PRETRAINED})")
