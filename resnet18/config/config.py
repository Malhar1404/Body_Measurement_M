"""
config.py - Configuration for Hip & Bust Prediction

Optimized for:
- ResNet-18 backbone
- 1-channel grayscale silhouettes
- 640×853 resolution (3:4 ratio)
- 2 outputs: hip, bust
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List


@dataclass
class PathConfig:
    """File paths configuration"""
    
    # Project root
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    
    # Data directories
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DIR: Path = DATA_DIR / "processed_resnet18"  # Separate folder
    
    # Image directories
    MASK_DIR: Path = RAW_DIR / "mask"
    MASK_LEFT_DIR: Path = RAW_DIR / "mask_left"
    
    # CSV files
    MEASUREMENTS_CSV: Path = RAW_DIR / "measurements.csv"
    METADATA_CSV: Path = RAW_DIR / "hwg_metadata.csv"
    PHOTO_MAP_CSV: Path = RAW_DIR / "subject_to_photo_map.csv"
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        self.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ImageConfig:
    """Image preprocessing configuration"""
    
    # Original dimensions (720 x 960)
    ORIGINAL_WIDTH: int = 720
    ORIGINAL_HEIGHT: int = 960
    
    # Target dimensions (640 x 853 for 3:4 ratio)
    TARGET_WIDTH: int = 720
    TARGET_HEIGHT: int = 960
    TARGET_SIZE: Tuple[int, int] = (720, 960)
    
    # Channels (1 for grayscale silhouettes)
    CHANNELS: int = 1
    
    # Normalization (for grayscale)
    PIXEL_MEAN: Tuple[float] = (0.5,)
    PIXEL_STD: Tuple[float] = (0.5,)
    
    # Processing
    NORMALIZE: bool = True
    RESIZE_METHOD: str = "lanczos"
    IMAGE_FORMAT: str = ".png"


@dataclass
class MeasurementConfig:
    """Measurement configuration"""
    
    # Only hip and bust!
    MEASUREMENT_COLUMNS: List[str] = None
    NUM_MEASUREMENTS: int = 2
    
    def __post_init__(self):
        if self.MEASUREMENT_COLUMNS is None:
            self.MEASUREMENT_COLUMNS = [
                "hip",
                "chest"  # "chest" in dataset = "bust" measurement
            ]


@dataclass
class TrainingConfig:
    """Training configuration"""
    
    # Data splits
    TRAIN_RATIO: float = 0.8
    VAL_RATIO: float = 0.2
    RANDOM_SEED: int = 42
    STRATIFY_BY: str = "gender"
    
    # Batch processing (larger batch size due to 1-channel)
    BATCH_SIZE: int = 48  # Can use 48-64 with 1-channel!
    NUM_WORKERS: int = 0
    
    # Augmentation
    USE_AUGMENTATION: bool = False
    AUGMENTATION_PROBABILITY: float = 0.3
    
    # Training hyperparameters
    INITIAL_LR: float = 1e-4
    WEIGHT_DECAY: float = 1e-3
    MAX_EPOCHS: int = 100
    
    # Early stopping
    EARLY_STOP_PATIENCE: int = 20
    EARLY_STOP_MIN_DELTA: float = 0.001
    
    # Scheduler
    LR_SCHEDULER_PATIENCE: int = 7
    LR_SCHEDULER_FACTOR: float = 0.5


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    
    # Backbone
    BACKBONE: str = "resnet18"
    PRETRAINED: bool = True
    
    # Input configuration
    NUM_VIEWS: int = 2
    USE_HEIGHT: bool = True
    USE_WEIGHT: bool = False
    
    # Output configuration
    NUM_MEASUREMENTS: int = 2  # Only hip and bust


class Config:
    """Master configuration class"""
    
    def __init__(self):
        self.paths = PathConfig()
        self.image = ImageConfig()
        self.measurement = MeasurementConfig()
        self.training = TrainingConfig()
        self.model = ModelConfig()
        
        # Create data alias for backward compatibility
        self.data = self.paths
    
    def __repr__(self):
        return (
            f"Config(\n"
            f"  Backbone: {self.model.BACKBONE}\n"
            f"  Image size: {self.image.TARGET_SIZE}\n"
            f"  Channels: {self.image.CHANNELS}\n"
            f"  Batch size: {self.training.BATCH_SIZE}\n"
            f"  Measurements: {self.measurement.MEASUREMENT_COLUMNS}\n"
            f")"
        )
