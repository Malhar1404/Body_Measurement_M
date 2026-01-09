"""
config.py - Configuration for Hip & Bust Prediction

Folder structure:
/home/ubuntu/Body_Measurement_M/
├── data/
│   └── train/
│       ├── mask/
│       ├── mask_left/
│       └── *.csv files
└── resnet18/
    └── config/
        └── config.py
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List
import os


@dataclass
class PathConfig:
    """File paths configuration"""
    
    # Get absolute path to project root
    # Method 1: Using __file__ (2 levels up from config.py)
    # resnet18/config/config.py -> resnet18/ -> project_root/
    CONFIG_DIR: Path = Path(__file__).parent  # resnet18/config/
    RESNET18_DIR: Path = CONFIG_DIR.parent    # resnet18/
    PROJECT_ROOT: Path = RESNET18_DIR.parent  # project_root/
    
    # Alternative method using environment variable (more reliable)
    # PROJECT_ROOT: Path = Path(os.environ.get('PROJECT_ROOT', Path(__file__).parent.parent.parent))
    
    # Data directories
    DATA_DIR: Path = PROJECT_ROOT / "data"
    TRAIN_DIR: Path = DATA_DIR / "raw" / "train"
    PROCESSED_DIR: Path = DATA_DIR / "processed_resnet18"
    
    # Image directories
    MASK_DIR: Path = TRAIN_DIR / 'mask'
    MASK_LEFT_DIR: Path = TRAIN_DIR / 'mask_left'

    # CSV files
    MEASUREMENTS_CSV: Path = TRAIN_DIR / "measurements.csv"
    METADATA_CSV: Path = TRAIN_DIR / "hwg_metadata.csv"
    PHOTO_MAP_CSV: Path = TRAIN_DIR / "subject_to_photo_map.csv"
    
    # For backward compatibility
    @property
    def RAW_DIR(self):
        return self.TRAIN_DIR
    
    def __post_init__(self):
        """Create directories if they don't exist and verify paths"""
        
        # Print debug info
        print(f"\n{'='*70}")
        print(f"PATH RESOLUTION DEBUG")
        print(f"{'='*70}")
        print(f"__file__:        {Path(__file__).absolute()}")
        print(f"CONFIG_DIR:      {self.CONFIG_DIR.absolute()}")
        print(f"RESNET18_DIR:    {self.RESNET18_DIR.absolute()}")
        print(f"PROJECT_ROOT:    {self.PROJECT_ROOT.absolute()}")
        print(f"DATA_DIR:        {self.DATA_DIR.absolute()}")
        print(f"TRAIN_DIR:       {self.TRAIN_DIR.absolute()}")
        print(f"{'='*70}")
        
        # Create processed directory
        self.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created/verified: {self.PROCESSED_DIR}")
        
        # Verify critical paths exist
        print(f"\n{'='*70}")
        print(f"VERIFYING PATHS")
        print(f"{'='*70}")
        
        checks = [
            ("Data directory", self.DATA_DIR),
            ("Train directory", self.TRAIN_DIR),
            ("Mask directory", self.MASK_DIR),
            ("Mask_left directory", self.MASK_LEFT_DIR),
            ("Measurements CSV", self.MEASUREMENTS_CSV),
            ("Metadata CSV", self.METADATA_CSV),
            ("Photo map CSV", self.PHOTO_MAP_CSV),
        ]
        
        errors = []
        for name, path in checks:
            exists = path.exists()
            status = "✓" if exists else "❌"
            print(f"{status} {name:25s}: {path}")
            if not exists:
                errors.append(f"❌ {name} not found: {path}")
        
        print(f"{'='*70}")
        
        if errors:
            print(f"\n⚠️  ERRORS FOUND:")
            for error in errors:
                print(f"  {error}")
            
            # Print helpful message
            print(f"\n💡 TROUBLESHOOTING:")
            print(f"1. Check if data folder exists:")
            print(f"   ls {self.DATA_DIR}")
            print(f"\n2. Check if train folder exists:")
            print(f"   ls {self.TRAIN_DIR}")
            print(f"\n3. List data folder contents:")
            print(f"   ls -la {self.DATA_DIR}")
            
            raise FileNotFoundError("\n".join(errors))
        
        print(f"\n✅ All required paths exist!\n")


@dataclass
class ImageConfig:
    """Image preprocessing configuration"""
    
    # Original dimensions (720 x 960)
    ORIGINAL_WIDTH: int = 720
    ORIGINAL_HEIGHT: int = 960
    
    # Target dimensions
    TARGET_WIDTH: int = 720
    TARGET_HEIGHT: int = 960
    TARGET_SIZE: Tuple[int, int] = (720, 960)
    
    # Channels
    CHANNELS: int = 1
    
    # Normalization
    PIXEL_MEAN: Tuple[float] = (0.5,)
    PIXEL_STD: Tuple[float] = (0.5,)
    
    # Processing
    NORMALIZE: bool = True
    RESIZE_METHOD: str = "lanczos"
    IMAGE_FORMAT: str = ".png"


@dataclass
class MeasurementConfig:
    """Measurement configuration"""
    
    MEASUREMENT_COLUMNS: List[str] = None
    NUM_MEASUREMENTS: int = 2
    
    def __post_init__(self):
        if self.MEASUREMENT_COLUMNS is None:
            self.MEASUREMENT_COLUMNS = ["hip", "chest"]


@dataclass
class TrainingConfig:
    """Training configuration"""
    
    TRAIN_RATIO: float = 0.8
    VAL_RATIO: float = 0.2
    RANDOM_SEED: int = 60
    STRATIFY_BY: str = "gender"
    
    BATCH_SIZE: int = 24
    NUM_WORKERS: int = 0
    
    USE_AUGMENTATION: bool = True
    AUGMENTATION_PROBABILITY: float = 0.3
    
    INITIAL_LR: float = 5e-5
    WEIGHT_DECAY: float = 1e-5
    MAX_EPOCHS: int = 100
    
    EARLY_STOP_PATIENCE: int = 20
    EARLY_STOP_MIN_DELTA: float = 0.001
    
    LR_SCHEDULER_PATIENCE: int = 7
    LR_SCHEDULER_FACTOR: float = 0.5


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    
    BACKBONE: str = "resnet18"
    PRETRAINED: bool = True
    
    NUM_VIEWS: int = 2
    USE_HEIGHT: bool = True
    USE_WEIGHT: bool = False
    
    NUM_MEASUREMENTS: int = 2


class Config:
    """Master configuration class"""
    
    def __init__(self):
        self.paths = PathConfig()
        self.image = ImageConfig()
        self.measurement = MeasurementConfig()
        self.training = TrainingConfig()
        self.model = ModelConfig()
        
        # Backward compatibility
        self.data = self.paths
    
    def __repr__(self):
        return (
            f"\n{'='*60}\n"
            f"CONFIG SUMMARY\n"
            f"{'='*60}\n"
            f"Model:\n"
            f"  Backbone: {self.model.BACKBONE}\n"
            f"  Pretrained: {self.model.PRETRAINED}\n"
            f"\n"
            f"Image:\n"
            f"  Size: {self.image.TARGET_SIZE}\n"
            f"  Channels: {self.image.CHANNELS}\n"
            f"\n"
            f"Training:\n"
            f"  Batch size: {self.training.BATCH_SIZE}\n"
            f"  Max epochs: {self.training.MAX_EPOCHS}\n"
            f"\n"
            f"Measurements:\n"
            f"  {', '.join(self.measurement.MEASUREMENT_COLUMNS)}\n"
            f"{'='*60}\n"
        )


if __name__ == "__main__":
    print("Testing configuration...")
    try:
        config = Config()
        print(config)
        print("✅ Configuration test PASSED!")
    except Exception as e:
        print(f"\n❌ Configuration test FAILED!")
        print(f"Error: {e}")
