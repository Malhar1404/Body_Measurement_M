from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List


@dataclass
class PathConfig:
    """File paths configuration"""
    
    # Project root (go up 2 levels from config.py)
    # resnet18/config/config.py -> resnet18/ -> project_root/
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    
    # Data directories
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DIR: Path = DATA_DIR / "train"  # ← Changed from "raw" to "train"
    PROCESSED_DIR: Path = DATA_DIR / "processed_resnet18"  # Separate folder
    
    # Image directories (inside train folder)
    MASK_DIR: Path = RAW_DIR / "mask"
    MASK_LEFT_DIR: Path = RAW_DIR / "mask_left"
    
    # CSV files (inside train folder)
    MEASUREMENTS_CSV: Path = RAW_DIR / "measurements.csv"
    METADATA_CSV: Path = RAW_DIR / "hwg_metadata.csv"
    PHOTO_MAP_CSV: Path = RAW_DIR / "subject_to_photo_map.csv"
    
    def __post_init__(self):
        """Create directories if they don't exist and verify paths"""
        # Create processed directory
        self.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        
        # Print paths for verification
        print(f"\n📁 Path Configuration:")
        print(f"  Project Root: {self.PROJECT_ROOT.absolute()}")
        print(f"  Data Dir: {self.DATA_DIR.absolute()}")
        print(f"  Train Dir: {self.RAW_DIR.absolute()}")
        print(f"  Processed Dir: {self.PROCESSED_DIR.absolute()}")
        
        # Verify critical paths exist
        errors = []
        
        if not self.DATA_DIR.exists():
            errors.append(f"❌ Data directory not found: {self.DATA_DIR}")
        
        if not self.RAW_DIR.exists():
            errors.append(f"❌ Train directory not found: {self.RAW_DIR}")
        
        if not self.MASK_DIR.exists():
            errors.append(f"❌ Mask directory not found: {self.MASK_DIR}")
        
        if not self.MASK_LEFT_DIR.exists():
            errors.append(f"❌ Mask_left directory not found: {self.MASK_LEFT_DIR}")
        
        if not self.MEASUREMENTS_CSV.exists():
            errors.append(f"❌ Measurements CSV not found: {self.MEASUREMENTS_CSV}")
        
        if not self.METADATA_CSV.exists():
            errors.append(f"❌ Metadata CSV not found: {self.METADATA_CSV}")
        
        if not self.PHOTO_MAP_CSV.exists():
            errors.append(f"❌ Photo map CSV not found: {self.PHOTO_MAP_CSV}")
        
        if errors:
            print("\n⚠️  Path Verification Errors:")
            for error in errors:
                print(f"  {error}")
            raise FileNotFoundError("\n".join(errors))
        
        print(f"✓ All required paths exist!")


@dataclass
class ImageConfig:
    """Image preprocessing configuration"""
    
    # Original dimensions (720 x 960)
    ORIGINAL_WIDTH: int = 720
    ORIGINAL_HEIGHT: int = 960
    
    # Target dimensions (keep original size)
    TARGET_WIDTH: int = 720
    TARGET_HEIGHT: int = 960
    TARGET_SIZE: Tuple[int, int] = (720, 960)  # (W, H)
    
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
    
    # Batch processing
    BATCH_SIZE: int = 32  # Adjust based on your GPU RAM
    NUM_WORKERS: int = 0  # Windows: 0, Linux/Mac: 4
    
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
            f"\n{'='*60}\n"
            f"CONFIG SUMMARY\n"
            f"{'='*60}\n"
            f"Model:\n"
            f"  Backbone: {self.model.BACKBONE}\n"
            f"  Pretrained: {self.model.PRETRAINED}\n"
            f"  Use height: {self.model.USE_HEIGHT}\n"
            f"\n"
            f"Image:\n"
            f"  Size: {self.image.TARGET_SIZE}\n"
            f"  Channels: {self.image.CHANNELS}\n"
            f"\n"
            f"Training:\n"
            f"  Batch size: {self.training.BATCH_SIZE}\n"
            f"  Learning rate: {self.training.INITIAL_LR}\n"
            f"  Max epochs: {self.training.MAX_EPOCHS}\n"
            f"  Augmentation: {self.training.USE_AUGMENTATION}\n"
            f"\n"
            f"Measurements:\n"
            f"  {', '.join(self.measurement.MEASUREMENT_COLUMNS)}\n"
            f"{'='*60}\n"
        )


def test_config():
    """Test configuration and verify paths."""
    print("="*60)
    print("TESTING CONFIGURATION")
    print("="*60)
    
    try:
        config = Config()
        print(config)
        
        # Test path existence
        print("\n📂 Verifying data files...")
        
        files_to_check = [
            ("Measurements CSV", config.paths.MEASUREMENTS_CSV),
            ("Metadata CSV", config.paths.METADATA_CSV),
            ("Photo Map CSV", config.paths.PHOTO_MAP_CSV),
        ]
        
        for name, path in files_to_check:
            exists = path.exists()
            status = "✓" if exists else "❌"
            print(f"  {status} {name}: {path.name}")
        
        # Check image directories
        print(f"\n📂 Checking image directories...")
        
        mask_count = len(list(config.paths.MASK_DIR.glob("*.png")))
        mask_left_count = len(list(config.paths.MASK_LEFT_DIR.glob("*.png")))
        
        print(f"  ✓ Mask images: {mask_count}")
        print(f"  ✓ Mask_left images: {mask_left_count}")
        
        print(f"\n{'='*60}")
        print(f"✅ CONFIGURATION TEST PASSED!")
        print(f"{'='*60}\n")
        return True
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"❌ CONFIGURATION TEST FAILED!")
        print(f"{'='*60}")
        print(f"\nError: {e}\n")
        
        print(f"Expected folder structure:")
        print(f"project_root/")
        print(f"├── data/")
        print(f"│   └── train/              ← Your data is here")
        print(f"│       ├── mask/")
        print(f"│       ├── mask_left/")
        print(f"│       ├── measurements.csv")
        print(f"│       ├── hwg_metadata.csv")
        print(f"│       └── subject_to_photo_map.csv")
        print(f"└── resnet18/")
        print(f"    └── config/")
        print(f"        └── config.py\n")
        
        return False


if __name__ == "__main__":
    test_config()