"""
dataset.py - PyTorch Dataset for BMnet Body Measurement

This module creates a PyTorch Dataset that loads silhouette images and measurements
for training the body measurement prediction model.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
from src.config.config import Config
from src.features.image_preprocessor import ImagePreprocessor
from src.features.measurement_preprocessor import MeasurementPreprocessor


class BMnetDataset(Dataset):
    """
    PyTorch Dataset for BMnet body measurement data.
    
    Returns:
        - mask_image: Front view silhouette (512×683×3)
        - mask_left_image: Side view silhouette (512×683×3)
        - height: Normalized height value
        - measurements: 14 normalized body measurements
        - photo_id: Photo identifier (for debugging)
    """
    
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_preprocessor: ImagePreprocessor,
        measurement_preprocessor: MeasurementPreprocessor,
        config: Config,
        augment: bool = False
    ):
        """
        Initialize dataset.
        
        Args:
            dataframe: DataFrame with photo_id, subject_id, measurements, metadata
            image_preprocessor: Fitted image preprocessor
            measurement_preprocessor: Fitted measurement preprocessor
            config: Configuration object
            augment: Whether to apply data augmentation (for training only)
        """
        self.df = dataframe.reset_index(drop=True)
        self.image_preprocessor = image_preprocessor
        self.measurement_preprocessor = measurement_preprocessor
        self.config = config
        self.augment = augment
        
        self.measurement_cols = config.measurement.MEASUREMENT_COLUMNS
        self.mask_dir = config.data.MASK_DIR
        self.mask_left_dir = config.data.MASK_LEFT_DIR
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with:
            - 'mask': Front view image tensor (3, 683, 512)
            - 'mask_left': Side view image tensor (3, 683, 512)
            - 'height': Normalized height (scalar)
            - 'measurements': Target measurements (14,)
            - 'photo_id': Photo identifier (string)
        """
        # Get row data
        row = self.df.iloc[idx]
        photo_id = row['photo_id']
        
        # Load and preprocess images
        mask_path = self.mask_dir / f"{photo_id}.png"
        mask_left_path = self.mask_left_dir / f"{photo_id}.png"
        
        try:
            # Transform images (returns numpy arrays)
            mask_img = self.image_preprocessor.transform(mask_path)
            mask_left_img = self.image_preprocessor.transform(mask_left_path)
            
            # Convert to torch tensors and transpose to (C, H, W)
            mask_tensor = torch.from_numpy(mask_img).permute(2, 0, 1).float()
            mask_left_tensor = torch.from_numpy(mask_left_img).permute(2, 0, 1).float()
            
        except Exception as e:
            print(f"Error loading images for {photo_id}: {e}")
            # Return zeros as fallback
            mask_tensor = torch.zeros(3, 683, 512)
            mask_left_tensor = torch.zeros(3, 683, 512)
        
        # Get height (normalized)
        height = row['height_cm']
        height_normalized = self.measurement_preprocessor.transform_height(height)
        height_tensor = torch.tensor(height_normalized, dtype=torch.float32)
        
        # Get measurements (normalized)
        measurements = row[self.measurement_cols].values.astype(np.float32)
        measurements_normalized = self.measurement_preprocessor.transform(measurements)
        measurements_tensor = torch.from_numpy(measurements_normalized).float()
        
        return {
            'mask': mask_tensor,
            'mask_left': mask_left_tensor,
            'height': height_tensor,
            'measurements': measurements_tensor,
            'photo_id': photo_id
        }


def create_dataloaders(config: Config) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Load processed data
    processed_dir = config.data.PROCESSED_DIR
    train_df = pd.read_csv(processed_dir / 'train_data.csv')
    val_df = pd.read_csv(processed_dir / 'val_data.csv')
    
    print(f"✓ Loaded train data: {len(train_df)} samples")
    print(f"✓ Loaded val data: {len(val_df)} samples")
    
    # Initialize preprocessors
    image_preprocessor = ImagePreprocessor(config)
    image_preprocessor.fit()
    
    measurement_preprocessor = MeasurementPreprocessor(config)
    # Load fitted parameters from training
    measurement_preprocessor.load_params(str(processed_dir / 'measurement_preprocessor_params.pkl'))
    
    print("✓ Loaded preprocessor parameters")
    
    # Create datasets
    train_dataset = BMnetDataset(
        train_df,
        image_preprocessor,
        measurement_preprocessor,
        config,
        augment=False  # Set to True if you want augmentation
    )
    
    val_dataset = BMnetDataset(
        val_df,
        image_preprocessor,
        measurement_preprocessor,
        config,
        augment=False
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.training.BATCH_SIZE,
        shuffle=True,
        num_workers=config.training.NUM_WORKERS,
        pin_memory=True,  # Faster GPU transfer
        drop_last=True    # Drop incomplete batches
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.training.BATCH_SIZE,
        shuffle=False,
        num_workers=config.training.NUM_WORKERS,
        pin_memory=True
    )
    
    print(f"✓ Created dataloaders:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    return train_loader, val_loader


# Quick test
if __name__ == "__main__":
    from config.config import Config
    
    print("="*80)
    print("Testing BMnetDataset")
    print("="*80)
    
    config = Config()
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)
    
    # Test loading one batch
    print("\n" + "="*80)
    print("Testing Data Loading")
    print("="*80)
    
    batch = next(iter(train_loader))
    
    print(f"\n✓ Batch loaded successfully!")
    print(f"  mask shape: {batch['mask'].shape}")
    print(f"  mask_left shape: {batch['mask_left'].shape}")
    print(f"  height shape: {batch['height'].shape}")
    print(f"  measurements shape: {batch['measurements'].shape}")
    print(f"  photo_ids: {batch['photo_id'][:3]}...")
    
    print(f"\n✓ Value ranges:")
    print(f"  mask: [{batch['mask'].min():.2f}, {batch['mask'].max():.2f}]")
    print(f"  height: [{batch['height'].min():.2f}, {batch['height'].max():.2f}]")
    print(f"  measurements: [{batch['measurements'].min():.2f}, {batch['measurements'].max():.2f}]")
    
    print("\n✅ Dataset test completed successfully!")
