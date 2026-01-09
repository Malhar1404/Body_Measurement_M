"""
dataset.py - PyTorch Dataset for Hip & Bust prediction

Handles 1-channel grayscale silhouettes.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
import albumentations as A

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from resnet18.config.config import Config


class HipBustDataset(Dataset):
    """
    Dataset for hip and bust prediction.
    """
    
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_preprocessor,
        measurement_preprocessor,
        config: Config,
        augment: bool = False
    ):
        self.df = dataframe.reset_index(drop=True)
        self.image_preprocessor = image_preprocessor
        self.measurement_preprocessor = measurement_preprocessor
        self.config = config
        self.augment = augment
        
        self.measurement_cols = config.measurement.MEASUREMENT_COLUMNS
        self.mask_dir = config.paths.MASK_DIR
        self.mask_left_dir = config.paths.MASK_LEFT_DIR
        
        # Augmentation pipeline for grayscale
        if self.augment:
            self.augment_transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.1,
                    rotate_limit=5,
                    border_mode=0,
                    p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,
                    contrast_limit=0.1,
                    p=0.3
                ),
                A.GaussNoise(var_limit=(5.0, 15.0), p=0.2),
            ])
            print(f"✓ Augmentation enabled")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        photo_id = row['photo_id']
        
        # Load images
        mask_path = self.mask_dir / f"{photo_id}.png"
        mask_left_path = self.mask_left_dir / f"{photo_id}.png"
        
        try:
            # Transform images (returns H, W, 1)
            mask_img = self.image_preprocessor.transform(mask_path)
            mask_left_img = self.image_preprocessor.transform(mask_left_path)
            
            # Apply augmentation if enabled
            if self.augment:
                # Denormalize for augmentation
                mask_img_uint8 = ((mask_img * self.image_preprocessor.std + self.image_preprocessor.mean) * 255).astype(np.uint8)
                mask_left_img_uint8 = ((mask_left_img * self.image_preprocessor.std + self.image_preprocessor.mean) * 255).astype(np.uint8)
                
                # Apply augmentation
                augmented_mask = self.augment_transform(image=mask_img_uint8)['image']
                augmented_left = self.augment_transform(image=mask_left_img_uint8)['image']
                
                # Re-normalize
                mask_img = (augmented_mask.astype(np.float32) / 255.0 - self.image_preprocessor.mean) / self.image_preprocessor.std
                mask_left_img = (augmented_left.astype(np.float32) / 255.0 - self.image_preprocessor.mean) / self.image_preprocessor.std
            
            # Convert to tensors (1, H, W)
            mask_tensor = torch.from_numpy(mask_img).permute(2, 0, 1).float()
            mask_left_tensor = torch.from_numpy(mask_left_img).permute(2, 0, 1).float()
            
        except Exception as e:
            print(f"Error loading {photo_id}: {e}")
            # Fallback: zeros
            mask_tensor = torch.zeros(1, self.config.image.TARGET_HEIGHT, 
                                     self.config.image.TARGET_WIDTH)
            mask_left_tensor = torch.zeros(1, self.config.image.TARGET_HEIGHT,
                                          self.config.image.TARGET_WIDTH)
        
        # Get height
        height = row['height_cm']
        height_normalized = self.measurement_preprocessor.transform_height(height)
        height_tensor = torch.tensor(height_normalized, dtype=torch.float32)
        
        # Get measurements (only hip and bust)
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


def create_dataloaders(config: Config, image_preprocessor, measurement_preprocessor,
                       train_df: pd.DataFrame, val_df: pd.DataFrame):
    """
    Create train and validation dataloaders.
    
    Args:
        config: Configuration
        image_preprocessor: Fitted image preprocessor
        measurement_preprocessor: Fitted measurement preprocessor
        train_df: Training dataframe
        val_df: Validation dataframe
    
    Returns:
        train_loader, val_loader
    """
    # Create datasets
    train_dataset = HipBustDataset(
        train_df,
        image_preprocessor,
        measurement_preprocessor,
        config,
        augment=config.training.USE_AUGMENTATION
    )
    
    val_dataset = HipBustDataset(
        val_df,
        image_preprocessor,
        measurement_preprocessor,
        config,
        augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.BATCH_SIZE,
        shuffle=True,
        num_workers=config.training.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.BATCH_SIZE,
        shuffle=False,
        num_workers=config.training.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"✓ Created dataloaders")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    return train_loader, val_loader
