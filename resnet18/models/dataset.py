"""
dataset.py - Dataset with Strong Augmentation
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2


class BodyMeasurementDataset(Dataset):
    """Dataset with augmentation for body measurement prediction."""
    
    def __init__(
        self, 
        dataframe,
        image_preprocessor,
        measurement_preprocessor,
        image_dir,
        is_training=True
    ):
        self.df = dataframe.reset_index(drop=True)
        self.image_preprocessor = image_preprocessor
        self.measurement_preprocessor = measurement_preprocessor
        self.image_dir = Path(image_dir)
        self.is_training = is_training
        
        # ===== AUGMENTATION FOR TRAINING =====
        if is_training:
            self.transform = A.Compose([
                # Geometric augmentations
                A.ShiftScaleRotate(
                    shift_limit=0.05,      # 5% shift
                    scale_limit=0.1,       # ±10% scale
                    rotate_limit=10,       # ±10 degrees
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.7
                ),
                A.HorizontalFlip(p=0.5),   # 50% flip (body symmetry)
                
                # Perspective & distortion
                A.ElasticTransform(
                    alpha=30,
                    sigma=5,
                    alpha_affine=5,
                    p=0.3
                ),
                A.GridDistortion(
                    num_steps=5,
                    distort_limit=0.1,
                    p=0.3
                ),
                
                # Pixel-level augmentations
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
                A.GaussianBlur(blur_limit=(3, 5), p=0.3),
                
                # Cutout (mask random regions)
                A.CoarseDropout(
                    max_holes=8,
                    max_height=16,
                    max_width=16,
                    min_holes=4,
                    min_height=8,
                    min_width=8,
                    fill_value=0,
                    p=0.3
                ),
            ], p=1.0)
        else:
            # No augmentation for validation
            self.transform = None
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load images
        mask_path = self.image_dir / row['mask_path']
        mask_left_path = self.image_dir / row['mask_left_path']
        
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask_left = cv2.imread(str(mask_left_path), cv2.IMREAD_GRAYSCALE)
        
        if mask is None or mask_left is None:
            raise FileNotFoundError(f"Image not found: {mask_path} or {mask_left_path}")
        
        # Apply augmentation (training only)
        if self.transform is not None:
            # Apply same transform to both images
            transformed = self.transform(image=mask)
            mask = transformed['image']
            
            transformed_left = self.transform(image=mask_left)
            mask_left = transformed_left['image']
        
        # Resize
        target_size = (224, 224)
        mask = cv2.resize(mask, target_size)
        mask_left = cv2.resize(mask_left, target_size)
        
        # Normalize to [0, 1]
        mask = mask.astype(np.float32) / 255.0
        mask_left = mask_left.astype(np.float32) / 255.0
        
        # Add channel dimension
        mask = np.expand_dims(mask, axis=0)
        mask_left = np.expand_dims(mask_left, axis=0)
        
        # Convert to tensors
        mask = torch.from_numpy(mask)
        mask_left = torch.from_numpy(mask_left)
        
        # Get measurements
        height = torch.tensor([row['height']], dtype=torch.float32)
        measurements = torch.tensor([row['hip'], row['bust']], dtype=torch.float32)
        
        return {
            'mask': mask,
            'mask_left': mask_left,
            'height': height,
            'measurements': measurements
        }


def create_dataloaders(config, image_preprocessor, measurement_preprocessor, train_df, val_df):
    """Create train and validation dataloaders with augmentation."""
    
    train_dataset = BodyMeasurementDataset(
        dataframe=train_df,
        image_preprocessor=image_preprocessor,
        measurement_preprocessor=measurement_preprocessor,
        image_dir=config.paths.PROCESSED_DIR / 'masks',
        is_training=True  # Enable augmentation
    )
    
    val_dataset = BodyMeasurementDataset(
        dataframe=val_df,
        image_preprocessor=image_preprocessor,
        measurement_preprocessor=measurement_preprocessor,
        image_dir=config.paths.PROCESSED_DIR / 'masks',
        is_training=False  # No augmentation
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.BATCH_SIZE,
        shuffle=True,
        num_workers=config.training.NUM_WORKERS,
        pin_memory=True,
        drop_last=True  # Drop incomplete batches
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.BATCH_SIZE,
        shuffle=False,
        num_workers=config.training.NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, val_loader
