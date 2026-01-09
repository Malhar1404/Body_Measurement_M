"""
preprocessing_pipeline.py - Data preprocessing pipeline

Prepares data for hip and bust prediction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from resnet18.config.config import Config
from resnet18.features.image_preprocessor import ImagePreprocessor
from resnet18.features.measurement_preprocessor import MeasurementPreprocessor


def load_and_merge_data(config: Config) -> pd.DataFrame:
    """
    Load and merge all CSV files.
    """
    print("📂 Loading data...")
    
    # Load CSVs
    measurements = pd.read_csv(config.paths.MEASUREMENTS_CSV)
    metadata = pd.read_csv(config.paths.METADATA_CSV)
    photo_map = pd.read_csv(config.paths.PHOTO_MAP_CSV)
    
    print(f"  Measurements: {len(measurements)} rows")
    print(f"  Metadata: {len(metadata)} rows")
    print(f"  Photo map: {len(photo_map)} rows")
    
    # Merge
    data = photo_map.merge(metadata, on='subject_id', how='left')
    data = data.merge(measurements, on='subject_id', how='left')
    
    print(f"✓ Merged data: {len(data)} rows")
    
    return data


def filter_and_clean(data: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Filter and clean data.
    """
    print("\n🧹 Cleaning data...")
    
    initial_count = len(data)
    
    # Required columns
    required_cols = ['photo_id', 'subject_id', 'height_cm', 'weight_kg', 'gender'] + config.measurement.MEASUREMENT_COLUMNS
    
    # Drop rows with missing values in required columns
    data = data.dropna(subset=required_cols)
    
    print(f"  Removed {initial_count - len(data)} rows with missing values")
    
    # Remove outliers (height between 140-200 cm)
    data = data[(data['height_cm'] >= 140) & (data['height_cm'] <= 200)]
    
    # Check if images exist
    valid_photos = []
    for idx, row in data.iterrows():
        photo_id = row['photo_id']
        mask_path = config.paths.MASK_DIR / f"{photo_id}.png"
        mask_left_path = config.paths.MASK_LEFT_DIR / f"{photo_id}.png"
        
        if mask_path.exists() and mask_left_path.exists():
            valid_photos.append(idx)
    
    data = data.loc[valid_photos]
    
    print(f"✓ Clean data: {len(data)} rows")
    print(f"  Unique subjects: {data['subject_id'].nunique()}")
    
    return data


def split_data(data: pd.DataFrame, config: Config):
    """
    Split data into train and validation sets by subject.
    """
    print("\n✂️ Splitting data...")
    
    # Get unique subjects
    unique_subjects = data['subject_id'].unique()
    
    # Split subjects
    train_subjects, val_subjects = train_test_split(
        unique_subjects,
        test_size=config.training.VAL_RATIO,
        random_state=config.training.RANDOM_SEED,
        shuffle=True
    )
    
    # Create splits
    train_data = data[data['subject_id'].isin(train_subjects)].copy()
    val_data = data[data['subject_id'].isin(val_subjects)].copy()
    
    print(f"✓ Train: {len(train_data)} photos, {len(train_subjects)} subjects")
    print(f"✓ Val:   {len(val_data)} photos, {len(val_subjects)} subjects")
    
    return train_data, val_data


def main():
    """Main preprocessing pipeline."""
    print("="*80)
    print("PREPROCESSING PIPELINE - HIP & BUST PREDICTION")
    print("="*80)
    
    config = Config()
    
    # Load data
    data = load_and_merge_data(config)
    
    # Clean data
    data = filter_and_clean(data, config)
    
    # Split data
    train_data, val_data = split_data(data, config)
    
    # Initialize preprocessors
    print("\n🔧 Fitting preprocessors...")
    
    image_preprocessor = ImagePreprocessor(config)
    image_preprocessor.fit()
    
    measurement_preprocessor = MeasurementPreprocessor(config)
    measurement_preprocessor.fit(train_data)
    
    # Save preprocessors
    image_preprocessor.save_params(
        str(config.paths.PROCESSED_DIR / 'image_preprocessor_params.pkl')
    )
    measurement_preprocessor.save_params(
        str(config.paths.PROCESSED_DIR / 'measurement_preprocessor_params.pkl')
    )
    
    # Save splits
    train_data.to_csv(config.paths.PROCESSED_DIR / 'train_data.csv', index=False)
    val_data.to_csv(config.paths.PROCESSED_DIR / 'val_data.csv', index=False)
    
    print(f"\n✅ Preprocessing complete!")
    print(f"📁 Saved to: {config.paths.PROCESSED_DIR}")
    print(f"  - train_data.csv ({len(train_data)} rows)")
    print(f"  - val_data.csv ({len(val_data)} rows)")
    print(f"  - image_preprocessor_params.pkl")
    print(f"  - measurement_preprocessor_params.pkl")
    
    # Print statistics
    print(f"\n📊 Measurement Statistics:")
    for col in config.measurement.MEASUREMENT_COLUMNS:
        print(f"  {col:10s}: Mean={train_data[col].mean():.2f} cm, Std={train_data[col].std():.2f} cm")


if __name__ == "__main__":
    main()
