"""
data_loader.py - Concrete Data Loader Implementation for BMnet Dataset

This module implements the BaseDataLoader interface to handle loading,
validation, and merging of BMnet CSV files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
import logging
import sys

# Add parent directory to path to import base_classes
sys.path.append(str(Path(__file__).parent.parent))
from utils.base_classes import BaseDataLoader

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class BMnetDataLoader(BaseDataLoader):
    """
    Concrete implementation of data loader for BMnet body measurement dataset.
    
    Handles loading of:
    - hwg_metadata.csv (subject demographics)
    - measurements.csv (14 body measurements)
    - subject_to_photo_map.csv (subject-to-photo mapping)
    
    Validates data integrity and merges into unified DataFrame.
    """
    
    def __init__(self, config):
        """Initialize BMnet data loader."""
        super().__init__(config)
        self.logger = logger
        self.merged_data = None
    
    def load_csv_files(self) -> Dict[str, pd.DataFrame]:
        """Load all three CSV files from the BMnet dataset."""
        self.logger.info("Loading CSV files...")
        
        try:
            # Load metadata (subject_id, gender, height_cm, weight_kg)
            self._data['metadata'] = pd.read_csv(self.config.data.HWG_METADATA_PATH)
            self.logger.info(f"✓ Loaded metadata: {self._data['metadata'].shape}")
            
            # Load measurements (subject_id + 14 measurements)
            self._data['measurements'] = pd.read_csv(self.config.data.MEASUREMENTS_PATH)
            self.logger.info(f"✓ Loaded measurements: {self._data['measurements'].shape}")
            
            # Load subject-to-photo mapping
            self._data['photo_map'] = pd.read_csv(self.config.data.SUBJECT_PHOTO_MAP_PATH)
            self.logger.info(f"✓ Loaded photo map: {self._data['photo_map'].shape}")
            
            return self._data
            
        except FileNotFoundError as e:
            self.logger.error(f"❌ File not found: {e}")
            raise
        except Exception as e:
            self.logger.error(f"❌ Error loading CSV files: {e}")
            raise
    
    def validate_data(self) -> bool:
        """Comprehensive data validation checks."""
        self.logger.info("Validating data integrity...")
        
        if not self._data:
            raise ValueError("No data loaded. Call load_csv_files() first.")
        
        metadata = self._data['metadata']
        measurements = self._data['measurements']
        photo_map = self._data['photo_map']
        
        # 1. Check for missing values
        self.logger.info("Checking for missing values...")
        for name, df in self._data.items():
            missing = df.isnull().sum()
            if missing.any():
                self.logger.warning(f"⚠️  Missing values in {name}:\n{missing[missing > 0]}")
            else:
                self.logger.info(f"✓ No missing values in {name}")
        
        # 2. Check for duplicate subject_ids and remove them
        self.logger.info("Checking for duplicates...")
        for name in ['metadata', 'measurements']:
            df = self._data[name]
            duplicates = df['subject_id'].duplicated().sum()
            if duplicates > 0:
                self.logger.warning(f"⚠️  Found {duplicates} duplicate subject_ids in {name}")
                # Show which subjects are duplicated
                dup_subjects = df[df['subject_id'].duplicated(keep=False)]['subject_id'].unique()
                self.logger.warning(f"    Duplicate subjects: {list(dup_subjects)[:5]}")
                
                # Remove duplicates - keep first occurrence
                self._data[name] = df.drop_duplicates(subset='subject_id', keep='first')
                self.logger.warning(f"    Removed duplicates, new shape: {self._data[name].shape}")
                
                # Update local reference
                if name == 'metadata':
                    metadata = self._data['metadata']
                elif name == 'measurements':
                    measurements = self._data['measurements']
            else:
                self.logger.info(f"✓ No duplicate subject_ids in {name}")
        
        # 3. Check subject consistency across files
        self.logger.info("Checking subject consistency...")
        metadata_subjects = set(metadata['subject_id'])
        measurement_subjects = set(measurements['subject_id'])
        photo_subjects = set(photo_map['subject_id'])
        
        common_subjects = metadata_subjects & measurement_subjects & photo_subjects
        
        self.logger.info(f"  Subjects in metadata: {len(metadata_subjects)}")
        self.logger.info(f"  Subjects in measurements: {len(measurement_subjects)}")
        self.logger.info(f"  Subjects in photo_map: {len(photo_subjects)}")
        self.logger.info(f"  Common subjects: {len(common_subjects)}")
        
        # 4. Validate measurement ranges (anthropometric constraints)
        self.logger.info("Validating measurement ranges...")
        measurement_cols = self.config.measurement.MEASUREMENT_COLUMNS
        
        ranges = {
            'height': (140, 190), 'ankle': (18, 35), 'arm-length': (35, 60),
            'bicep': (15, 60), 'calf': (25, 60), 'chest': (70, 160),
            'forearm': (15, 45), 'hip': (75, 165), 'leg-length': (60, 100),
            'shoulder-breadth': (25, 45), 'shoulder-to-crotch': (45, 75),
            'thigh': (35, 90), 'waist': (55, 150), 'wrist': (12, 25)
        }
        
        outliers_found = False
        for col in measurement_cols:
            if col in ranges:
                min_val, max_val = ranges[col]
                out_of_range = ((measurements[col] < min_val) | 
                            (measurements[col] > max_val)).sum()
                if out_of_range > 0:
                    self.logger.warning(
                        f"⚠️  {col}: {out_of_range} values outside range [{min_val}, {max_val}]"
                    )
                    outliers_found = True
        
        if not outliers_found:
            self.logger.info("✓ All measurements within reasonable ranges")
        
        self.logger.info("✅ Data validation completed!")
        return True

    
    def merge_datasets(self) -> pd.DataFrame:
        """
        Merge all datasets into a single unified DataFrame.
        
        Strategy:
        1. Start with photo_map (one row per photo)
        2. Merge with measurements (one subject → multiple photos)
        3. Merge with metadata
        
        Returns:
            Merged DataFrame where each row = one photo with all data
        """
        self.logger.info("Merging datasets...")
        
        if not self._data:
            raise ValueError("No data loaded. Call load_csv_files() first.")
        
        photo_map = self._data['photo_map'].copy()
        measurements = self._data['measurements'].copy()
        metadata = self._data['metadata'].copy()
        
        # Start with photo_map as base
        merged = photo_map.copy()
        self.logger.info(f"Starting with photo_map: {merged.shape}")
        
        # Merge with measurements
        merged = merged.merge(measurements, on='subject_id', how='left')
        self.logger.info(f"After merging measurements: {merged.shape}")
        
        # Merge with metadata
        merged = merged.merge(metadata, on='subject_id', how='left')
        self.logger.info(f"After merging metadata: {merged.shape}")
        
        # Store merged data
        self.merged_data = merged
        self._data['merged'] = merged
        
        self.logger.info(f"✅ Successfully merged datasets: {merged.shape}")
        self.logger.info(f"   Columns: {list(merged.columns)}")
        
        return merged
    
    def get_photos_per_subject(self) -> pd.Series:
        """Get count of photos per subject."""
        if 'photo_map' not in self._data:
            raise ValueError("Photo map not loaded")
        return self._data['photo_map'].groupby('subject_id').size()
    
    def summary(self) -> Dict:
        """Get summary statistics of loaded data."""
        if not self._data:
            return {"status": "No data loaded"}
        
        summary = {
            'files_loaded': list(self._data.keys()),
            'num_subjects': {
                'metadata': self._data['metadata']['subject_id'].nunique() if 'metadata' in self._data else 0,
                'measurements': self._data['measurements']['subject_id'].nunique() if 'measurements' in self._data else 0,
                'photo_map': self._data['photo_map']['subject_id'].nunique() if 'photo_map' in self._data else 0,
            },
            'num_photos': len(self._data['photo_map']) if 'photo_map' in self._data else 0,
        }
        
        if 'photo_map' in self._data:
            photos_per_subject = self.get_photos_per_subject()
            summary['photos_per_subject'] = {
                'min': int(photos_per_subject.min()),
                'max': int(photos_per_subject.max()),
                'mean': float(photos_per_subject.mean()),
                'median': float(photos_per_subject.median())
            }
        
        if self.merged_data is not None:
            summary['merged_shape'] = self.merged_data.shape
        
        return summary


# Quick test
if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent / 'config'))
    from src.config.config import Config
    
    print("="*80)
    print("Testing BMnetDataLoader")
    print("="*80)
    
    config = Config()
    loader = BMnetDataLoader(config)
    
    # Load data
    loader.load_csv_files()
    
    # Validate
    loader.validate_data()
    
    # Merge
    merged_df = loader.merge_datasets()
    
    # Get summary
    summary = loader.summary()
    print("\n" + "="*80)
    print("DATA LOADER SUMMARY")
    print("="*80)
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print("\n✅ Data loader test completed successfully!")
