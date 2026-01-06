"""
preprocessing_pipeline.py - Master Preprocessing Pipeline for BMnet

This pipeline orchestrates the complete data preprocessing workflow:
1. Load and validate CSV data
2. Create train/validation splits
3. Fit preprocessors on training data
4. Transform all data
5. Save processed datasets and parameters
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import pickle
import logging
import sys
from sklearn.model_selection import train_test_split

from src.config.config import Config
from src.data.data_loader import BMnetDataLoader
from src.features.image_preprocessor import ImagePreprocessor
from src.features.measurement_preprocessor import MeasurementPreprocessor

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class BMnetPreprocessingPipeline:
    """
    Complete preprocessing pipeline for BMnet body measurement dataset.
    
    Usage:
        pipeline = BMnetPreprocessingPipeline()
        pipeline.run()
    """
    
    def __init__(self, config: Config = None):
        """Initialize the pipeline."""
        self.config = config if config else Config()
        self.data_loader = None
        self.image_preprocessor = None
        self.measurement_preprocessor = None
        self.train_df = None
        self.val_df = None
        self.logger = logger
    
    def run(self, save_outputs: bool = True) -> Dict:
        """
        Execute the complete preprocessing pipeline.
        
        Args:
            save_outputs: Whether to save processed data and parameters
            
        Returns:
            Dictionary with pipeline results
        """
        self.logger.info("="*80)
        self.logger.info("STARTING BMNET PREPROCESSING PIPELINE")
        self.logger.info("="*80)
        
        # Step 1: Load and merge data
        merged_data = self._load_data()
        
        # Step 2: Create train/val splits
        self._create_splits(merged_data)
        
        # Step 3: Fit preprocessors on training data
        self._fit_preprocessors()
        
        # Step 4: Save outputs
        if save_outputs:
            self._save_outputs()
        
        # Step 5: Get statistics
        results = self._get_statistics()
        
        self.logger.info("="*80)
        self.logger.info("✅ PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
        self.logger.info("="*80)
        
        return results
    
    def _load_data(self) -> pd.DataFrame:
        """Load and validate all CSV data."""
        self.logger.info("\n" + "="*80)
        self.logger.info("STEP 1: Loading Data")
        self.logger.info("="*80)
        
        # Initialize data loader
        self.data_loader = BMnetDataLoader(self.config)
        
        # Load CSV files
        self.data_loader.load_csv_files()
        
        # Validate data
        self.data_loader.validate_data()
        
        # Merge datasets
        merged_data = self.data_loader.merge_datasets()
        
        self.logger.info(f"\n✓ Total samples (photos): {len(merged_data)}")
        self.logger.info(f"✓ Total unique subjects: {merged_data['subject_id'].nunique()}")
        
        return merged_data
    
    def _create_splits(self, data: pd.DataFrame):
        """Create train/validation splits by subject."""
        self.logger.info("\n" + "="*80)
        self.logger.info("STEP 2: Creating Train/Val Splits")
        self.logger.info("="*80)
        
        # Get unique subjects
        unique_subjects = data['subject_id'].unique()
        
        # Split subjects (not photos!) into train/val
        train_subjects, val_subjects = train_test_split(
            unique_subjects,
            test_size=self.config.training.VAL_RATIO,
            random_state=self.config.training.RANDOM_SEED,
            shuffle=True
        )
        
        # Get all photos for train and val subjects
        self.train_df = data[data['subject_id'].isin(train_subjects)].copy()
        self.val_df = data[data['subject_id'].isin(val_subjects)].copy()
        
        self.logger.info(f"\n✓ Train subjects: {len(train_subjects)}")
        self.logger.info(f"✓ Train photos: {len(self.train_df)}")
        self.logger.info(f"✓ Val subjects: {len(val_subjects)}")
        self.logger.info(f"✓ Val photos: {len(self.val_df)}")
        self.logger.info(f"✓ Split ratio: {len(self.train_df)/(len(self.train_df)+len(self.val_df)):.2%} / "
                        f"{len(self.val_df)/(len(self.train_df)+len(self.val_df)):.2%}")
    
    def _fit_preprocessors(self):
        """Fit preprocessors on training data ONLY."""
        self.logger.info("\n" + "="*80)
        self.logger.info("STEP 3: Fitting Preprocessors")
        self.logger.info("="*80)
        
        # Initialize preprocessors
        self.image_preprocessor = ImagePreprocessor(self.config)
        self.measurement_preprocessor = MeasurementPreprocessor(self.config)
        
        # Fit image preprocessor (no learning needed for ImageNet stats)
        self.logger.info("\nFitting image preprocessor...")
        self.image_preprocessor.fit()
        
        # Fit measurement preprocessor on TRAINING data only
        self.logger.info("\nFitting measurement preprocessor...")
        self.measurement_preprocessor.fit(self.train_df)
        
        self.logger.info("\n✓ All preprocessors fitted successfully!")
    
    def _save_outputs(self):
        """Save processed data and fitted parameters."""
        self.logger.info("\n" + "="*80)
        self.logger.info("STEP 4: Saving Outputs")
        self.logger.info("="*80)
        
        # Create output directories
        processed_dir = self.config.data.PROCESSED_DIR
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Save train/val dataframes
        train_path = processed_dir / 'train_data.csv'
        val_path = processed_dir / 'val_data.csv'
        
        self.train_df.to_csv(train_path, index=False)
        self.val_df.to_csv(val_path, index=False)
        
        self.logger.info(f"✓ Saved train data: {train_path}")
        self.logger.info(f"✓ Saved val data: {val_path}")
        
        # Save measurement preprocessor parameters
        params_path = processed_dir / 'measurement_preprocessor_params.pkl'
        self.measurement_preprocessor.save_params(str(params_path))
        self.logger.info(f"✓ Saved preprocessor parameters: {params_path}")
        
        # Save measurement statistics
        stats_path = processed_dir / 'measurement_statistics.csv'
        stats = self.measurement_preprocessor.get_measurement_statistics()
        stats.to_csv(stats_path, index=False)
        self.logger.info(f"✓ Saved measurement statistics: {stats_path}")
        
        # Save split information
        split_info = {
            'train_subjects': self.train_df['subject_id'].unique().tolist(),
            'val_subjects': self.val_df['subject_id'].unique().tolist(),
            'train_photos': len(self.train_df),
            'val_photos': len(self.val_df),
            'random_seed': self.config.training.RANDOM_SEED
        }
        split_info_path = processed_dir / 'split_info.pkl'
        with open(split_info_path, 'wb') as f:
            pickle.dump(split_info, f)
        self.logger.info(f"✓ Saved split info: {split_info_path}")
    
    def _get_statistics(self) -> Dict:
        """Get pipeline statistics."""
        stats = {
            'total_subjects': self.train_df['subject_id'].nunique() + self.val_df['subject_id'].nunique(),
            'train': {
                'subjects': self.train_df['subject_id'].nunique(),
                'photos': len(self.train_df),
                'photos_per_subject': len(self.train_df) / self.train_df['subject_id'].nunique()
            },
            'val': {
                'subjects': self.val_df['subject_id'].nunique(),
                'photos': len(self.val_df),
                'photos_per_subject': len(self.val_df) / self.val_df['subject_id'].nunique()
            },
            'measurements': {
                'num_measurements': len(self.config.measurement.MEASUREMENT_COLUMNS),
                'columns': self.config.measurement.MEASUREMENT_COLUMNS
            },
            'images': {
                'target_size': self.config.image.TARGET_SIZE,
                'normalization': 'ImageNet',
                'num_views': self.config.model.NUM_VIEWS
            }
        }
        
        return stats


def main():
    """Main function to run the pipeline."""
    print("\n" + "="*80)
    print("BMNET BODY MEASUREMENT - PREPROCESSING PIPELINE")
    print("="*80)
    
    # Initialize and run pipeline
    pipeline = BMnetPreprocessingPipeline()
    results = pipeline.run(save_outputs=True)
    
    # Print summary
    print("\n" + "="*80)
    print("PIPELINE SUMMARY")
    print("="*80)
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total subjects: {results['total_subjects']}")
    print(f"   Train subjects: {results['train']['subjects']} ({results['train']['photos']} photos)")
    print(f"   Val subjects: {results['val']['subjects']} ({results['val']['photos']} photos)")
    
    print(f"\n📐 Image Configuration:")
    print(f"   Target size: {results['images']['target_size']}")
    print(f"   Normalization: {results['images']['normalization']}")
    print(f"   Views per sample: {results['images']['num_views']}")
    
    print(f"\n📏 Measurements:")
    print(f"   Number of measurements: {results['measurements']['num_measurements']}")
    print(f"   Measurements: {', '.join(results['measurements']['columns'][:5])}...")
    
    print("\n" + "="*80)
    print("✅ PREPROCESSING COMPLETE - READY FOR MODEL TRAINING!")
    print("="*80)
    print("\nNext steps:")
    print("1. Check data/processed/ directory for outputs")
    print("2. Review measurement_statistics.csv for normalization params")
    print("3. Start model training with processed data")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
