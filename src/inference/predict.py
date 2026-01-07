"""
predict.py - Inference Script for Body Measurement Prediction

Features:
- Load trained model from checkpoint
- Predict from single image pair or batch
- Denormalize predictions to centimeters
- Visualize predictions
- Export results to CSV/Excel
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from typing import Dict, Union
import matplotlib.pyplot as plt
import seaborn as sns

from src.config.config import Config
from src.models.body_measurement_model import create_model
from src.features.image_preprocessor import ImagePreprocessor
from src.features.measurement_preprocessor import MeasurementPreprocessor


class BodyMeasurementPredictor:
    """
    Predictor class for body measurement inference.
    """
    
    def __init__(self, checkpoint_path: str, config: Config = None):
        """
        Initialize predictor.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            config: Configuration object (optional)
        """
        self.config = config if config else Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🚀 Using device: {self.device}")
        print(f"📂 Loading model from: {checkpoint_path}")
        
        # Create model
        self.model = create_model(self.config, model_type='resnet50', pretrained=False)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded from epoch {checkpoint['epoch']}")
        print(f"✓ Validation loss: {checkpoint.get('val_loss', 'N/A')}")
        
        # Initialize preprocessors
        self.image_preprocessor = ImagePreprocessor(self.config)
        self.image_preprocessor.fit()
        
        self.measurement_preprocessor = MeasurementPreprocessor(self.config)
        # Load fitted parameters
        params_path = self.config.data.PROCESSED_DIR / 'measurement_preprocessor_params.pkl'
        if params_path.exists():
            self.measurement_preprocessor.load_params(str(params_path))
            print("✓ Loaded measurement normalization parameters")
        else:
            print("⚠️  Warning: Measurement normalization parameters not found")
        
        self.measurement_names = self.config.measurement.MEASUREMENT_COLUMNS
    
    def predict_single(self, mask_path: Union[str, Path], 
                      mask_left_path: Union[str, Path],
                      height_cm: float) -> Dict[str, float]:
        """
        Predict body measurements for a single subject.
        
        Args:
            mask_path: Path to front view silhouette
            mask_left_path: Path to side view silhouette
            height_cm: Height in centimeters
            
        Returns:
            Dictionary with predicted measurements in centimeters
        """
        with torch.no_grad():
            # Preprocess images
            mask_img = self.image_preprocessor.transform(mask_path)
            mask_left_img = self.image_preprocessor.transform(mask_left_path)
            
            # Convert to tensors
            mask_tensor = torch.from_numpy(mask_img).permute(2, 0, 1).unsqueeze(0).float()
            mask_left_tensor = torch.from_numpy(mask_left_img).permute(2, 0, 1).unsqueeze(0).float()
            
            # Normalize height
            height_normalized = self.measurement_preprocessor.transform_height(height_cm)
            height_tensor = torch.tensor([height_normalized], dtype=torch.float32)
            
            # Move to device
            mask_tensor = mask_tensor.to(self.device)
            mask_left_tensor = mask_left_tensor.to(self.device)
            height_tensor = height_tensor.to(self.device)
            
            # Predict
            predictions = self.model(mask_tensor, mask_left_tensor, height_tensor)
            
            # Denormalize predictions
            predictions_np = predictions.cpu().numpy()
            predictions_cm = self.measurement_preprocessor.inverse_transform(predictions_np)[0]
            
            # Create result dictionary
            results = {
                'height_cm': height_cm,
                **{name: float(pred) for name, pred in zip(self.measurement_names, predictions_cm)}
            }
            
            return results
    
    def predict_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict body measurements for batch of subjects.
        
        Args:
            data: DataFrame with columns: photo_id, height_cm
                  Images should be in mask/ and mask_left/ directories
            
        Returns:
            DataFrame with predictions
        """
        results = []
        
        print(f"🔮 Predicting for {len(data)} samples...")
        
        for idx, row in data.iterrows():
            photo_id = row['photo_id']
            height = row['height_cm']
            
            # Get image paths
            mask_path = self.config.data.MASK_DIR / f"{photo_id}.png"
            mask_left_path = self.config.data.MASK_LEFT_DIR / f"{photo_id}.png"
            
            if not mask_path.exists() or not mask_left_path.exists():
                print(f"⚠️  Skipping {photo_id}: Images not found")
                continue
            
            try:
                # Predict
                predictions = self.predict_single(mask_path, mask_left_path, height)
                predictions['photo_id'] = photo_id
                results.append(predictions)
                
                if (idx + 1) % 100 == 0:
                    print(f"  Processed {idx + 1}/{len(data)} samples")
                    
            except Exception as e:
                print(f"❌ Error processing {photo_id}: {e}")
                continue
        
        results_df = pd.DataFrame(results)
        print(f"✓ Predictions complete: {len(results_df)} samples")
        
        return results_df
    
    def predict_and_compare(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict and compare with ground truth measurements.
        
        Args:
            data: DataFrame with photo_id, height_cm, and ground truth measurements
            
        Returns:
            DataFrame with predictions, ground truth, and errors
        """
        # Get predictions
        predictions_df = self.predict_batch(data)
        
        # Merge with ground truth
        merged = predictions_df.merge(
            data[['photo_id'] + self.measurement_names],
            on='photo_id',
            suffixes=('_pred', '_true')
        )
        
        # Calculate errors
        for measurement in self.measurement_names:
            pred_col = f"{measurement}_pred"
            true_col = f"{measurement}_true"
            error_col = f"{measurement}_error"
            
            if pred_col in merged.columns and true_col in merged.columns:
                merged[error_col] = merged[pred_col] - merged[true_col]
                merged[f"{measurement}_abs_error"] = merged[error_col].abs()
        
        return merged
    
    def visualize_predictions(self, photo_id: str, predictions: Dict[str, float],
                            ground_truth: Dict[str, float] = None,
                            save_path: str = None):
        """
        Visualize predictions vs ground truth.
        
        Args:
            photo_id: Photo identifier
            predictions: Predicted measurements
            ground_truth: True measurements (optional)
            save_path: Path to save plot (optional)
        """
        measurements = [m for m in self.measurement_names if m in predictions]
        pred_values = [predictions[m] for m in measurements]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(measurements))
        width = 0.35
        
        if ground_truth:
            true_values = [ground_truth.get(m, 0) for m in measurements]
            
            ax.bar(x - width/2, pred_values, width, label='Predicted', alpha=0.8)
            ax.bar(x + width/2, true_values, width, label='Ground Truth', alpha=0.8)
            
            # Calculate errors
            errors = [abs(p - t) for p, t in zip(pred_values, true_values)]
            mae = np.mean(errors)
            ax.set_title(f'Predictions vs Ground Truth - {photo_id}\nMAE: {mae:.2f} cm', 
                        fontsize=14, fontweight='bold')
        else:
            ax.bar(x, pred_values, width, alpha=0.8)
            ax.set_title(f'Predicted Measurements - {photo_id}', 
                        fontsize=14, fontweight='bold')
        
        ax.set_xlabel('Measurement', fontsize=12)
        ax.set_ylabel('Value (cm)', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('-', ' ').title() for m in measurements], 
                          rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved plot: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def export_results(self, results_df: pd.DataFrame, output_path: Union[str, Path]):
        """
        Export results to Excel.
        
        Args:
            results_df: DataFrame with results
            output_path: Output file path (.xlsx or .csv)
        """
        output_path = Path(output_path)
        
        if output_path.suffix == '.xlsx':
            results_df.to_excel(output_path, index=False)
        else:
            results_df.to_csv(output_path, index=False)
        
        print(f"✓ Results exported to: {output_path}")


def main():
    """Main function with usage examples."""
    print("="*80)
    print("BODY MEASUREMENT PREDICTION")
    print("="*80)
    
    # Initialize predictor
    config = Config()
    predictor = BodyMeasurementPredictor(
        checkpoint_path='checkpoints/best_model.pth',
        config=config
    )
    
    # Example 1: Single prediction
    print("\n" + "="*80)
    print("Example 1: Single Prediction")
    print("="*80)
    
    # You need to provide actual paths to test images
    mask_path = config.data.MASK_DIR / "sample_image_001.png"
    mask_left_path = config.data.MASK_LEFT_DIR / "sample_image_001.png"
    height = 170.0  # cm
    
    if mask_path.exists() and mask_left_path.exists():
        predictions = predictor.predict_single(mask_path, mask_left_path, height)
        
        print(f"\n📏 Predictions for height {height} cm:")
        for measurement, value in predictions.items():
            if measurement != 'height_cm':
                print(f"  {measurement:20s}: {value:6.2f} cm")
    else:
        print("⚠️  Sample images not found. Skipping single prediction example.")
    
    # Example 2: Batch prediction on validation set
    print("\n" + "="*80)
    print("Example 2: Batch Prediction on Validation Set")
    print("="*80)
    
    val_data_path = config.data.PROCESSED_DIR / 'val_data.csv'
    
    if val_data_path.exists():
        val_data = pd.read_csv(val_data_path)
        
        # Predict on first 10 samples
        sample_data = val_data.head(10)
        
        # Get predictions and compare
        results = predictor.predict_and_compare(sample_data)
        
        # Calculate overall metrics
        error_cols = [col for col in results.columns if col.endswith('_abs_error')]
        
        print("\n📊 Error Statistics (cm):")
        for col in error_cols:
            measurement = col.replace('_abs_error', '')
            mae = results[col].mean()
            print(f"  {measurement:20s}: MAE = {mae:6.2f} cm")
        
        # Export results
        output_dir = Path('predictions')
        output_dir.mkdir(exist_ok=True)
        
        predictor.export_results(results, output_dir / 'validation_predictions.xlsx')
        
        # Visualize first sample
        if len(results) > 0:
            first_sample = results.iloc[0]
            photo_id = first_sample['photo_id']
            
            predictions = {m: first_sample[f"{m}_pred"] for m in config.measurement.MEASUREMENT_COLUMNS}
            ground_truth = {m: first_sample[f"{m}_true"] for m in config.measurement.MEASUREMENT_COLUMNS}
            
            predictor.visualize_predictions(
                photo_id=photo_id,
                predictions=predictions,
                ground_truth=ground_truth,
                save_path=output_dir / f'prediction_{photo_id}.png'
            )
    else:
        print("⚠️  Validation data not found.")
    
    print("\n" + "="*80)
    print("✅ PREDICTION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
