"""
predict.py - Inference for Hip & Bust Prediction
Works with YOUR trained HipBustModel
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from resnet18.config.config import Config
from resnet18.models.model import create_model
from resnet18.features.image_preprocessor import ImagePreprocessor
from resnet18.features.measurement_preprocessor import MeasurementPreprocessor


class HipBustPredictor:
    """Predictor for hip and bust measurements."""
    
    def __init__(self, checkpoint_path: str = None, config: Config = None):
        self.config = config if config else Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🚀 Device: {self.device}")
        
        # Find checkpoint
        if checkpoint_path is None:
            checkpoint_path = self._find_best_checkpoint()
        else:
            if not Path(checkpoint_path).exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        
        # Load model
        self.model = create_model(self.config)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"\n✓ Model loaded")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Val Loss: {checkpoint.get('best_val_loss', 'N/A')}")
        
        # Load preprocessors
        print(f"\n📂 Loading preprocessors...")
        self.image_preprocessor = ImagePreprocessor(self.config)
        self.image_preprocessor.load_params(
            str(self.config.paths.PROCESSED_DIR / 'image_preprocessor_params.pkl')
        )
        
        self.measurement_preprocessor = MeasurementPreprocessor(self.config)
        self.measurement_preprocessor.load_params(
            str(self.config.paths.PROCESSED_DIR / 'measurement_preprocessor_params.pkl')
        )
        
        print("✓ Preprocessors loaded\n")
    
    def _find_best_checkpoint(self):
        """Find best checkpoint."""
        checkpoint_dir = Path('checkpoints_resnet18')
        
        for pattern in ['best*.pth', 'last_checkpoint.pth']:
            checkpoints = list(checkpoint_dir.glob(pattern))
            if checkpoints:
                return str(sorted(checkpoints)[-1])
        
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    
    def _find_mask_paths(self, row):
        """Find mask paths."""
        photo_id = row.get('photo_id', row.get('image_id', None))
        
        # Option 1: From CSV
        if 'mask_path' in row and 'mask_left_path' in row:
            mask_front = Path('data/raw/train') / row['mask_path']
            mask_side = Path('data/raw/train') / row['mask_left_path']
            
            if mask_front.exists() and mask_side.exists():
                return mask_front, mask_side
        
        # Option 2: Direct construction
        if photo_id:
            mask_front = Path('data/raw/train/mask') / f"{photo_id}.png"
            mask_side = Path('data/raw/train/mask_left') / f"{photo_id}.png"
            
            if mask_front.exists() and mask_side.exists():
                return mask_front, mask_side
        
        return None, None
    
    @torch.no_grad()
    def predict_single(self, mask_front_path, mask_side_path, height_cm: float) -> Dict[str, float]:
        """
        Predict for single image.
        
        EXACTLY matches your training pipeline in dataset.py:
        1. Load grayscale
        2. Resize to 224x224
        3. Normalize to [0, 1]
        4. Add channel dimension
        5. Convert to tensor
        """
        # Load images (grayscale)
        mask_front = cv2.imread(str(mask_front_path), cv2.IMREAD_GRAYSCALE)
        mask_side = cv2.imread(str(mask_side_path), cv2.IMREAD_GRAYSCALE)
        
        if mask_front is None or mask_side is None:
            raise FileNotFoundError(f"Failed to load images")
        
        # Resize to 224x224
        mask_front = cv2.resize(mask_front, (224, 224))
        mask_side = cv2.resize(mask_side, (224, 224))
        
        # Normalize to [0, 1]
        mask_front = mask_front.astype(np.float32) / 255.0
        mask_side = mask_side.astype(np.float32) / 255.0
        
        # Add channel dimension: (H, W) -> (1, H, W)
        mask_front = np.expand_dims(mask_front, axis=0)
        mask_side = np.expand_dims(mask_side, axis=0)
        
        # Convert to tensors: (1, H, W) -> (1, 1, H, W)
        mask_front_tensor = torch.from_numpy(mask_front).unsqueeze(0).float()
        mask_side_tensor = torch.from_numpy(mask_side).unsqueeze(0).float()
        
        # ===== HEIGHT: Match training exactly =====
        # Training does: torch.tensor([row['height']]) -> shape (1,)
        # DataLoader doesn't batch single values, keeps as (B,)
        # So for inference with batch_size=1: shape should be (1,)
        
        height_normalized = self.measurement_preprocessor.transform_height(height_cm)
        height_tensor = torch.tensor([height_normalized], dtype=torch.float32)  # Shape: (1,)
        
        # Move to device
        mask_front_tensor = mask_front_tensor.to(self.device)
        mask_side_tensor = mask_side_tensor.to(self.device)
        height_tensor = height_tensor.to(self.device)
        
        # Predict
        # Your model expects: (mask_front, mask_side, height)
        # mask_front: (B, 1, 224, 224)
        # mask_side: (B, 1, 224, 224)
        # height: (B,)
        predictions = self.model(mask_front_tensor, mask_side_tensor, height_tensor)
        
        # Denormalize
        predictions_np = predictions.cpu().numpy()
        predictions_cm = self.measurement_preprocessor.inverse_transform(predictions_np)[0]
        
        return {
            'height_cm': height_cm,
            'hip_predicted': float(predictions_cm[0]),
            'bust_predicted': float(predictions_cm[1])
        }
    
    @torch.no_grad()
    def predict_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict for batch."""
        results = []
        
        print(f"🔮 Predicting for {len(data)} samples...")
        
        for idx, row in tqdm(data.iterrows(), total=len(data), desc="Predicting"):
            photo_id = row.get('photo_id', row.get('image_id', idx))
            height = row['height']
            actual_hip = row.get('hip', None)
            actual_bust = row.get('bust', None)
            
            # Find mask paths
            mask_front_path, mask_side_path = self._find_mask_paths(row)
            
            if mask_front_path is None or mask_side_path is None:
                print(f"⚠️  Skipping {photo_id}: Masks not found")
                continue
            
            try:
                # Predict
                preds = self.predict_single(mask_front_path, mask_side_path, height)
                
                # Store results
                result = {
                    'photo_id': photo_id,
                    'height_cm': height,
                    'hip_actual': actual_hip,
                    'hip_predicted': preds['hip_predicted'],
                    'bust_actual': actual_bust,
                    'bust_predicted': preds['bust_predicted'],
                }
                
                # Calculate errors if ground truth available
                if actual_hip is not None and not np.isnan(actual_hip):
                    result['hip_error_cm'] = abs(preds['hip_predicted'] - actual_hip)
                    result['hip_error_pct'] = (abs(preds['hip_predicted'] - actual_hip) / actual_hip) * 100
                else:
                    result['hip_error_cm'] = None
                    result['hip_error_pct'] = None
                
                if actual_bust is not None and not np.isnan(actual_bust):
                    result['bust_error_cm'] = abs(preds['bust_predicted'] - actual_bust)
                    result['bust_error_pct'] = (abs(preds['bust_predicted'] - actual_bust) / actual_bust) * 100
                else:
                    result['bust_error_cm'] = None
                    result['bust_error_pct'] = None
                
                results.append(result)
                
            except Exception as e:
                print(f"❌ Error {photo_id}: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def calculate_metrics(self, results_df: pd.DataFrame) -> Dict:
        """Calculate metrics."""
        metrics = {}
        
        # Hip metrics
        if 'hip_error_cm' in results_df.columns:
            hip_errors = results_df['hip_error_cm'].dropna()
            if len(hip_errors) > 0:
                metrics['hip'] = {
                    'MAE': hip_errors.mean(),
                    'RMSE': np.sqrt((hip_errors ** 2).mean()),
                    'Max Error': hip_errors.max(),
                    'Median': hip_errors.median(),
                    'Samples': len(hip_errors)
                }
        
        # Bust metrics
        if 'bust_error_cm' in results_df.columns:
            bust_errors = results_df['bust_error_cm'].dropna()
            if len(bust_errors) > 0:
                metrics['bust'] = {
                    'MAE': bust_errors.mean(),
                    'RMSE': np.sqrt((bust_errors ** 2).mean()),
                    'Max Error': bust_errors.max(),
                    'Median': bust_errors.median(),
                    'Samples': len(bust_errors)
                }
        
        # Combined
        if 'hip_error_cm' in results_df.columns and 'bust_error_cm' in results_df.columns:
            all_errors = pd.concat([
                results_df['hip_error_cm'].dropna(),
                results_df['bust_error_cm'].dropna()
            ])
            if len(all_errors) > 0:
                metrics['combined'] = {
                    'MAE': all_errors.mean(),
                    'RMSE': np.sqrt((all_errors ** 2).mean()),
                    'Median': all_errors.median(),
                }
        
        return metrics
    
    def print_metrics(self, metrics: Dict):
        """Print metrics."""
        print("\n" + "="*80)
        print("📊 MODEL PERFORMANCE METRICS")
        print("="*80)
        
        if not metrics:
            print("❌ No metrics available")
            return
        
        for measurement, values in metrics.items():
            print(f"\n🎯 {measurement.upper()}:")
            print("-" * 60)
            for metric_name, value in values.items():
                if metric_name == 'Samples':
                    print(f"  {metric_name:15s}: {value}")
                else:
                    print(f"  {metric_name:15s}: {value:6.2f} cm")
        
        print("\n" + "="*80)
    
    def export_results(self, results_df: pd.DataFrame, output_path: str = 'predictions_results.csv'):
        """Export to CSV."""
        results_df.to_csv(output_path, index=False)
        print(f"✓ Results exported: {output_path}")


def main():
    """Main prediction function."""
    config = Config()
    
    print("="*80)
    print("HIP & BUST PREDICTION - INFERENCE")
    print("="*80)
    
    # ===== MANUAL CHECKPOINT PATH =====
    CHECKPOINT_PATH = 'checkpoints_resnet18/best_epoch18_valloss0.1222_20260109_083616.pth'
    # Or auto-detect:
    # CHECKPOINT_PATH = None
    
    print(f"\n🎯 Model: {CHECKPOINT_PATH}\n")
    
    # Create predictor
    predictor = HipBustPredictor(
        checkpoint_path=CHECKPOINT_PATH,
        config=config
    )
    
    # Load validation data
    val_data_path = config.paths.PROCESSED_DIR / 'val_data.csv'
    
    if not val_data_path.exists():
        print(f"❌ File not found: {val_data_path}")
        return
    
    val_data = pd.read_csv(val_data_path)
    print(f"📊 Loaded {len(val_data)} validation samples\n")
    
    # Predict on subset (for testing)
    # val_data = val_data.head(50)  # Test with 50 samples first
    
    # Predict
    results = predictor.predict_batch(val_data)
    
    if len(results) == 0:
        print("\n❌ No predictions generated!")
        return
    
    print(f"\n✓ Predictions complete: {len(results)} samples")
    
    # Calculate metrics
    metrics = predictor.calculate_metrics(results)
    predictor.print_metrics(metrics)
    
    # Export
    predictor.export_results(results, 'predictions_with_errors.csv')
    
    # Show sample
    print("\n" + "="*80)
    print("📋 SAMPLE PREDICTIONS (First 10)")
    print("="*80)
    
    cols = ['photo_id', 'height_cm', 'hip_actual', 'hip_predicted', 'hip_error_cm',
            'bust_actual', 'bust_predicted', 'bust_error_cm']
    available = [c for c in cols if c in results.columns]
    
    print(results[available].head(10).to_string(index=False))
    print("="*80)


if __name__ == "__main__":
    main()
