"""
predict.py - Inference with correct data structure paths
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from resnet18.config.config import Config
from resnet18.models.model import create_model
from resnet18.features.image_preprocessor import ImagePreprocessor
from resnet18.features.measurement_preprocessor import MeasurementPreprocessor


class HipBustPredictor:
    """Predictor for hip and bust measurements with error analysis."""
    
    def __init__(self, checkpoint_path: str = None, config: Config = None):
        self.config = config if config else Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🚀 Device: {self.device}")
        
        # Find best checkpoint or use provided path
        if checkpoint_path is None:
            checkpoint_path = self._find_best_checkpoint()
        else:
            checkpoint_path = str(checkpoint_path)
            if not Path(checkpoint_path).exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        
        # Load model
        self.model = create_model(self.config)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Display checkpoint info
        epoch = checkpoint.get('epoch', 'N/A')
        val_loss = checkpoint.get('best_val_loss', 'N/A')
        hip_mae = checkpoint.get('best_hip_mae', 'N/A')
        bust_mae = checkpoint.get('best_bust_mae', 'N/A')
        
        print(f"\n✓ Checkpoint Info:")
        print(f"  Epoch: {epoch}")
        print(f"  Val Loss: {val_loss}")
        print(f"  Best Hip MAE: {hip_mae}")
        print(f"  Best Bust MAE: {bust_mae}")
        
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
        """Find best checkpoint automatically."""
        checkpoint_dir = Path('checkpoints_resnet18')
        
        # Priority order: best_loss > best_hip > best_bust > last
        for pattern in ['best_loss*.pth', 'best_hip*.pth', 'best_bust*.pth', 'last_checkpoint.pth']:
            checkpoints = list(checkpoint_dir.glob(pattern))
            if checkpoints:
                return str(sorted(checkpoints)[-1])
        
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    
    def _find_mask_paths(self, row, base_dir: Path):
        """
        Find mask paths based on your data structure.
        
        Your structure:
        data/raw/train/mask/xxx.png
        data/raw/train/mask_left/xxx.png
        """
        
        # Try different possible locations
        photo_id = row.get('photo_id', row.get('image_id', None))
        
        # Option 1: From train_data.csv paths
        if 'mask_path' in row and 'mask_left_path' in row:
            # Paths are relative in CSV
            mask_path = base_dir / 'raw' / 'train' / row['mask_path']
            mask_left_path = base_dir / 'raw' / 'train' / row['mask_left_path']
            
            if mask_path.exists() and mask_left_path.exists():
                return mask_path, mask_left_path
        
        # Option 2: Direct construction from photo_id
        if photo_id:
            mask_path = base_dir / 'raw' / 'train' / 'mask' / f"{photo_id}.png"
            mask_left_path = base_dir / 'raw' / 'train' / 'mask_left' / f"{photo_id}.png"
            
            if mask_path.exists() and mask_left_path.exists():
                return mask_path, mask_left_path
        
        # Option 3: Check processed_resnet18 folder
        if 'mask_path' in row and 'mask_left_path' in row:
            mask_path = self.config.paths.PROCESSED_DIR / 'masks' / row['mask_path']
            mask_left_path = self.config.paths.PROCESSED_DIR / 'masks' / row['mask_left_path']
            
            if mask_path.exists() and mask_left_path.exists():
                return mask_path, mask_left_path
        
        return None, None
    
    @torch.no_grad()
    def predict_single(self, mask_path, mask_left_path, height_cm: float) -> Dict[str, float]:
        """Predict for single image."""
        # Load images
        mask_img = self.image_preprocessor.transform(str(mask_path))
        mask_left_img = self.image_preprocessor.transform(str(mask_left_path))
        
        # To tensors
        mask_tensor = torch.from_numpy(mask_img).permute(2, 0, 1).unsqueeze(0).float()
        mask_left_tensor = torch.from_numpy(mask_left_img).permute(2, 0, 1).unsqueeze(0).float()
        
        # Normalize height
        height_normalized = self.measurement_preprocessor.transform_height(height_cm)
        height_tensor = torch.tensor([[height_normalized]], dtype=torch.float32)
        
        # Move to device
        mask_tensor = mask_tensor.to(self.device)
        mask_left_tensor = mask_left_tensor.to(self.device)
        height_tensor = height_tensor.to(self.device)
        
        # Predict
        predictions = self.model(mask_tensor, mask_left_tensor, height_tensor)
        
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
        """Predict for batch with ground truth comparison."""
        results = []
        
        print(f"🔮 Predicting for {len(data)} samples...")
        
        base_dir = self.config.paths.DATA_DIR
        
        for idx, row in tqdm(data.iterrows(), total=len(data), desc="Predicting"):
            photo_id = row.get('photo_id', row.get('image_id', idx))
            height = row['height']
            actual_hip = row.get('hip', None)
            actual_bust = row.get('bust', None)
            
            # Find mask paths
            mask_path, mask_left_path = self._find_mask_paths(row, base_dir)
            
            if mask_path is None or mask_left_path is None:
                print(f"⚠️  Skipping {photo_id}: Masks not found")
                continue
            
            try:
                # Predict
                preds = self.predict_single(mask_path, mask_left_path, height)
                
                # Add actual values and errors
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
                import traceback
                traceback.print_exc()
                continue
        
        return pd.DataFrame(results)
    
    def calculate_metrics(self, results_df: pd.DataFrame) -> Dict:
        """Calculate detailed metrics."""
        metrics = {}
        
        # Hip metrics
        if 'hip_error_cm' in results_df.columns:
            hip_errors = results_df['hip_error_cm'].dropna()
            if len(hip_errors) > 0:
                metrics['hip'] = {
                    'MAE': hip_errors.mean(),
                    'RMSE': np.sqrt((hip_errors ** 2).mean()),
                    'Max Error': hip_errors.max(),
                    'Min Error': hip_errors.min(),
                    'Std': hip_errors.std(),
                    'Median': hip_errors.median(),
                    'MAE %': results_df['hip_error_pct'].dropna().mean(),
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
                    'Min Error': bust_errors.min(),
                    'Std': bust_errors.std(),
                    'Median': bust_errors.median(),
                    'MAE %': results_df['bust_error_pct'].dropna().mean(),
                    'Samples': len(bust_errors)
                }
        
        # Combined metrics
        if 'hip_error_cm' in results_df.columns and 'bust_error_cm' in results_df.columns:
            all_errors = pd.concat([
                results_df['hip_error_cm'].dropna(),
                results_df['bust_error_cm'].dropna()
            ])
            
            if len(all_errors) > 0:
                metrics['combined'] = {
                    'MAE': all_errors.mean(),
                    'RMSE': np.sqrt((all_errors ** 2).mean()),
                    'Max Error': all_errors.max(),
                    'Min Error': all_errors.min(),
                    'Std': all_errors.std(),
                    'Median': all_errors.median(),
                    'Samples': len(all_errors)
                }
        
        return metrics
    
    def print_metrics(self, metrics: Dict):
        """Print metrics in a nice format."""
        print("\n" + "="*80)
        print("📊 MODEL PERFORMANCE METRICS")
        print("="*80)
        
        if not metrics:
            print("❌ No metrics available (no ground truth data)")
            return
        
        for measurement, values in metrics.items():
            print(f"\n🎯 {measurement.upper()} MEASUREMENTS:")
            print("-" * 60)
            for metric_name, value in values.items():
                if metric_name == 'Samples':
                    print(f"  {metric_name:15s}: {value}")
                elif 'MAE %' in metric_name:
                    print(f"  {metric_name:15s}: {value:6.2f}%")
                else:
                    print(f"  {metric_name:15s}: {value:6.2f} cm")
        
        print("\n" + "="*80)
    
    def visualize_results(self, results_df: pd.DataFrame, save_path: str = None):
        """Create comprehensive visualization."""
        # Check if we have ground truth
        has_ground_truth = ('hip_actual' in results_df.columns and 
                           results_df['hip_actual'].notna().any())
        
        if not has_ground_truth:
            print("⚠️  No ground truth available, skipping visualization")
            return
        
        fig = plt.figure(figsize=(20, 12))
        
        # ===== 1. Side-by-side comparison table =====
        ax1 = plt.subplot(3, 3, 1)
        ax1.axis('tight')
        ax1.axis('off')
        
        # Prepare table data (first 10 samples)
        table_data = []
        for idx, row in results_df.head(10).iterrows():
            table_data.append([
                f"{row.get('photo_id', idx)}",
                f"{row['hip_actual']:.1f}" if pd.notna(row.get('hip_actual')) else 'N/A',
                f"{row['hip_predicted']:.1f}",
                f"{row.get('hip_error_cm', 0):.1f}" if pd.notna(row.get('hip_error_cm')) else 'N/A',
                f"{row['bust_actual']:.1f}" if pd.notna(row.get('bust_actual')) else 'N/A',
                f"{row['bust_predicted']:.1f}",
                f"{row.get('bust_error_cm', 0):.1f}" if pd.notna(row.get('bust_error_cm')) else 'N/A',
            ])
        
        table = ax1.table(
            cellText=table_data,
            colLabels=['ID', 'Hip\nActual', 'Hip\nPred', 'Hip\nError',
                      'Bust\nActual', 'Bust\nPred', 'Bust\nError'],
            cellLoc='center',
            loc='center',
            colWidths=[0.1, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Color code header
        for i in range(7):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax1.set_title('Prediction vs Actual (Top 10 Samples)', fontsize=14, fontweight='bold', pad=20)
        
        # ===== 2. Hip: Predicted vs Actual scatter =====
        ax2 = plt.subplot(3, 3, 2)
        if 'hip_actual' in results_df.columns:
            hip_actual = results_df['hip_actual'].dropna()
            hip_pred = results_df.loc[hip_actual.index, 'hip_predicted']
            
            ax2.scatter(hip_actual, hip_pred, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
            
            # Perfect prediction line
            min_val = min(hip_actual.min(), hip_pred.min())
            max_val = max(hip_actual.max(), hip_pred.max())
            ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            
            # R² score
            from sklearn.metrics import r2_score
            r2 = r2_score(hip_actual, hip_pred)
            
            ax2.set_xlabel('Actual Hip (cm)', fontsize=11)
            ax2.set_ylabel('Predicted Hip (cm)', fontsize=11)
            ax2.set_title(f'Hip Prediction (R² = {r2:.3f})', fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # ===== 3. Bust: Predicted vs Actual scatter =====
        ax3 = plt.subplot(3, 3, 3)
        if 'bust_actual' in results_df.columns:
            bust_actual = results_df['bust_actual'].dropna()
            bust_pred = results_df.loc[bust_actual.index, 'bust_predicted']
            
            ax3.scatter(bust_actual, bust_pred, alpha=0.6, s=50, color='orange', edgecolors='k', linewidth=0.5)
            
            # Perfect prediction line
            min_val = min(bust_actual.min(), bust_pred.min())
            max_val = max(bust_actual.max(), bust_pred.max())
            ax3.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            
            # R² score
            from sklearn.metrics import r2_score
            r2 = r2_score(bust_actual, bust_pred)
            
            ax3.set_xlabel('Actual Bust (cm)', fontsize=11)
            ax3.set_ylabel('Predicted Bust (cm)', fontsize=11)
            ax3.set_title(f'Bust Prediction (R² = {r2:.3f})', fontsize=12, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Continue with other plots...
        # (Rest of visualization code remains the same)
        
        plt.suptitle('Hip & Bust Prediction Analysis', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Visualization saved: {save_path}")
        
        plt.show()
    
    def export_results(self, results_df: pd.DataFrame, output_path: str = None):
        """Export results to CSV."""
        if output_path is None:
            output_path = 'predictions_results.csv'
        
        results_df.to_csv(output_path, index=False)
        print(f"✓ Results exported: {output_path}")


def main():
    """Main prediction function."""
    config = Config()
    
    print("="*80)
    print("HIP & BUST PREDICTION - INFERENCE")
    print("="*80)
    
    # ===== MANUAL MODEL PATH =====
    # Specify your checkpoint path here
    CHECKPOINT_PATH = 'checkpoints_resnet18/best_epoch18_valloss0.1222_20260109_083616.pth'
    # Or use auto-detection
    # CHECKPOINT_PATH = None
    
    print(f"\n🎯 Selected Model: {CHECKPOINT_PATH}\n")
    
    # Create predictor
    predictor = HipBustPredictor(
        checkpoint_path=CHECKPOINT_PATH,
        config=config
    )
    
    # Load validation data
    val_data_path = config.paths.PROCESSED_DIR / 'val_data.csv'
    
    if not val_data_path.exists():
        print(f"❌ Validation data not found: {val_data_path}")
        print("   Please run data preprocessing first!")
        return
    
    val_data = pd.read_csv(val_data_path)
    
    print(f"\n📊 Loaded {len(val_data)} validation samples")
    print(f"   Columns: {list(val_data.columns)}")
    
    # Predict
    results = predictor.predict_batch(val_data)
    
    if len(results) == 0:
        print("\n❌ No predictions generated! Check your data paths.")
        return
    
    print(f"\n✓ Predictions complete: {len(results)} samples")
    
    # Calculate and print metrics
    metrics = predictor.calculate_metrics(results)
    predictor.print_metrics(metrics)
    
    # Create visualizations
    if metrics:
        print("\n📊 Generating visualizations...")
        try:
            predictor.visualize_results(
                results,
                save_path='prediction_analysis.png'
            )
        except Exception as e:
            print(f"⚠️  Visualization error: {e}")
    
    # Export results
    predictor.export_results(results, 'predictions_with_errors.csv')
    
    # Print sample results
    print("\n" + "="*80)
    print("📋 SAMPLE PREDICTIONS (First 10)")
    print("="*80)
    
    display_cols = ['photo_id', 'height_cm', 'hip_actual', 'hip_predicted', 'hip_error_cm',
                    'bust_actual', 'bust_predicted', 'bust_error_cm']
    available_cols = [col for col in display_cols if col in results.columns]
    
    print(results[available_cols].head(10).to_string(index=False))
    print("="*80)


if __name__ == "__main__":
    main()
