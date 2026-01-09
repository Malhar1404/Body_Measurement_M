# """
# predict.py - Inference for hip and bust prediction
# """

# import torch
# import pandas as pd
# from pathlib import Path
# from typing import Dict
# from tqdm import tqdm

# import sys
# sys.path.append(str(Path(__file__).parent.parent.parent))
# from resnet18.config.config import Config
# from resnet18.models.model import create_model
# from resnet18.features.image_preprocessor import ImagePreprocessor
# from resnet18.features.measurement_preprocessor import MeasurementPreprocessor


# class HipBustPredictor:
#     """Predictor for hip and bust measurements."""
    
#     def __init__(self, checkpoint_path: str = None, config: Config = None):
#         self.config = config if config else Config()
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
#         print(f"🚀 Device: {self.device}")
        
#         # Find best checkpoint
#         if checkpoint_path is None:
#             checkpoint_path = self._find_best_checkpoint()
        
#         print(f"📂 Loading: {checkpoint_path}")
        
#         # Load model
#         self.model = create_model(self.config)
#         checkpoint = torch.load(checkpoint_path, map_location=self.device)
#         self.model.load_state_dict(checkpoint['model_state_dict'])
#         self.model = self.model.to(self.device)
#         self.model.eval()
        
#         print(f"✓ Epoch: {checkpoint['epoch']}")
#         print(f"✓ Val loss: {checkpoint.get('best_val_loss', 'N/A')}")
        
#         # Load preprocessors
#         self.image_preprocessor = ImagePreprocessor(self.config)
#         self.image_preprocessor.load_params(
#             str(self.config.paths.PROCESSED_DIR / 'image_preprocessor_params.pkl')
#         )
        
#         self.measurement_preprocessor = MeasurementPreprocessor(self.config)
#         self.measurement_preprocessor.load_params(
#             str(self.config.paths.PROCESSED_DIR / 'measurement_preprocessor_params.pkl')
#         )
    
#     def _find_best_checkpoint(self):
#         """Find best checkpoint."""
#         checkpoint_dir = Path('checkpoints_resnet18')
#         best_files = list(checkpoint_dir.glob('best*.pth'))
        
#         if not best_files:
#             return str(checkpoint_dir / 'last_checkpoint.pth')
        
#         return str(best_files[0])
    
#     @torch.no_grad()
#     def predict_single(self, mask_path, mask_left_path, height_cm: float) -> Dict[str, float]:
#         """Predict for single image."""
#         # Load images
#         mask_img = self.image_preprocessor.transform(mask_path)
#         mask_left_img = self.image_preprocessor.transform(mask_left_path)
        
#         # To tensors
#         mask_tensor = torch.from_numpy(mask_img).permute(2, 0, 1).unsqueeze(0).float()
#         mask_left_tensor = torch.from_numpy(mask_left_img).permute(2, 0, 1).unsqueeze(0).float()
        
#         height_normalized = self.measurement_preprocessor.transform_height(height_cm)
#         height_tensor = torch.tensor([height_normalized], dtype=torch.float32)
        
#         # Move to device
#         mask_tensor = mask_tensor.to(self.device)
#         mask_left_tensor = mask_left_tensor.to(self.device)
#         height_tensor = height_tensor.to(self.device)
        
#         # Predict
#         predictions = self.model(mask_tensor, mask_left_tensor, height_tensor)
        
#         # Denormalize
#         predictions_np = predictions.cpu().numpy()
#         predictions_cm = self.measurement_preprocessor.inverse_transform(predictions_np)[0]
        
#         return {
#             'height_cm': height_cm,
#             'hip_cm': float(predictions_cm[0]),
#             'bust_cm': float(predictions_cm[1])
#         }
    
#     @torch.no_grad()
#     def predict_batch(self, data: pd.DataFrame) -> pd.DataFrame:
#         """Predict for batch."""
#         results = []
        
#         print(f"🔮 Predicting for {len(data)} samples...")
        
#         for _, row in tqdm(data.iterrows(), total=len(data)):
#             photo_id = row['photo_id']
#             height = row['height_cm']
            
#             mask_path = self.config.paths.MASK_DIR / f"{photo_id}.png"
#             mask_left_path = self.config.paths.MASK_LEFT_DIR / f"{photo_id}.png"
            
#             if not mask_path.exists() or not mask_left_path.exists():
#                 continue
            
#             try:
#                 preds = self.predict_single(mask_path, mask_left_path, height)
#                 preds['photo_id'] = photo_id
#                 results.append(preds)
#             except Exception as e:
#                 print(f"Error {photo_id}: {e}")
#                 continue
        
#         return pd.DataFrame(results)
#     def dark_to_light_skin(pixelated,
#                        dark_threshold=70,
#                         skin_color=(180, 210, 235)):
#         """
#         Convert dark/black shaded pixels to light skin tone

#         Args:
#             pixelated: pixelated face ROI (BGR)
#             dark_threshold: brightness cutoff (0–255)
#             skin_color: light skin tone (BGR)
#         """

#         # Convert to HSV
#         hsv = cv2.cvtColor(pixelated, cv2.COLOR_BGR2HSV)
#         h, s, v = cv2.split(hsv)

#         # Mask dark pixels (low brightness)
#         dark_mask = v < dark_threshold

#         # Create skin color image
#         skin_img = np.zeros_like(pixelated)
#         skin_img[:] = skin_color

#         # Blend skin color with original texture
#         gray = cv2.cvtColor(pixelated, cv2.COLOR_BGR2GRAY)
#         gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

#         blended_skin = cv2.addWeighted(skin_img, 0.75, gray, 0.25, 0)

#         # Replace only dark pixels
#         result = pixelated.copy()
#         result[dark_mask] = blended_skin[dark_mask]

#         return result

# def main():
#     config = Config()
#     predictor = HipBustPredictor(config=config)
    
#     # Load validation data
#     val_data = pd.read_csv(config.paths.PROCESSED_DIR / 'val_data.csv')
    
#     # Predict
#     results = predictor.predict_batch(val_data.head(10))
    
#     print(results)


# if __name__ == "__main__":
#     main()


"""
predict.py - Inference for hip and bust prediction with detailed error analysis
"""
"""
predict.py - Inference with Manual Model Path Selection
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
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
            checkpoint_path = str(checkpoint_path)  # Convert to string if Path object
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
                return str(sorted(checkpoints)[-1])  # Get latest
        
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    
    def list_available_checkpoints(self):
        """List all available checkpoints."""
        checkpoint_dir = Path('checkpoints_resnet18')
        checkpoints = sorted(checkpoint_dir.glob('*.pth'))
        
        print("\n" + "="*80)
        print("📂 AVAILABLE CHECKPOINTS")
        print("="*80)
        
        if not checkpoints:
            print("❌ No checkpoints found in", checkpoint_dir)
            return []
        
        for i, ckpt in enumerate(checkpoints, 1):
            # Load checkpoint info
            try:
                checkpoint = torch.load(ckpt, map_location='cpu')
                epoch = checkpoint.get('epoch', 'N/A')
                val_loss = checkpoint.get('best_val_loss', 'N/A')
                hip_mae = checkpoint.get('best_hip_mae', 'N/A')
                bust_mae = checkpoint.get('best_bust_mae', 'N/A')
                
                print(f"\n{i}. {ckpt.name}")
                print(f"   Epoch: {epoch} | Val Loss: {val_loss}")
                print(f"   Hip MAE: {hip_mae} | Bust MAE: {bust_mae}")
            except Exception as e:
                print(f"\n{i}. {ckpt.name}")
                print(f"   ⚠️  Error reading checkpoint: {e}")
        
        print("="*80 + "\n")
        return checkpoints
    
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
        
        for _, row in tqdm(data.iterrows(), total=len(data)):
            photo_id = row.get('photo_id', row.get('image_id', None))
            height = row['height']
            actual_hip = row.get('hip', None)
            actual_bust = row.get('bust', None)
            
            # Construct image paths
            mask_path = self.config.paths.PROCESSED_DIR / 'masks' / row['mask_path']
            mask_left_path = self.config.paths.PROCESSED_DIR / 'masks' / row['mask_left_path']
            
            if not mask_path.exists() or not mask_left_path.exists():
                print(f"⚠️  Skipping {photo_id}: Images not found")
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
                if actual_hip is not None:
                    result['hip_error_cm'] = abs(preds['hip_predicted'] - actual_hip)
                    result['hip_error_pct'] = (abs(preds['hip_predicted'] - actual_hip) / actual_hip) * 100
                
                if actual_bust is not None:
                    result['bust_error_cm'] = abs(preds['bust_predicted'] - actual_bust)
                    result['bust_error_pct'] = (abs(preds['bust_predicted'] - actual_bust) / actual_bust) * 100
                
                results.append(result)
                
            except Exception as e:
                print(f"❌ Error {photo_id}: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def calculate_metrics(self, results_df: pd.DataFrame) -> Dict:
        """Calculate detailed metrics."""
        metrics = {}
        
        # Hip metrics
        if 'hip_error_cm' in results_df.columns:
            hip_errors = results_df['hip_error_cm'].dropna()
            metrics['hip'] = {
                'MAE': hip_errors.mean(),
                'RMSE': np.sqrt((hip_errors ** 2).mean()),
                'Max Error': hip_errors.max(),
                'Min Error': hip_errors.min(),
                'Std': hip_errors.std(),
                'Median': hip_errors.median(),
                'MAE %': results_df['hip_error_pct'].mean(),
            }
        
        # Bust metrics
        if 'bust_error_cm' in results_df.columns:
            bust_errors = results_df['bust_error_cm'].dropna()
            metrics['bust'] = {
                'MAE': bust_errors.mean(),
                'RMSE': np.sqrt((bust_errors ** 2).mean()),
                'Max Error': bust_errors.max(),
                'Min Error': bust_errors.min(),
                'Std': bust_errors.std(),
                'Median': bust_errors.median(),
                'MAE %': results_df['bust_error_pct'].mean(),
            }
        
        # Combined metrics
        if 'hip_error_cm' in results_df.columns and 'bust_error_cm' in results_df.columns:
            all_errors = pd.concat([
                results_df['hip_error_cm'].dropna(),
                results_df['bust_error_cm'].dropna()
            ])
            
            metrics['combined'] = {
                'MAE': all_errors.mean(),
                'RMSE': np.sqrt((all_errors ** 2).mean()),
                'Max Error': all_errors.max(),
                'Min Error': all_errors.min(),
                'Std': all_errors.std(),
                'Median': all_errors.median(),
            }
        
        return metrics
    
    def print_metrics(self, metrics: Dict):
        """Print metrics in a nice format."""
        print("\n" + "="*80)
        print("📊 MODEL PERFORMANCE METRICS")
        print("="*80)
        
        for measurement, values in metrics.items():
            print(f"\n🎯 {measurement.upper()} MEASUREMENTS:")
            print("-" * 60)
            for metric_name, value in values.items():
                if 'MAE %' in metric_name:
                    print(f"  {metric_name:15s}: {value:6.2f}%")
                else:
                    print(f"  {metric_name:15s}: {value:6.2f} cm")
        
        print("\n" + "="*80)
    
    def visualize_results(self, results_df: pd.DataFrame, save_path: str = None):
        """Create comprehensive visualization."""
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
                f"{row.get('hip_error_cm', 0):.1f}" if 'hip_error_cm' in row else 'N/A',
                f"{row['bust_actual']:.1f}" if pd.notna(row.get('bust_actual')) else 'N/A',
                f"{row['bust_predicted']:.1f}",
                f"{row.get('bust_error_cm', 0):.1f}" if 'bust_error_cm' in row else 'N/A',
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
            r2 = r2_score(bust_actual, bust_pred)
            
            ax3.set_xlabel('Actual Bust (cm)', fontsize=11)
            ax3.set_ylabel('Predicted Bust (cm)', fontsize=11)
            ax3.set_title(f'Bust Prediction (R² = {r2:.3f})', fontsize=12, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # ===== 4. Hip error distribution =====
        ax4 = plt.subplot(3, 3, 4)
        if 'hip_error_cm' in results_df.columns:
            hip_errors = results_df['hip_error_cm'].dropna()
            ax4.hist(hip_errors, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
            ax4.axvline(hip_errors.mean(), color='red', linestyle='--', lw=2, label=f'Mean: {hip_errors.mean():.2f} cm')
            ax4.axvline(hip_errors.median(), color='green', linestyle='--', lw=2, label=f'Median: {hip_errors.median():.2f} cm')
            ax4.set_xlabel('Hip Error (cm)', fontsize=11)
            ax4.set_ylabel('Frequency', fontsize=11)
            ax4.set_title('Hip Error Distribution', fontsize=12, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # ===== 5. Bust error distribution =====
        ax5 = plt.subplot(3, 3, 5)
        if 'bust_error_cm' in results_df.columns:
            bust_errors = results_df['bust_error_cm'].dropna()
            ax5.hist(bust_errors, bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
            ax5.axvline(bust_errors.mean(), color='red', linestyle='--', lw=2, label=f'Mean: {bust_errors.mean():.2f} cm')
            ax5.axvline(bust_errors.median(), color='green', linestyle='--', lw=2, label=f'Median: {bust_errors.median():.2f} cm')
            ax5.set_xlabel('Bust Error (cm)', fontsize=11)
            ax5.set_ylabel('Frequency', fontsize=11)
            ax5.set_title('Bust Error Distribution', fontsize=12, fontweight='bold')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # ===== 6. Error box plot comparison =====
        ax6 = plt.subplot(3, 3, 6)
        if 'hip_error_cm' in results_df.columns and 'bust_error_cm' in results_df.columns:
            error_data = [
                results_df['hip_error_cm'].dropna(),
                results_df['bust_error_cm'].dropna()
            ]
            bp = ax6.boxplot(error_data, labels=['Hip', 'Bust'], patch_artist=True)
            bp['boxes'][0].set_facecolor('skyblue')
            bp['boxes'][1].set_facecolor('lightcoral')
            ax6.set_ylabel('Error (cm)', fontsize=11)
            ax6.set_title('Error Comparison: Hip vs Bust', fontsize=12, fontweight='bold')
            ax6.grid(True, alpha=0.3, axis='y')
        
        # ===== 7. Hip: Bland-Altman plot =====
        ax7 = plt.subplot(3, 3, 7)
        if 'hip_actual' in results_df.columns:
            hip_actual = results_df['hip_actual'].dropna()
            hip_pred = results_df.loc[hip_actual.index, 'hip_predicted']
            
            mean_vals = (hip_actual + hip_pred) / 2
            diff_vals = hip_pred - hip_actual
            
            ax7.scatter(mean_vals, diff_vals, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
            ax7.axhline(diff_vals.mean(), color='red', linestyle='--', lw=2, label=f'Mean: {diff_vals.mean():.2f}')
            ax7.axhline(diff_vals.mean() + 1.96*diff_vals.std(), color='gray', linestyle='--', lw=1, label='±1.96 SD')
            ax7.axhline(diff_vals.mean() - 1.96*diff_vals.std(), color='gray', linestyle='--', lw=1)
            ax7.axhline(0, color='black', linestyle='-', lw=1)
            ax7.set_xlabel('Mean of Actual & Predicted (cm)', fontsize=11)
            ax7.set_ylabel('Difference (Predicted - Actual)', fontsize=11)
            ax7.set_title('Hip: Bland-Altman Plot', fontsize=12, fontweight='bold')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
        
        # ===== 8. Bust: Bland-Altman plot =====
        ax8 = plt.subplot(3, 3, 8)
        if 'bust_actual' in results_df.columns:
            bust_actual = results_df['bust_actual'].dropna()
            bust_pred = results_df.loc[bust_actual.index, 'bust_predicted']
            
            mean_vals = (bust_actual + bust_pred) / 2
            diff_vals = bust_pred - bust_actual
            
            ax8.scatter(mean_vals, diff_vals, alpha=0.6, s=50, color='orange', edgecolors='k', linewidth=0.5)
            ax8.axhline(diff_vals.mean(), color='red', linestyle='--', lw=2, label=f'Mean: {diff_vals.mean():.2f}')
            ax8.axhline(diff_vals.mean() + 1.96*diff_vals.std(), color='gray', linestyle='--', lw=1, label='±1.96 SD')
            ax8.axhline(diff_vals.mean() - 1.96*diff_vals.std(), color='gray', linestyle='--', lw=1)
            ax8.axhline(0, color='black', linestyle='-', lw=1)
            ax8.set_xlabel('Mean of Actual & Predicted (cm)', fontsize=11)
            ax8.set_ylabel('Difference (Predicted - Actual)', fontsize=11)
            ax8.set_title('Bust: Bland-Altman Plot', fontsize=12, fontweight='bold')
            ax8.legend()
            ax8.grid(True, alpha=0.3)
        
        # ===== 9. Metrics summary text =====
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        metrics = self.calculate_metrics(results_df)
        
        summary_text = "📊 OVERALL METRICS\n" + "="*40 + "\n\n"
        
        if 'hip' in metrics:
            summary_text += "🎯 HIP:\n"
            summary_text += f"  MAE:  {metrics['hip']['MAE']:.2f} cm\n"
            summary_text += f"  RMSE: {metrics['hip']['RMSE']:.2f} cm\n"
            summary_text += f"  MAE%: {metrics['hip']['MAE %']:.2f}%\n\n"
        
        if 'bust' in metrics:
            summary_text += "🎯 BUST:\n"
            summary_text += f"  MAE:  {metrics['bust']['MAE']:.2f} cm\n"
            summary_text += f"  RMSE: {metrics['bust']['RMSE']:.2f} cm\n"
            summary_text += f"  MAE%: {metrics['bust']['MAE %']:.2f}%\n\n"
        
        if 'combined' in metrics:
            summary_text += "🎯 COMBINED:\n"
            summary_text += f"  MAE:  {metrics['combined']['MAE']:.2f} cm\n"
            summary_text += f"  RMSE: {metrics['combined']['RMSE']:.2f} cm\n"
        
        ax9.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
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
    """Main prediction function with manual model path selection."""
    config = Config()
    
    print("="*80)
    print("HIP & BUST PREDICTION - INFERENCE")
    print("="*80)
    
    # ===== MANUAL MODEL PATH SELECTION =====
    
    # Option 1: Specify exact checkpoint path
    CHECKPOINT_PATH = 'checkpoints_resnet18/best_epoch18_valloss0.1222_20260109_083616.pth'
    
    # Option 2: Use best loss model
    # CHECKPOINT_PATH = 'checkpoints_resnet18/best_loss_epoch10_valloss0.8545.pth'
    
    # Option 3: Use best bust model
    # CHECKPOINT_PATH = 'checkpoints_resnet18/best_bust_epoch12_bustmae2.8cm.pth'
    
    # Option 4: Use last checkpoint
    # CHECKPOINT_PATH = 'checkpoints_resnet18/last_checkpoint.pth'
    
    # Option 5: Let it auto-detect (leave as None)
    # CHECKPOINT_PATH = None
    
    print(f"\n🎯 Selected Model: {CHECKPOINT_PATH}\n")
    
    # Create predictor with manual path
    predictor = HipBustPredictor(
        checkpoint_path=CHECKPOINT_PATH,
        config=config
    )
    
    # Optional: List all available checkpoints
    # predictor.list_available_checkpoints()
    
    # Load validation data
    val_data = pd.read_csv(config.paths.PROCESSED_DIR / 'val_data.csv')
    
    print(f"\n📊 Loaded {len(val_data)} validation samples")
    
    # Predict
    results = predictor.predict_batch(val_data)
    
    print(f"\n✓ Predictions complete: {len(results)} samples")
    
    # Calculate and print metrics
    metrics = predictor.calculate_metrics(results)
    predictor.print_metrics(metrics)
    
    # Create visualizations
    print("\n📊 Generating visualizations...")
    predictor.visualize_results(
        results,
        save_path='prediction_analysis.png'
    )
    
    # Export results
    predictor.export_results(results, 'predictions_with_errors.csv')
    
    # Print sample results
    print("\n" + "="*80)
    print("📋 SAMPLE PREDICTIONS (First 10)")
    print("="*80)
    print(results[['photo_id', 'hip_actual', 'hip_predicted', 'hip_error_cm',
                   'bust_actual', 'bust_predicted', 'bust_error_cm']].head(10).to_string(index=False))
    print("="*80)


if __name__ == "__main__":
    main()
