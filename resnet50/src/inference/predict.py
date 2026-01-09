import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Union, List
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.config.config import Config
from src.models.body_measurement_model import create_model
from src.features.image_preprocessor import ImagePreprocessor
from src.features.measurement_preprocessor import MeasurementPreprocessor


class BodyMeasurementPredictor:
    """
    Predictor class for body measurement inference.
    Compatible with train_advanced.py checkpoints.
    """
    
    def __init__(self, checkpoint_path: str = None, config: Config = None, auto_find_best: bool = True):
        """
        Initialize predictor.
        
        Args:
            checkpoint_path: Path to checkpoint. If None and auto_find_best=True, finds best model
            config: Configuration object
            auto_find_best: Automatically find best checkpoint if path not provided
        """
        self.config = config if config else Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🚀 Using device: {self.device}")
        
        # Find best checkpoint if not provided
        if checkpoint_path is None and auto_find_best:
            checkpoint_path = self._find_best_checkpoint()
        
        if checkpoint_path is None:
            raise ValueError("No checkpoint provided and couldn't find best model")
        
        print(f"📂 Loading model from: {checkpoint_path}")
        
        # Create model
        self.model = create_model(self.config, model_type='resnet50', pretrained=False)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Print checkpoint info
        self.checkpoint_info = {
            'epoch': checkpoint.get('epoch', 'N/A'),
            'best_val_loss': checkpoint.get('best_val_loss', 'N/A'),
            'path': checkpoint_path
        }
        
        print(f"✓ Model loaded from epoch {self.checkpoint_info['epoch']}")
        print(f"✓ Best validation loss: {self.checkpoint_info['best_val_loss']}")
        
        # Initialize preprocessors
        self.image_preprocessor = ImagePreprocessor(self.config)
        self.image_preprocessor.fit()
        
        self.measurement_preprocessor = MeasurementPreprocessor(self.config)
        params_path = self.config.data.PROCESSED_DIR / 'measurement_preprocessor_params.pkl'
        
        if params_path.exists():
            self.measurement_preprocessor.load_params(str(params_path))
            print("✓ Loaded measurement normalization parameters")
        else:
            raise FileNotFoundError(f"Measurement preprocessor params not found at {params_path}")
        
        self.measurement_names = self.config.measurement.MEASUREMENT_COLUMNS
    
    def _find_best_checkpoint(self) -> str:
        """
        Find the best checkpoint in checkpoints/ directory.
        Looks for files with 'best' in name or lowest valloss.
        """
        checkpoint_dir = Path('checkpoints')
        
        if not checkpoint_dir.exists():
            return None
        
        # Look for files with 'best' in name
        best_files = list(checkpoint_dir.glob('*.pth'))
        
        if not best_files:
            print("⚠️  No 'best' checkpoint found, using last_checkpoint.pth")
            last_ckpt = checkpoint_dir / 'last_checkpoint.pth'
            return str(last_ckpt) if last_ckpt.exists() else None
        
        # If multiple, find one with lowest val_loss in filename
        if len(best_files) > 1:
            best_file = None
            lowest_loss = float('inf')
            
            for f in best_files:
                try:
                    # Extract valloss from filename (format: valloss=0.1234)
                    if 'valloss=' in f.stem:
                        loss_str = f.stem.split('valloss=')[1].split('_')[0]
                        loss = float(loss_str)
                        if loss < lowest_loss:
                            lowest_loss = loss
                            best_file = f
                except:
                    continue
            
            if best_file:
                print(f"✓ Found best checkpoint: {best_file.name}")
                return str(best_file)
        
        return str(best_files[0])
    
    @torch.no_grad()
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
        # Preprocess images
        mask_img = self.image_preprocessor.transform(mask_path)
        mask_left_img = self.image_preprocessor.transform(mask_left_path)
        
        # Convert to tensors (C, H, W)
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
    
    @torch.no_grad()
    def predict_batch(self, data: pd.DataFrame, batch_size: int = 32) -> pd.DataFrame:
        """
        Predict body measurements for batch of subjects (GPU optimized).
        
        Args:
            data: DataFrame with columns: photo_id, height_cm
            batch_size: Batch size for GPU processing
            
        Returns:
            DataFrame with predictions
        """
        results = []
        
        print(f"🔮 Predicting for {len(data)} samples (batch_size={batch_size})...")
        
        # Process in batches
        for i in tqdm(range(0, len(data), batch_size), desc="Predicting"):
            batch_data = data.iloc[i:i+batch_size]
            
            batch_masks = []
            batch_masks_left = []
            batch_heights = []
            batch_photo_ids = []
            
            # Load batch
            for _, row in batch_data.iterrows():
                photo_id = row['photo_id']
                height = row['height_cm']
                
                mask_path = self.config.data.MASK_DIR / f"{photo_id}.png"
                mask_left_path = self.config.data.MASK_LEFT_DIR / f"{photo_id}.png"
                
                if not mask_path.exists() or not mask_left_path.exists():
                    continue
                
                try:
                    mask_img = self.image_preprocessor.transform(mask_path)
                    mask_left_img = self.image_preprocessor.transform(mask_left_path)
                    
                    batch_masks.append(torch.from_numpy(mask_img).permute(2, 0, 1))
                    batch_masks_left.append(torch.from_numpy(mask_left_img).permute(2, 0, 1))
                    batch_heights.append(self.measurement_preprocessor.transform_height(height))
                    batch_photo_ids.append(photo_id)
                except Exception as e:
                    print(f"⚠️  Error loading {photo_id}: {e}")
                    continue
            
            if len(batch_masks) == 0:
                continue
            
            # Stack into tensors
            masks_tensor = torch.stack(batch_masks).float().to(self.device)
            masks_left_tensor = torch.stack(batch_masks_left).float().to(self.device)
            heights_tensor = torch.tensor(batch_heights, dtype=torch.float32).to(self.device)
            
            # Predict
            predictions = self.model(masks_tensor, masks_left_tensor, heights_tensor)
            
            # Denormalize
            predictions_np = predictions.cpu().numpy()
            predictions_cm = self.measurement_preprocessor.inverse_transform(predictions_np)
            
            # Store results
            for j, photo_id in enumerate(batch_photo_ids):
                result = {
                    'photo_id': photo_id,
                    'height_cm': batch_data[batch_data['photo_id'] == photo_id]['height_cm'].values[0],
                    **{name: float(pred) for name, pred in zip(self.measurement_names, predictions_cm[j])}
                }
                results.append(result)
        
        results_df = pd.DataFrame(results)
        print(f"✓ Predictions complete: {len(results_df)}/{len(data)} samples")
        
        return results_df
    
    def predict_and_compare(self, data: pd.DataFrame, batch_size: int = 32) -> pd.DataFrame:
        """
        Predict and compare with ground truth measurements.
        
        Args:
            data: DataFrame with photo_id, height_cm, and ground truth measurements
            batch_size: Batch size for processing
            
        Returns:
            DataFrame with predictions, ground truth, and errors
        """
        # Get predictions
        predictions_df = self.predict_batch(data, batch_size=batch_size)
        
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
            
            if pred_col in merged.columns and true_col in merged.columns:
                merged[f"{measurement}_error"] = merged[pred_col] - merged[true_col]
                merged[f"{measurement}_abs_error"] = merged[f"{measurement}_error"].abs()
                merged[f"{measurement}_pct_error"] = (merged[f"{measurement}_error"] / merged[true_col] * 100)
        
        return merged
    
    def visualize_predictions(self, photo_id: str, predictions: Dict[str, float],
                            ground_truth: Dict[str, float] = None,
                            save_path: str = None):
        """Visualize predictions vs ground truth."""
        measurements = [m for m in self.measurement_names if m in predictions]
        pred_values = [predictions[m] for m in measurements]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(measurements))
        width = 0.35
        
        if ground_truth:
            true_values = [ground_truth.get(m, 0) for m in measurements]
            
            bars1 = ax.bar(x - width/2, pred_values, width, label='Predicted', alpha=0.8, color='skyblue')
            bars2 = ax.bar(x + width/2, true_values, width, label='Ground Truth', alpha=0.8, color='lightcoral')
            
            # Add value labels on bars
            for bar in bars1 + bars2:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=8)
            
            errors = [abs(p - t) for p, t in zip(pred_values, true_values)]
            mae = np.mean(errors)
            ax.set_title(f'Predictions vs Ground Truth - {photo_id}\nMAE: {mae:.2f} cm', 
                        fontsize=14, fontweight='bold')
        else:
            bars = ax.bar(x, pred_values, width, alpha=0.8, color='skyblue')
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=8)
            ax.set_title(f'Predicted Measurements - {photo_id}', 
                        fontsize=14, fontweight='bold')
        
        ax.set_xlabel('Measurement', fontsize=12)
        ax.set_ylabel('Value (cm)', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('-', '\n').title() for m in measurements], 
                          rotation=45, ha='right', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved plot: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def export_results(self, results_df: pd.DataFrame, output_path: Union[str, Path]):
        """Export results to Excel or CSV."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix == '.xlsx':
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                results_df.to_excel(writer, sheet_name='Predictions', index=False)
                
                # Add checkpoint info sheet
                info_df = pd.DataFrame([self.checkpoint_info])
                info_df.to_excel(writer, sheet_name='Model_Info', index=False)
        else:
            results_df.to_csv(output_path, index=False)
        
        print(f"✓ Results exported to: {output_path}")


def main():
    """Main function with usage examples."""
    print("="*80)
    print("BODY MEASUREMENT PREDICTION")
    print("="*80)
    
    config = Config()
    
    # Auto-find best checkpoint
    predictor = BodyMeasurementPredictor(config=config, auto_find_best=True)
    
    # Predict on validation set
    val_data_path = config.data.PROCESSED_DIR / 'val_data.csv'
    
    if val_data_path.exists():
        val_data = pd.read_csv(val_data_path)
        
        print(f"\n📊 Predicting on {len(val_data)} validation samples...")
        
        # Get predictions and compare
        results = predictor.predict_and_compare(val_data, batch_size=32)
        
        # Calculate metrics
        print("\n📏 Error Statistics (cm):")
        print(f"{'Measurement':<20} {'MAE':<10} {'Mean Error':<15} {'Std Error':<12}")
        print("-" * 80)
        
        for measurement in config.measurement.MEASUREMENT_COLUMNS:
            error_col = f"{measurement}_abs_error"
            if error_col in results.columns:
                mae = results[error_col].mean()
                mean_err = results[f"{measurement}_error"].mean()
                std_err = results[f"{measurement}_error"].std()
                print(f"{measurement:<20} {mae:<10.2f} {mean_err:<15.2f} {std_err:<12.2f}")
        
        # Export
        output_dir = Path('predictions')
        predictor.export_results(results, output_dir / 'validation_predictions.xlsx')
        
        # Visualize sample
        if len(results) > 0:
            sample = results.iloc[0]
            photo_id = sample['photo_id']
            
            preds = {m: sample[f"{m}_pred"] for m in config.measurement.MEASUREMENT_COLUMNS}
            truths = {m: sample[f"{m}_true"] for m in config.measurement.MEASUREMENT_COLUMNS}
            
            predictor.visualize_predictions(
                photo_id=photo_id,
                predictions=preds,
                ground_truth=truths,
                save_path=output_dir / f'sample_prediction.png'
            )
    
    print("\n" + "="*80)
    print("✅ PREDICTION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()