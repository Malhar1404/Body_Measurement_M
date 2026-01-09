"""
predict.py - Inference for hip and bust prediction
"""

import torch
import pandas as pd
from pathlib import Path
from typing import Dict
from tqdm import tqdm

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
        
        # Find best checkpoint
        if checkpoint_path is None:
            checkpoint_path = self._find_best_checkpoint()
        
        print(f"📂 Loading: {checkpoint_path}")
        
        # Load model
        self.model = create_model(self.config)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Epoch: {checkpoint['epoch']}")
        print(f"✓ Val loss: {checkpoint.get('best_val_loss', 'N/A')}")
        
        # Load preprocessors
        self.image_preprocessor = ImagePreprocessor(self.config)
        self.image_preprocessor.load_params(
            str(self.config.paths.PROCESSED_DIR / 'image_preprocessor_params.pkl')
        )
        
        self.measurement_preprocessor = MeasurementPreprocessor(self.config)
        self.measurement_preprocessor.load_params(
            str(self.config.paths.PROCESSED_DIR / 'measurement_preprocessor_params.pkl')
        )
    
    def _find_best_checkpoint(self):
        """Find best checkpoint."""
        checkpoint_dir = Path('checkpoints_resnet18')
        best_files = list(checkpoint_dir.glob('best*.pth'))
        
        if not best_files:
            return str(checkpoint_dir / 'last_checkpoint.pth')
        
        return str(best_files[0])
    
    @torch.no_grad()
    def predict_single(self, mask_path, mask_left_path, height_cm: float) -> Dict[str, float]:
        """Predict for single image."""
        # Load images
        mask_img = self.image_preprocessor.transform(mask_path)
        mask_left_img = self.image_preprocessor.transform(mask_left_path)
        
        # To tensors
        mask_tensor = torch.from_numpy(mask_img).permute(2, 0, 1).unsqueeze(0).float()
        mask_left_tensor = torch.from_numpy(mask_left_img).permute(2, 0, 1).unsqueeze(0).float()
        
        height_normalized = self.measurement_preprocessor.transform_height(height_cm)
        height_tensor = torch.tensor([height_normalized], dtype=torch.float32)
        
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
            'hip_cm': float(predictions_cm[0]),
            'bust_cm': float(predictions_cm[1])
        }
    
    @torch.no_grad()
    def predict_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict for batch."""
        results = []
        
        print(f"🔮 Predicting for {len(data)} samples...")
        
        for _, row in tqdm(data.iterrows(), total=len(data)):
            photo_id = row['photo_id']
            height = row['height_cm']
            
            mask_path = self.config.paths.MASK_DIR / f"{photo_id}.png"
            mask_left_path = self.config.paths.MASK_LEFT_DIR / f"{photo_id}.png"
            
            if not mask_path.exists() or not mask_left_path.exists():
                continue
            
            try:
                preds = self.predict_single(mask_path, mask_left_path, height)
                preds['photo_id'] = photo_id
                results.append(preds)
            except Exception as e:
                print(f"Error {photo_id}: {e}")
                continue
        
        return pd.DataFrame(results)


def main():
    config = Config()
    predictor = HipBustPredictor(config=config)
    
    # Load validation data
    val_data = pd.read_csv(config.paths.PROCESSED_DIR / 'val_data.csv')
    
    # Predict
    results = predictor.predict_batch(val_data.head(10))
    
    print(results)


if __name__ == "__main__":
    main()
