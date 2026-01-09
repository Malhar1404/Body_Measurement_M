"""
image_preprocessor.py - Image Preprocessing for Silhouette Data

This module handles loading, resizing, and normalizing silhouette images
from the mask and mask_left directories.
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, Dict, Union
import logging
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.base_classes import BasePreprocessor

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class ImagePreprocessor(BasePreprocessor):
    """
    Preprocessor for silhouette images.
    
    Handles:
    - Loading images from mask and mask_left directories
    - Resizing to 512×683 (maintains 3:4 aspect ratio)
    - Normalization using ImageNet statistics
    - Converting to numpy arrays for ResNet input
    """
    
    def __init__(self, config):
        """Initialize image preprocessor."""
        super().__init__(config)
        self.logger = logger
        self.target_size = config.image.TARGET_SIZE  # (512, 683)
        self.normalize = config.image.NORMALIZE
        self.mean = np.array(config.image.PIXEL_MEAN)  # ImageNet means
        self.std = np.array(config.image.PIXEL_STD)    # ImageNet stds
    
    def fit(self, data=None) -> 'ImagePreprocessor':
        """
        Fit the preprocessor.
        
        For images with ImageNet normalization, this just sets up parameters.
        No learning needed since we use predefined ImageNet statistics.
        """
        self._params = {
            'target_size': self.target_size,
            'mean': self.mean,
            'std': self.std,
            'normalize': self.normalize
        }
        self.is_fitted = True
        self.logger.info("✓ Image preprocessor fitted with ImageNet parameters")
        return self
    
    def transform(self, image: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Transform a single image.
        
        Steps:
        1. Load image (if path provided)
        2. Convert to RGB
        3. Resize to 512×683
        4. Convert to numpy array
        5. Normalize with ImageNet stats
        
        Args:
            image: Image path, PIL Image, or numpy array
            
        Returns:
            Preprocessed image as numpy array of shape (683, 512, 3)
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        
        # Step 1: Load image if path provided
        if isinstance(image, (str, Path)):
            image = self._load_image(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        # Step 2: Convert to RGB (silhouettes might be grayscale/RGBA)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Step 3: Resize to target dimensions
        image = self._resize_image(image)
        
        # Step 4: Convert to numpy array [H, W, C]
        img_array = np.array(image, dtype=np.float32)
        
        # Step 5: Normalize
        if self.normalize:
            img_array = self._normalize_image(img_array)
        else:
            # Just scale to [0, 1]
            img_array = img_array / 255.0
        
        return img_array
    
    def transform_pair(self, mask_path: Union[str, Path], 
                      mask_left_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform a pair of images (mask and mask_left).
        
        Args:
            mask_path: Path to front view silhouette
            mask_left_path: Path to side view silhouette
            
        Returns:
            Tuple of (front_image, side_image) as numpy arrays
        """
        front = self.transform(mask_path)
        side = self.transform(mask_left_path)
        return front, side
    
    def _load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """Load image from disk."""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            image = Image.open(image_path)
            return image
        except Exception as e:
            raise RuntimeError(f"Error loading image {image_path}: {e}")
    
    def _resize_image(self, image: Image.Image) -> Image.Image:
        """
        Resize image to target dimensions using high-quality LANCZOS resampling.
        
        Args:
            image: PIL Image to resize
            
        Returns:
            Resized PIL Image (512×683)
        """
        target_width, target_height = self.target_size
        
        # PIL resize expects (width, height)
        resized = image.resize(
            (target_width, target_height),
            resample=Image.Resampling.LANCZOS
        )
        
        return resized
    
    def _normalize_image(self, img_array: np.ndarray) -> np.ndarray:
        """
        Normalize image using ImageNet mean and std.
        
        Formula: (pixel / 255 - mean) / std
        
        This prepares images for ResNet pretrained on ImageNet.
        """
        # Scale to [0, 1]
        img_array = img_array / 255.0
        
        # Apply ImageNet normalization
        img_array = (img_array - self.mean) / self.std
        
        return img_array
    
    def denormalize_image(self, img_array: np.ndarray) -> np.ndarray:
        """
        Reverse normalization for visualization.
        
        Args:
            img_array: Normalized image array
            
        Returns:
            Image array in range [0, 255] as uint8
        """
        if not self.normalize:
            return (img_array * 255).astype(np.uint8)
        
        # Reverse normalization
        img_array = (img_array * self.std) + self.mean
        img_array = img_array * 255
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        
        return img_array
    
    def get_image_paths(self, photo_id: str, 
                       mask_dir: Optional[Path] = None,
                       mask_left_dir: Optional[Path] = None) -> Dict[str, Path]:
        """
        Get paths for both views of a photo.
        
        Args:
            photo_id: Photo identifier (filename without extension)
            mask_dir: Directory containing front view images
            mask_left_dir: Directory containing side view images
            
        Returns:
            Dictionary with 'mask' and 'mask_left' paths
        """
        if mask_dir is None:
            mask_dir = self.config.data.MASK_DIR
        if mask_left_dir is None:
            mask_left_dir = self.config.data.MASK_LEFT_DIR
        
        image_format = self.config.image.IMAGE_FORMAT
        
        paths = {
            'mask': mask_dir / f"{photo_id}{image_format}",
            'mask_left': mask_left_dir / f"{photo_id}{image_format}"
        }
        
        # Check if files exist and try alternative extensions
        for view, path in paths.items():
            if not path.exists():
                for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG']:
                    alt_path = path.with_suffix(ext)
                    if alt_path.exists():
                        paths[view] = alt_path
                        break
                else:
                    self.logger.warning(f"⚠️  Image not found: {path}")
        
        return paths


# Quick test
if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent / 'config'))
    from src.config.config import Config
    
    print("="*80)
    print("Testing ImagePreprocessor")
    print("="*80)
    
    config = Config()
    preprocessor = ImagePreprocessor(config)
    
    # Fit preprocessor
    preprocessor.fit()
    
    print(f"\n✓ Target size: {preprocessor.target_size}")
    print(f"✓ Normalization: {preprocessor.normalize}")
    print(f"✓ ImageNet mean: {preprocessor.mean}")
    print(f"✓ ImageNet std: {preprocessor.std}")
    
    # Try to load a sample image if available
    mask_dir = config.data.MASK_DIR
    if mask_dir.exists():
        sample_images = list(mask_dir.glob('*.png'))[:1]
        if sample_images:
            print(f"\n✓ Testing with sample image: {sample_images[0].name}")
            img = preprocessor.transform(sample_images[0])
            print(f"✓ Processed image shape: {img.shape}")
            print(f"✓ Image value range: [{img.min():.2f}, {img.max():.2f}]")
            print(f"✓ Expected: normalized values around [-2, 2]")
        else:
            print("\n⚠️  No sample images found for testing")
    else:
        print(f"\n⚠️  Mask directory not found: {mask_dir}")
    
    print("\n✅ Image preprocessor test completed successfully!")
