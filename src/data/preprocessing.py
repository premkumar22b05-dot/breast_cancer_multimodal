"""Image preprocessing pipeline for mammography.

Handles DICOM conversion, normalization, background suppression, and augmentation.
"""

import logging
from typing import Dict, Optional

import numpy as np
from scipy import ndimage
from skimage import exposure, measure

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """Preprocessing pipeline for mammography images.
    
    Operations:
    - CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - Normalization (Z-score, Min-Max, Robust)
    - Background suppression (breast region extraction)
    - Local patch extraction
    
    Args:
        config: Configuration dictionary with preprocessing settings
    """

    def __init__(self, config: Dict):
        """Initialize preprocessing pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.image_size = config['data']['image_size']
        self.use_clahe = config['data']['use_clahe']
        self.clahe_clip_limit = config['data']['clahe_clip_limit']
        self.normalize_method = config['data']['normalize_method']
        self.background_threshold = config['data']['background_threshold']

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Apply full preprocessing pipeline.
        
        Args:
            image: Raw image array (H, W)
            
        Returns:
            Preprocessed image (H, W) normalized to [0, 1] or [-1, 1]
        """
        # Step 1: Normalize raw intensity
        image = self._normalize_intensity(image)
        
        # Step 2: Apply CLAHE if enabled
        if self.use_clahe:
            image = self._apply_clahe(image)
        
        # Step 3: Suppress background (pectoral muscle, etc.)
        image = self._suppress_background(image)
        
        # Step 4: Resize to target size
        image = self._resize_image(image)
        
        # Step 5: Normalize for model input
        image = self._normalize_for_model(image)
        
        return image

    def _normalize_intensity(self, image: np.ndarray) -> np.ndarray:
        """Normalize raw pixel values to [0, 1].
        
        Args:
            image: Raw image
            
        Returns:
            Normalized image [0, 1]
        """
        image = image.astype(np.float32)
        
        # Handle potential negative values
        if image.min() < 0:
            image = image - image.min()
        
        # Normalize to [0, 1]
        max_val = image.max()
        if max_val > 0:
            image = image / max_val
        
        return image

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization.
        
        Improves local contrast while limiting noise amplification.
        
        Args:
            image: Image in [0, 1]
            
        Returns:
            CLAHE-enhanced image in [0, 1]
        """
        # Convert to uint8 for skimage
        image_uint8 = (image * 255).astype(np.uint8)
        
        # Apply CLAHE
        try:
            image_clahe = exposure.equalize_adapthist(
                image_uint8,
                kernel_size=self.config['data']['clahe_tile_size'],
                clip_limit=self.clahe_clip_limit / 100.0,
            )
            return image_clahe
        except Exception as e:
            logger.warning(f"CLAHE failed: {e}. Returning original image.")
            return image

    def _suppress_background(self, image: np.ndarray) -> np.ndarray:
        """Suppress background and extract breast region.
        
        Uses simple thresholding to identify breast tissue.
        
        Args:
            image: Image in [0, 1]
            
        Returns:
            Image with background suppressed
        """
        # Threshold to get breast region
        threshold = self.background_threshold
        breast_mask = image > threshold
        
        # Remove small artifacts
        min_size = int(0.01 * image.size)
        labeled = measure.label(breast_mask)
        for region in measure.regionprops(labeled):
            if region.area < min_size:
                breast_mask[labeled == region.label] = False
        
        # Apply mask
        image_masked = image * breast_mask.astype(np.float32)
        
        return image_masked

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target size.
        
        Args:
            image: Image of any size
            
        Returns:
            Image resized to (image_size, image_size)
        """
        from scipy.ndimage import zoom
        
        current_size = image.shape
        if current_size == (self.image_size, self.image_size):
            return image
        
        # Calculate zoom factors
        zoom_factors = (
            self.image_size / current_size[0],
            self.image_size / current_size[1],
        )
        
        # Resize using zoom
        image_resized = zoom(image, zoom_factors, order=1)  # Bilinear interpolation
        
        # Ensure exact size
        if image_resized.shape != (self.image_size, self.image_size):
            image_resized = image_resized[:self.image_size, :self.image_size]
            if image_resized.shape != (self.image_size, self.image_size):
                # Pad if needed
                padded = np.zeros((self.image_size, self.image_size), dtype=image_resized.dtype)
                padded[:image_resized.shape[0], :image_resized.shape[1]] = image_resized
                image_resized = padded
        
        return image_resized

    def _normalize_for_model(self, image: np.ndarray) -> np.ndarray:
        """Normalize image for model input.
        
        Applies z-score, min-max, or robust normalization.
        
        Args:
            image: Image in [0, 1]
            
        Returns:
            Normalized image
        """
        if self.normalize_method == 'z_score':
            # Z-score normalization
            mean = image.mean()
            std = image.std()
            if std > 0:
                image = (image - mean) / std
            else:
                image = image - mean
            
        elif self.normalize_method == 'min_max':
            # Already in [0, 1]
            pass
        
        elif self.normalize_method == 'robust':
            # Robust normalization using percentiles
            q1 = np.percentile(image, 25)
            q3 = np.percentile(image, 75)
            iqr = q3 - q1
            if iqr > 0:
                image = (image - q1) / iqr
            else:
                image = image - q1
        
        return image.astype(np.float32)

    def get_local_patches(self, image: np.ndarray, num_patches: int = 4) -> np.ndarray:
        """Extract local patches from image.
        
        Useful for local texture analysis.
        
        Args:
            image: Image (H, W)
            num_patches: Number of patches to extract
            
        Returns:
            Patches (num_patches, patch_size, patch_size)
        """
        patch_size = self.config['data']['local_patch_size']
        h, w = image.shape
        
        patches = []
        
        # Extract patches in a grid
        stride = (h - patch_size) // int(np.sqrt(num_patches))
        
        for i in range(0, h - patch_size, stride):
            for j in range(0, w - patch_size, stride):
                patch = image[i:i+patch_size, j:j+patch_size]
                patches.append(patch)
                if len(patches) >= num_patches:
                    break
            if len(patches) >= num_patches:
                break
        
        # Pad if needed
        while len(patches) < num_patches:
            patches.append(np.zeros((patch_size, patch_size), dtype=image.dtype))
        
        return np.array(patches[:num_patches])
