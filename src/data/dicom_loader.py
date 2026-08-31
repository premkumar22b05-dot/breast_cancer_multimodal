"""DICOM loading utilities for mammography images.

Handles loading, validation, and metadata extraction from DICOM files.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import pydicom
except ImportError:
    pydicom = None

logger = logging.getLogger(__name__)


class DICOMLoader:
    """Loads and validates DICOM files.
    
    Args:
        config: Configuration dictionary
    """

    def __init__(self, config: Dict):
        """Initialize DICOM loader.
        
        Args:
            config: Configuration dictionary with data paths
        """
        self.config = config
        self.raw_dir = Path(config['data']['raw_dir'])
        
        if pydicom is None:
            logger.warning("pydicom not available. Install with: pip install pydicom")

    def load_dicom(self, dicom_path: str) -> np.ndarray:
        """Load pixel data from DICOM file.
        
        Args:
            dicom_path: Path to DICOM file
            
        Returns:
            Pixel data as numpy array (H, W)
        """
        if not Path(dicom_path).exists():
            raise FileNotFoundError(f"DICOM file not found: {dicom_path}")
        
        if pydicom is None:
            raise RuntimeError("pydicom not installed")
        
        try:
            ds = pydicom.dcmread(dicom_path)
            pixel_data = ds.pixel_array
            
            # Convert to float and normalize
            pixel_data = pixel_data.astype(np.float32)
            
            # Flip if necessary (depends on acquisition)
            if ds.PhotometricInterpretation == 'MONOCHROME1':
                pixel_data = np.max(pixel_data) - pixel_data
            
            return pixel_data
            
        except Exception as e:
            logger.error(f"Error loading DICOM {dicom_path}: {e}")
            raise

    def get_dicom_metadata(self, dicom_path: str) -> Dict:
        """Extract metadata from DICOM file.
        
        Args:
            dicom_path: Path to DICOM file
            
        Returns:
            Dictionary with metadata
        """
        if pydicom is None:
            return {}
        
        try:
            ds = pydicom.dcmread(dicom_path)
            
            metadata = {
                'patient_id': str(ds.PatientID) if hasattr(ds, 'PatientID') else 'Unknown',
                'patient_age': int(ds.PatientAge[:-1]) if hasattr(ds, 'PatientAge') else None,
                'modality': str(ds.Modality) if hasattr(ds, 'Modality') else 'MG',
                'manufacturer': str(ds.Manufacturer) if hasattr(ds, 'Manufacturer') else 'Unknown',
                'pixel_spacing': ds.PixelSpacing if hasattr(ds, 'PixelSpacing') else None,
                'imager_pixel_spacing': ds.ImagerPixelSpacing if hasattr(ds, 'ImagerPixelSpacing') else None,
            }
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Could not extract full metadata from {dicom_path}: {e}")
            return {}

    def extract_metadata_from_dicoms(self) -> pd.DataFrame:
        """Extract metadata from all DICOM files in raw directory.
        
        Returns:
            DataFrame with image metadata
        """
        if pydicom is None:
            logger.error("pydicom required for DICOM metadata extraction")
            return pd.DataFrame()
        
        dicom_dir = self.raw_dir / 'images'
        if not dicom_dir.exists():
            logger.warning(f"DICOM directory not found: {dicom_dir}")
            return pd.DataFrame()
        
        metadata_list = []
        
        for dicom_file in dicom_dir.glob('**/*.dcm'):
            try:
                meta = self.get_dicom_metadata(str(dicom_file))
                meta['dicom_path'] = str(dicom_file.relative_to(self.raw_dir))
                metadata_list.append(meta)
            except Exception as e:
                logger.warning(f"Error processing {dicom_file}: {e}")
        
        if not metadata_list:
            logger.warning("No DICOM metadata extracted. Check directory structure.")
            return pd.DataFrame()
        
        return pd.DataFrame(metadata_list)

    def validate_dicom(self, dicom_path: str) -> Tuple[bool, str]:
        """Validate DICOM file integrity.
        
        Args:
            dicom_path: Path to DICOM file
            
        Returns:
            Tuple of (is_valid, message)
        """
        if not Path(dicom_path).exists():
            return False, "File does not exist"
        
        if pydicom is None:
            return True, "pydicom not available for validation"
        
        try:
            ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
            
            # Check required fields
            if not hasattr(ds, 'PatientID'):
                return False, "Missing PatientID"
            
            if ds.Modality != 'MG':
                return False, f"Invalid modality: {ds.Modality} (expected MG)"
            
            return True, "Valid mammography DICOM"
            
        except Exception as e:
            return False, f"DICOM read error: {str(e)}"
