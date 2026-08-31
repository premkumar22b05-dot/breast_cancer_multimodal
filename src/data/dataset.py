"""PyTorch Dataset class for breast cancer detection.

Handles loading mammographic images and clinical data with proper preprocessing.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .preprocessing import PreprocessingPipeline
from .dicom_loader import DICOMLoader

logger = logging.getLogger(__name__)


class BreastCancerDataset(Dataset):
    """Dataset class for mammography and clinical data.
    
    Supports multimodal (imaging + clinical) and image-only modes.
    Handles patient-level organization with multiple views per exam.
    
    Args:
        config: Configuration dictionary
        split: 'train', 'val', or 'test'
        transform: Optional transforms for images
    """

    def __init__(
        self,
        config: Dict,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
    ):
        """Initialize dataset.
        
        Args:
            config: Configuration dictionary with data settings
            split: Dataset split ('train', 'val', 'test')
            transform: Optional image transforms
        """
        self.config = config
        self.split = split
        self.transform = transform
        
        # Initialize components
        self.dicom_loader = DICOMLoader(config)
        self.preprocessor = PreprocessingPipeline(config)
        
        # Load metadata
        self._load_metadata()
        
        # Load split assignments
        self._load_splits()
        
        # Filter to current split
        self._filter_to_split()
        
        # Load clinical data if available
        self.has_clinical = False
        self._load_clinical_data()
        
        logger.info(
            f"Initialized {split} dataset with {len(self)} samples. "
            f"Clinical data: {self.has_clinical}"
        )

    def _load_metadata(self) -> None:
        """Load mammography metadata from DICOM or CSV."""
        metadata_path = Path(self.config['data']['raw_dir']) / 'metadata.csv'
        
        if metadata_path.exists():
            self.metadata = pd.read_csv(metadata_path)
            logger.info(f"Loaded metadata from {metadata_path}")
        else:
            # Auto-extract from DICOM files
            logger.info("Extracting metadata from DICOM files...")
            self.metadata = self.dicom_loader.extract_metadata_from_dicoms()
            
            # Save for future use
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            self.metadata.to_csv(metadata_path, index=False)

    def _load_splits(self) -> None:
        """Load train/val/test split assignments."""
        splits_dir = Path(self.config['data']['splits_dir'])
        
        # Check if splits already exist
        train_split_file = splits_dir / f"{self.split}_patients.txt"
        
        if train_split_file.exists():
            # Load existing splits
            self.patient_ids = set()
            for split_name in ['train', 'val', 'test']:
                split_file = splits_dir / f"{split_name}_patients.txt"
                if split_file.exists():
                    with open(split_file) as f:
                        if split_name == self.split:
                            self.patient_ids = set(line.strip() for line in f)
        else:
            # Create new splits (should be done by DataSplitter)
            logger.warning(f"No split file found for {self.split}. Creating...")
            from .splits import DataSplitter
            splitter = DataSplitter(self.config, self.metadata)
            splitter.create_splits()
            self._load_splits()  # Recursive call

    def _filter_to_split(self) -> None:
        """Filter metadata to current split."""
        self.data = self.metadata[
            self.metadata['patient_id'].isin(self.patient_ids)
        ].reset_index(drop=True)
        
        logger.info(f"Filtered to {len(self.data)} exams for {self.split} split")

    def _load_clinical_data(self) -> None:
        """Load paired clinical data if available."""
        clinical_path = Path(self.config['data']['clinical_dir']) / 'clinical_data.csv'
        
        if not clinical_path.exists():
            logger.info("No clinical data file found. Using image-only mode.")
            self.clinical_data = None
            return
        
        clinical_df = pd.read_csv(clinical_path)
        
        # Merge with image metadata
        self.data = self.data.merge(
            clinical_df,
            on=['patient_id', 'exam_id'],
            how='left'
        )
        
        # Check if clinical data is available for this split
        if self.data[self.config['data']['clinical_features']].isnull().all().any():
            logger.warning("Clinical data has many missing values. Check alignment.")
        else:
            self.has_clinical = True
            logger.info("Clinical data loaded successfully.")
        
        self.clinical_data = clinical_df

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample.
        
        Returns:
            Dictionary with:
                - images: (4, H, W) tensor for 4 views
                - target: binary label
                - metadata: patient/exam identifiers
                - clinical: clinical features (if available)
        """
        row = self.data.iloc[idx]
        
        # Load images for all 4 views
        images = self._load_images(row)
        
        # Get target label
        target = torch.tensor(
            row['target'],
            dtype=torch.float32
        )
        
        # Package metadata
        metadata = {
            'patient_id': row['patient_id'],
            'exam_id': row['exam_id'],
            'age': float(row.get('age', -1)),
            'breast_density': row.get('breast_density', 'Unknown'),
        }
        
        # Package clinical features if available
        clinical = None
        if self.has_clinical:
            clinical = self._get_clinical_features(row)
        
        # Apply transforms if provided
        if self.transform:
            images = self.transform(images)
        
        return {
            'images': images,
            'target': target,
            'metadata': metadata,
            'clinical': clinical,
        }

    def _load_images(self, row: pd.Series) -> torch.Tensor:
        """Load and preprocess images for all 4 views.
        
        Args:
            row: Metadata row
            
        Returns:
            Tensor of shape (4, H, W) for CC-L, CC-R, MLO-L, MLO-R
        """
        views = ['CC_L', 'CC_R', 'MLO_L', 'MLO_R']
        images = []
        
        for view in views:
            # Construct DICOM file path
            dicom_path = (
                Path(self.config['data']['raw_dir']) /
                f"{row['patient_id']}_{row['exam_id']}_{view}.dcm"
            )
            
            if not dicom_path.exists():
                logger.warning(f"DICOM not found: {dicom_path}. Using zeros.")
                img = np.zeros(
                    (self.config['data']['image_size'],
                     self.config['data']['image_size']),
                    dtype=np.float32
                )
            else:
                # Load DICOM
                img = self.dicom_loader.load_dicom(str(dicom_path))
                
                # Preprocess
                img = self.preprocessor.preprocess(img)
            
            images.append(img)
        
        # Stack into (4, H, W)
        images = np.stack(images, axis=0)  # (4, H, W)
        
        # Convert to tensor
        images = torch.from_numpy(images).float()
        
        return images

    def _get_clinical_features(self, row: pd.Series) -> torch.Tensor:
        """Extract clinical features for the sample.
        
        Args:
            row: Metadata row with clinical data
            
        Returns:
            Tensor of clinical features
        """
        clinical_features = []
        
        # Continuous features
        for feat in self.config['data']['continuous_features']:
            val = row.get(feat, 0)
            if pd.isna(val):
                val = 0  # Simple imputation (should use proper train-only imputation)
            clinical_features.append(float(val))
        
        # Categorical features (one-hot encoded)
        for feat in self.config['data']['categorical_features']:
            val = row.get(feat, 'Unknown')
            # Simplified: just return categorical index
            # Proper implementation would use learned embeddings
            clinical_features.append(hash(str(val)) % 10 / 10.0)
        
        return torch.tensor(clinical_features, dtype=torch.float32)


class MultimodalDataLoader:
    """Wrapper for creating data loaders with proper sampling."""
    
    def __init__(self, config: Dict):
        """Initialize data loader factory.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
    
    def get_dataloaders(
        self,
        batch_size: Optional[int] = None,
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Get train/val/test dataloaders.
        
        Args:
            batch_size: Batch size (uses config if None)
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        if batch_size is None:
            batch_size = self.config['training']['batch_size']
        
        # Create datasets
        train_dataset = BreastCancerDataset(self.config, split='train')
        val_dataset = BreastCancerDataset(self.config, split='val')
        test_dataset = BreastCancerDataset(self.config, split='test')
        
        # Create loaders with proper settings
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config['compute']['num_workers'],
            pin_memory=self.config['compute']['pin_memory'],
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config['compute']['num_workers'],
            pin_memory=self.config['compute']['pin_memory'],
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config['compute']['num_workers'],
            pin_memory=self.config['compute']['pin_memory'],
        )
        
        return train_loader, val_loader, test_loader
