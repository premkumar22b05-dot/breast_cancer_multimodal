"""Train/val/test splitting with patient-level stratification.

Ensures no patient appears in multiple splits (prevents data leakage).
"""

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataSplitter:
    """Creates patient-level train/val/test splits.
    
    Ensures:
    - No patient appears in multiple splits
    - Class balance across splits (stratification)
    - Reproducible splits (fixed seed)
    
    Args:
        config: Configuration dictionary
        metadata: DataFrame with image metadata
    """

    def __init__(self, config: Dict, metadata: pd.DataFrame):
        """Initialize data splitter.
        
        Args:
            config: Configuration dictionary with split settings
            metadata: Image metadata DataFrame
        """
        self.config = config
        self.metadata = metadata
        self.seed = config['reproducibility']['seed']
        self.splits_dir = Path(config['data']['splits_dir'])
        self.splits_dir.mkdir(parents=True, exist_ok=True)

    def create_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create train/val/test splits at patient level.
        
        Returns:
            Tuple of (train_patients, val_patients, test_patients)
        """
        # Get unique patients with their labels
        patient_data = self._aggregate_patient_labels()
        
        # Get train ratio
        train_ratio = self.config['data_split']['train_ratio']
        val_ratio = self.config['data_split']['val_ratio']
        test_ratio = self.config['data_split']['test_ratio']
        
        # First split: train + temp (val+test)
        train_patients, temp_patients = train_test_split(
            patient_data,
            train_size=train_ratio,
            test_size=val_ratio + test_ratio,
            random_state=self.seed,
            stratify=patient_data[['target']],
        )
        
        # Second split: val and test
        val_size = val_ratio / (val_ratio + test_ratio)
        val_patients, test_patients = train_test_split(
            temp_patients,
            train_size=val_size,
            test_size=1 - val_size,
            random_state=self.seed,
            stratify=temp_patients[['target']],
        )
        
        logger.info(
            f"Created splits:\n"
            f"  Train: {len(train_patients)} patients ({train_ratio*100:.1f}%)\n"
            f"  Val:   {len(val_patients)} patients ({val_ratio*100:.1f}%)\n"
            f"  Test:  {len(test_patients)} patients ({test_ratio*100:.1f}%)"
        )
        
        # Save splits
        self._save_splits(train_patients, val_patients, test_patients)
        
        # Verify no leakage
        self._verify_no_leakage(train_patients, val_patients, test_patients)
        
        return train_patients, val_patients, test_patients

    def _aggregate_patient_labels(self) -> pd.DataFrame:
        """Aggregate labels at patient level.
        
        For patients with multiple exams, use the maximum label (any positive = positive).
        
        Returns:
            DataFrame with one row per unique patient
        """
        patient_data = self.metadata.groupby('patient_id').agg({
            'target': 'max',  # If any exam is positive, patient is positive
            'age': 'mean',
            'breast_density': 'first',
        }).reset_index()
        
        logger.info(f"Aggregated to {len(patient_data)} unique patients")
        
        return patient_data

    def _save_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """Save splits to text files.
        
        Args:
            train_df: Train patient data
            val_df: Val patient data
            test_df: Test patient data
        """
        for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            split_file = self.splits_dir / f"{split_name}_patients.txt"
            split_df['patient_id'].to_csv(split_file, index=False, header=False)
            logger.info(f"Saved {split_name} split to {split_file}")

    def _verify_no_leakage(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """Verify no patient appears in multiple splits.
        
        Args:
            train_df: Train patient data
            val_df: Val patient data
            test_df: Test patient data
        """
        train_set = set(train_df['patient_id'])
        val_set = set(val_df['patient_id'])
        test_set = set(test_df['patient_id'])
        
        train_val_overlap = train_set & val_set
        train_test_overlap = train_set & test_set
        val_test_overlap = val_set & test_set
        
        if train_val_overlap or train_test_overlap or val_test_overlap:
            logger.error(
                f"DATA LEAKAGE DETECTED:\n"
                f"  train ∩ val:  {len(train_val_overlap)} patients\n"
                f"  train ∩ test: {len(train_test_overlap)} patients\n"
                f"  val ∩ test:   {len(val_test_overlap)} patients"
            )
            raise ValueError("Data leakage detected in splits")
        else:
            logger.info(
                f"✓ No data leakage detected\n"
                f"  train ∩ val:  0\n"
                f"  train ∩ test: 0\n"
                f"  val ∩ test:   0"
            )

    def verify_split_integrity(self) -> bool:
        """Verify that all patients are assigned to exactly one split.
        
        Returns:
            True if splits are valid
        """
        # Load split files
        train_patients = set()
        val_patients = set()
        test_patients = set()
        
        for split_name, split_set in [('train', train_patients), ('val', val_patients), ('test', test_patients)]:
            split_file = self.splits_dir / f"{split_name}_patients.txt"
            if split_file.exists():
                with open(split_file) as f:
                    split_set.update(line.strip() for line in f if line.strip())
        
        # Check coverage
        all_unique_patients = set(self.metadata['patient_id'].unique())
        assigned_patients = train_patients | val_patients | test_patients
        
        missing = all_unique_patients - assigned_patients
        if missing:
            logger.warning(f"Missing assignment for {len(missing)} patients")
            return False
        
        logger.info(f"✓ All {len(all_unique_patients)} patients assigned to splits")
        return True
