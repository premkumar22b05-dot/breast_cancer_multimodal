"""Stacking Ensemble with out-of-fold meta-learning.

Combines multiple base learners with a meta-learner for final prediction.
Uses out-of-fold training to prevent data leakage.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

logger = logging.getLogger(__name__)


class StackingEnsemble:
    """Stacking ensemble combining multiple base models.
    
    Architecture:
    1. Split data into K folds
    2. For each fold:
       - Train base learners on K-1 folds
       - Predict on held-out fold and test set
    3. Use out-of-fold predictions as features for meta-learner
    4. Train meta-learner on out-of-fold predictions
    5. Final prediction: meta-learner(test predictions)
    
    Args:
        config: Configuration dictionary
    """

    def __init__(self, config: Dict):
        """Initialize Stacking Ensemble.
        
        Args:
            config: Configuration dictionary with ensemble settings
        """
        self.config = config
        self.cv_folds = config['ensemble']['cv_folds']
        self.random_state = config['reproducibility']['seed']
        
        # Base learners
        self.base_models = config['ensemble']['base_models']
        self.base_learners = {}
        self.meta_learner = None
        
        logger.info(
            f"StackingEnsemble: {len(self.base_models)} base learners, "
            f"cv_folds={self.cv_folds}"
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """Train stacking ensemble with out-of-fold predictions.
        
        Args:
            X: Feature matrix (N, num_features)
            y: Target labels (N,)
        """
        logger.info(f"Training ensemble on {X.shape[0]} samples")
        
        # Initialize out-of-fold predictions
        oof_predictions = np.zeros((X.shape[0], len(self.base_models)))
        
        # Stratified K-fold
        skf = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state,
        )
        
        # Train base learners
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            logger.info(f"Fold {fold_idx + 1}/{self.cv_folds}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train each base learner
            for model_idx, model_name in enumerate(self.base_models):
                model = self._get_base_learner(model_name, fold_idx)
                
                # Train
                model.fit(X_train, y_train)
                
                # Predict on validation fold
                if hasattr(model, 'predict_proba'):
                    oof_predictions[val_idx, model_idx] = model.predict_proba(X_val)[:, 1]
                else:
                    oof_predictions[val_idx, model_idx] = model.predict(X_val)
                
                # Store model
                key = f"{model_name}_fold_{fold_idx}"
                self.base_learners[key] = model
        
        logger.info(f"Out-of-fold predictions shape: {oof_predictions.shape}")
        logger.info(f"OOF prediction stats: mean={oof_predictions.mean():.4f}, "
                   f"std={oof_predictions.std():.4f}")
        
        # Train meta-learner on OOF predictions
        logger.info("Training meta-learner...")
        self.meta_learner = self._get_meta_learner()
        self.meta_learner.fit(oof_predictions, y)
        
        logger.info("Ensemble training complete")

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Predict using stacking ensemble.
        
        Args:
            X: Feature matrix (N, num_features)
            
        Returns:
            Predictions (N,) in [0, 1]
        """
        # Get predictions from base learners (average across folds)
        base_predictions = np.zeros((X.shape[0], len(self.base_models)))
        
        # For each base model, average predictions across all folds
        for model_idx, model_name in enumerate(self.base_models):
            fold_predictions = []
            
            for fold_idx in range(self.cv_folds):
                key = f"{model_name}_fold_{fold_idx}"
                if key in self.base_learners:
                    model = self.base_learners[key]
                    
                    if hasattr(model, 'predict_proba'):
                        pred = model.predict_proba(X)[:, 1]
                    else:
                        pred = model.predict(X)
                    
                    fold_predictions.append(pred)
            
            # Average across folds
            if fold_predictions:
                base_predictions[:, model_idx] = np.mean(fold_predictions, axis=0)
        
        # Meta-learner prediction
        if hasattr(self.meta_learner, 'predict_proba'):
            final_predictions = self.meta_learner.predict_proba(base_predictions)[:, 1]
        else:
            final_predictions = self.meta_learner.predict(base_predictions)
        
        return final_predictions

    def _get_base_learner(self, model_name: str, fold_idx: int):
        """Create base learner instance.
        
        Args:
            model_name: Name of base learner
            fold_idx: Fold index (for reproducibility)
            
        Returns:
            Base learner model
        """
        seed = self.random_state + fold_idx
        
        if model_name == 'xgboost':
            return xgb.XGBClassifier(
                n_estimators=self.config['ensemble'].get('xgb_n_estimators', 100),
                max_depth=self.config['ensemble'].get('xgb_max_depth', 5),
                learning_rate=self.config['ensemble'].get('xgb_learning_rate', 0.1),
                random_state=seed,
                n_jobs=-1,
                verbosity=0,
            )
        
        elif model_name == 'mlp':
            return MLPClassifier(
                hidden_layer_sizes=(64, 32),
                learning_rate_init=self.config['ensemble'].get('mlp_learning_rate', 0.001),
                max_iter=self.config['ensemble'].get('mlp_max_iter', 200),
                random_state=seed,
            )
        
        elif model_name == 'logistic_regression':
            return LogisticRegression(
                max_iter=1000,
                random_state=seed,
            )
        
        else:
            logger.warning(f"Unknown base learner: {model_name}. Using LogisticRegression.")
            return LogisticRegression(max_iter=1000, random_state=seed)

    def _get_meta_learner(self):
        """Create meta-learner.
        
        Returns:
            Meta-learner model
        """
        meta_learner_name = self.config['ensemble'].get('meta_learner', 'xgboost')
        
        if meta_learner_name == 'xgboost':
            return xgb.XGBClassifier(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=0,
            )
        
        elif meta_learner_name == 'logistic_regression':
            return LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
            )
        
        else:
            logger.warning(f"Unknown meta-learner: {meta_learner_name}. Using LogisticRegression.")
            return LogisticRegression(max_iter=1000, random_state=self.random_state)
