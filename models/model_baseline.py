"""
Baseline Model Module
======================
LightGBM-based classifier for price direction prediction.

Usage:
    from models import BaselineModel
    from config import get_config

    config = get_config()
    model = BaselineModel(config)

    # Train
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
"""

import logging
from typing import Optional, Dict, Any, Tuple
import numpy as np
import lightgbm as lgb

from config.config_loader import Config

logger = logging.getLogger(__name__)


class BaselineModel:
    """
    LightGBM classifier for 3-class price direction prediction.

    Classes:
        -1: Down (price decrease > threshold)
         0: Neutral (small price change)
         1: Up (price increase > threshold)
    """

    def __init__(
            self,
            config: Config,
            params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize baseline model.

        Args:
            config: Configuration object
            params: Optional custom LightGBM parameters
        """
        self.config = config

        # Get parameters from config or use provided
        if params is not None:
            self.params = params
        else:
            self.params = dict(config.model.lgbm_params)

        # Model instance
        self._model: Optional[lgb.LGBMClassifier] = None
        self._feature_names: Optional[list] = None
        self._is_fitted = False

        # Training metadata
        self.training_info: Dict[str, Any] = {}

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            feature_names: Optional[list] = None,
            early_stopping_rounds: Optional[int] = None
    ) -> 'BaselineModel':
        """
        Train the model.

        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            feature_names: Feature names for interpretability
            early_stopping_rounds: Early stopping patience

        Returns:
            self: Trained model
        """
        logger.info(f"Training LightGBM model: X shape {X.shape}")

        # Map labels from {-1, 0, 1} to {0, 1, 2} for LightGBM
        y_mapped = y + 1  # {-1, 0, 1} -> {0, 1, 2}

        if y_val is not None:
            y_val_mapped = y_val + 1

        # Store feature names
        self._feature_names = feature_names

        # Create model with parameters
        self._model = lgb.LGBMClassifier(**self.params)

        # Prepare callbacks
        callbacks = []

        if early_stopping_rounds is None:
            early_stopping_rounds = self.config.training.early_stopping_rounds

        if X_val is not None and y_val is not None:
            callbacks.append(
                lgb.early_stopping(early_stopping_rounds, verbose=False)
            )
            callbacks.append(lgb.log_evaluation(period=100))

            # Fit with validation
            self._model.fit(
                X, y_mapped,
                eval_set=[(X_val, y_val_mapped)],
                eval_names=['validation'],
                callbacks=callbacks
            )
        else:
            # Fit without validation
            self._model.fit(X, y_mapped)

        self._is_fitted = True

        # Store training info
        self.training_info = {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_classes': 3,
            'best_iteration': self._model.best_iteration_ if hasattr(self._model, 'best_iteration_') else self.params.get('n_estimators'),
            'feature_names': feature_names,
        }

        logger.info(f"Model trained. Best iteration: {self.training_info['best_iteration']}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            np.ndarray: Predicted labels {-1, 0, 1}
        """
        self._check_fitted()

        # LightGBM predicts {0, 1, 2}, map back to {-1, 0, 1}
        y_pred = self._model.predict(X)
        return y_pred - 1

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            np.ndarray: Probabilities (n_samples, 3) for classes {-1, 0, 1}
                        Column order: [P(down), P(neutral), P(up)]
        """
        self._check_fitted()

        return self._model.predict_proba(X)

    def predict_direction_with_confidence(
            self,
            X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict direction with confidence scores.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Tuple[direction, confidence]:
                direction: -1 (down), 0 (neutral), 1 (up)
                confidence: Probability of predicted class
        """
        proba = self.predict_proba(X)
        direction = self.predict(X)
        confidence = np.max(proba, axis=1)

        return direction, confidence

    def get_feature_importance(
            self,
            importance_type: str = 'gain'
    ) -> Dict[str, float]:
        """
        Get feature importance scores.

        Args:
            importance_type: 'gain', 'split', or 'cover'

        Returns:
            Dict[str, float]: Feature name to importance mapping
        """
        self._check_fitted()

        importances = self._model.feature_importances_

        if self._feature_names is not None:
            return dict(zip(self._feature_names, importances))
        else:
            return {f'feature_{i}': imp for i, imp in enumerate(importances)}

    def get_top_features(self, top_n: int = 20) -> list:
        """
        Get top N most important features.

        Args:
            top_n: Number of features to return

        Returns:
            list: List of (feature_name, importance) tuples
        """
        importance = self.get_feature_importance()
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:top_n]

    def get_model(self) -> lgb.LGBMClassifier:
        """Get underlying LightGBM model."""
        self._check_fitted()
        return self._model

    def is_fitted(self) -> bool:
        """Check if model is trained."""
        return self._is_fitted

    def _check_fitted(self):
        """Raise error if model not fitted."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from config import get_config
    from data.data_fetcher import DataFetcher
    from features.feature_engineering import FeatureEngineer

    config = get_config()
    fetcher = DataFetcher(config)
    engineer = FeatureEngineer(config)

    # Fetch and prepare data
    from datetime import datetime
    start = datetime(2023, 1, 1)
    end = datetime(2024, 6, 1)
    df = fetcher.fetch_historical(start, end)
    df_features = engineer.build_features(df)
    X, y, feature_names = engineer.prepare_training_data(df_features)

    # Split train/test (time-based)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Train model
    model = BaselineModel(config)
    model.fit(X_train, y_train, X_test, y_test, feature_names)

    # Predict
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)

    # Accuracy
    accuracy = (y_pred == y_test).mean()
    print(f"\nTest Accuracy: {accuracy:.4f}")

    # Class distribution
    print("\nPrediction distribution:")
    for label in [-1, 0, 1]:
        count = (y_pred == label).sum()
        print(f"  {label}: {count} ({count/len(y_pred)*100:.1f}%)")

    # Top features
    print("\nTop 10 Features:")
    for name, importance in model.get_top_features(10):
        print(f"  {name}: {importance:.4f}")
