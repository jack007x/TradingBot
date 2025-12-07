"""
Model Trainer Module
=====================
Handles model training with time-series cross-validation.

Usage:
    from models import ModelTrainer
    from config import get_config

    config = get_config()
    trainer = ModelTrainer(config)

    # Train with time-series CV
    model, metrics = trainer.train_with_cv(X, y, feature_names)

    # Train simple split
    model, metrics = trainer.train_simple(X, y, feature_names)
"""

import logging
from datetime import datetime
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from config.config_loader import Config
from models.model_baseline import BaselineModel

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Trains ML models with proper time-series methodology.

    Features:
    - Time-series cross-validation (no future data leakage)
    - Gap between train and validation to prevent leakage
    - Metric aggregation across folds
    - Hyperparameter validation
    """

    def __init__(self, config: Config):
        """
        Initialize trainer.

        Args:
            config: Configuration object
        """
        self.config = config
        self.n_splits = config.training.cv_splits
        self.val_split = config.training.validation_split
        self.gap_bars = config.training.gap_bars

    def train_with_cv(
            self,
            X: np.ndarray,
            y: np.ndarray,
            feature_names: Optional[List[str]] = None
    ) -> Tuple[BaselineModel, Dict[str, Any]]:
        """
        Train model using time-series cross-validation.

        The final model is trained on all data, while CV metrics
        are used to estimate generalization performance.

        Args:
            X: Feature matrix
            y: Labels
            feature_names: Feature names

        Returns:
            Tuple[model, cv_metrics]: Trained model and CV metrics
        """
        logger.info(f"Training with {self.n_splits}-fold time-series CV")
        logger.info(f"Data: {X.shape[0]} samples, {X.shape[1]} features")

        # Time series split with gap
        tscv = TimeSeriesSplit(
            n_splits=self.n_splits,
            gap=self.gap_bars
        )

        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            logger.info(f"Fold {fold + 1}/{self.n_splits}: train={len(train_idx)}, val={len(val_idx)}")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Train model for this fold
            model = BaselineModel(self.config)
            model.fit(
                X_train, y_train,
                X_val, y_val,
                feature_names=feature_names
            )

            # Evaluate on validation set
            metrics = self._evaluate_fold(model, X_val, y_val)
            metrics['fold'] = fold + 1
            metrics['train_size'] = len(train_idx)
            metrics['val_size'] = len(val_idx)
            fold_metrics.append(metrics)

            logger.info(f"  Fold {fold + 1} - Accuracy: {metrics['accuracy']:.4f}, "
                        f"Up Precision: {metrics.get('precision_up', 0):.4f}")

        # Aggregate CV metrics
        cv_metrics = self._aggregate_cv_metrics(fold_metrics)
        cv_metrics['fold_details'] = fold_metrics

        # Train final model on all data
        logger.info("Training final model on all data...")

        # Use last portion as validation for early stopping
        split_idx = int(len(X) * (1 - self.val_split))
        X_train_final = X[:split_idx]
        y_train_final = y[:split_idx]
        X_val_final = X[split_idx:]
        y_val_final = y[split_idx:]

        final_model = BaselineModel(self.config)
        final_model.fit(
            X_train_final, y_train_final,
            X_val_final, y_val_final,
            feature_names=feature_names
        )

        # Add training metadata
        cv_metrics['training_date'] = datetime.now().isoformat()
        cv_metrics['total_samples'] = len(X)
        cv_metrics['n_features'] = X.shape[1]
        cv_metrics['feature_names'] = feature_names

        logger.info(f"CV Results - Mean Accuracy: {cv_metrics['mean_accuracy']:.4f} "
                    f"(+/- {cv_metrics['std_accuracy']:.4f})")

        return final_model, cv_metrics

    def train_simple(
            self,
            X: np.ndarray,
            y: np.ndarray,
            feature_names: Optional[List[str]] = None,
            val_split: Optional[float] = None
    ) -> Tuple[BaselineModel, Dict[str, Any]]:
        """
        Train model with simple train/validation split.

        Args:
            X: Feature matrix
            y: Labels
            feature_names: Feature names
            val_split: Validation split ratio

        Returns:
            Tuple[model, metrics]: Trained model and metrics
        """
        if val_split is None:
            val_split = self.val_split

        logger.info(f"Training with {(1-val_split)*100:.0f}/{val_split*100:.0f} split")

        # Time-ordered split
        split_idx = int(len(X) * (1 - val_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        logger.info(f"Train: {len(X_train)}, Validation: {len(X_val)}")

        # Train model
        model = BaselineModel(self.config)
        model.fit(
            X_train, y_train,
            X_val, y_val,
            feature_names=feature_names
        )

        # Evaluate
        metrics = self._evaluate_fold(model, X_val, y_val)
        metrics['training_date'] = datetime.now().isoformat()
        metrics['train_size'] = len(X_train)
        metrics['val_size'] = len(X_val)
        metrics['n_features'] = X.shape[1]
        metrics['feature_names'] = feature_names

        logger.info(f"Validation Accuracy: {metrics['accuracy']:.4f}")

        return model, metrics

    def _evaluate_fold(
            self,
            model: BaselineModel,
            X: np.ndarray,
            y: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model on a single fold/split.

        Returns classification metrics.
        """
        y_pred = model.predict(X)
        proba = model.predict_proba(X)

        metrics = {}

        # Overall accuracy
        metrics['accuracy'] = (y_pred == y).mean()

        # Per-class metrics
        for label, name in [(-1, 'down'), (0, 'neutral'), (1, 'up')]:
            # Precision: TP / (TP + FP)
            mask_pred = y_pred == label
            if mask_pred.sum() > 0:
                precision = (y[mask_pred] == label).mean()
            else:
                precision = 0.0
            metrics[f'precision_{name}'] = precision

            # Recall: TP / (TP + FN)
            mask_true = y == label
            if mask_true.sum() > 0:
                recall = (y_pred[mask_true] == label).mean()
            else:
                recall = 0.0
            metrics[f'recall_{name}'] = recall

            # F1 score
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            metrics[f'f1_{name}'] = f1

            # Class support (actual count)
            metrics[f'support_{name}'] = mask_true.sum()

        # Average probability for correct predictions
        correct_mask = y_pred == y
        if correct_mask.sum() > 0:
            correct_proba = proba[np.arange(len(y)), (y + 1).astype(int)]
            metrics['avg_correct_proba'] = correct_proba[correct_mask].mean()
        else:
            metrics['avg_correct_proba'] = 0.0

        # Directional accuracy (ignoring neutral predictions)
        non_neutral_mask = y_pred != 0
        if non_neutral_mask.sum() > 0:
            metrics['directional_accuracy'] = (y_pred[non_neutral_mask] == y[non_neutral_mask]).mean()
        else:
            metrics['directional_accuracy'] = 0.0

        return metrics

    def _aggregate_cv_metrics(
            self,
            fold_metrics: List[Dict[str, float]]
    ) -> Dict[str, Any]:
        """Aggregate metrics across CV folds."""
        aggregated = {}

        # Numeric keys to aggregate
        numeric_keys = [k for k in fold_metrics[0].keys()
                        if isinstance(fold_metrics[0][k], (int, float))
                        and k not in ['fold', 'train_size', 'val_size']]

        for key in numeric_keys:
            values = [m[key] for m in fold_metrics]
            aggregated[f'mean_{key}'] = np.mean(values)
            aggregated[f'std_{key}'] = np.std(values)
            aggregated[f'min_{key}'] = np.min(values)
            aggregated[f'max_{key}'] = np.max(values)

        return aggregated


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
    trainer = ModelTrainer(config)

    # Fetch and prepare data
    from datetime import datetime
    start = datetime(2022, 1, 1)
    end = datetime(2024, 6, 1)
    df = fetcher.fetch_historical(start, end)
    df_features = engineer.build_features(df)
    X, y, feature_names = engineer.prepare_training_data(df_features)

    print(f"Data prepared: {X.shape}")

    # Train with CV
    print("\n" + "=" * 60)
    print("Training with Time-Series CV")
    print("=" * 60)
    model, cv_metrics = trainer.train_with_cv(X, y, feature_names)

    print(f"\nCV Results:")
    print(f"  Mean Accuracy: {cv_metrics['mean_accuracy']:.4f} (+/- {cv_metrics['std_accuracy']:.4f})")
    print(f"  Mean Directional Accuracy: {cv_metrics['mean_directional_accuracy']:.4f}")
    print(f"  Mean Up Precision: {cv_metrics['mean_precision_up']:.4f}")
    print(f"  Mean Down Precision: {cv_metrics['mean_precision_down']:.4f}")

    # Show top features
    print("\nTop 10 Features:")
    for name, imp in model.get_top_features(10):
        print(f"  {name}: {imp:.4f}")
