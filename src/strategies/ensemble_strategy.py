"""
Performance-Weighted Ensemble Strategy

PROBLEM WITH EQUAL WEIGHTING:
- Not all models perform equally well
- Poor models (dir_acc < 50%, low correlation) hurt ensemble
- Equal weights waste good model predictions

SOLUTION - DYNAMIC PERFORMANCE WEIGHTING:
- Weight models by: (directional_accuracy - 0.5) × correlation × 1000
- Exclude models below thresholds (dir_acc < 50%, corr < 0.03)
- Heavily favor models that are BOTH accurate AND correlated
- Auto-adjusts weights based on validation performance

EXAMPLE WEIGHTING:
Model A: 53% dir_acc, 0.18 corr → weight = (0.53-0.50) * 0.18 * 1000 = 5.4
Model B: 50.5% dir_acc, 0.10 corr → weight = (0.505-0.50) * 0.10 * 1000 = 0.5
Model C: 49% dir_acc, 0.05 corr → weight = 0 (excluded)

Normalized: Model A: 91.5%, Model B: 8.5%, Model C: 0%
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger


class PerformanceWeightedEnsemble:
    """
    Weight models based on their actual validation performance.

    Models are weighted by:
    - Directional accuracy (must be > 50%)
    - Correlation with actual returns (must be > 0.03)
    - Combined score = (dir_acc - 0.5) × correlation × 1000

    This heavily weights models that are both accurate AND well-correlated.
    """

    def __init__(
        self,
        models: Dict[str, Any],
        metrics: Dict[str, Dict[str, float]],
        min_dir_acc: float = 0.50,
        min_correlation: float = 0.03
    ):
        """
        Initialize ensemble with performance-based weighting.

        Args:
            models: Dictionary of model_name -> model_object
            metrics: Dictionary of model_name -> {
                'directional_accuracy': float,
                'correlation': float,
                ... other metrics ...
            }
            min_dir_acc: Minimum directional accuracy to include (default 50%)
            min_correlation: Minimum correlation to include (default 0.03)
        """
        self.models = models
        self.metrics = metrics
        self.min_dir_acc = min_dir_acc
        self.min_correlation = min_correlation

        # Calculate dynamic weights based on performance
        self.weights = self.calculate_dynamic_weights(metrics)

        # Log ensemble configuration
        self._log_ensemble_configuration()

    def calculate_dynamic_weights(
        self,
        metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Calculate weights from multiple performance metrics.

        Scoring formula:
        - edge = directional_accuracy - 0.50 (edge over random)
        - score = edge × correlation × 1000
        - Exclude if dir_acc < min_dir_acc OR correlation < min_correlation

        Args:
            metrics: Performance metrics for each model

        Returns:
            Dictionary of normalized weights (sum to 1.0)
        """
        scores = {}

        logger.info("=" * 70)
        logger.info("CALCULATING ENSEMBLE WEIGHTS")
        logger.info("=" * 70)

        for name, m in metrics.items():
            dir_acc = m.get('directional_accuracy', 0.5)
            corr = m.get('correlation', 0)

            # Check minimum thresholds
            if dir_acc < self.min_dir_acc or corr < self.min_correlation:
                scores[name] = 0.0
                logger.warning(
                    f"❌ {name:12} EXCLUDED: "
                    f"dir_acc={dir_acc:.4f} (min {self.min_dir_acc:.2f}), "
                    f"corr={corr:.4f} (min {self.min_correlation:.2f})"
                )
                continue

            # Calculate score: (accuracy edge) × (correlation) × 1000
            # This heavily weights models that are both accurate AND correlated
            edge = dir_acc - 0.50  # Edge over random (50%)
            score = edge * corr * 1000
            scores[name] = max(0, score)

            logger.info(
                f"✅ {name:12} INCLUDED: "
                f"dir_acc={dir_acc:.4f}, corr={corr:.4f}, "
                f"edge={edge:.4f}, score={score:.3f}"
            )

        # Normalize to sum to 1.0
        total = sum(scores.values())

        if total == 0:
            logger.warning("⚠️  NO MODELS MEET THRESHOLD! Using equal weights as fallback")
            return {name: 1.0 / len(scores) for name in scores}

        weights = {name: score / total for name, score in scores.items()}

        logger.info("=" * 70)
        logger.info("FINAL ENSEMBLE WEIGHTS:")
        for name in sorted(weights.keys(), key=lambda x: weights[x], reverse=True):
            weight = weights[name]
            if weight > 0:
                logger.info(f"  {name:12}: {weight*100:5.1f}%")
            else:
                logger.info(f"  {name:12}:   0.0% (excluded)")
        logger.info("=" * 70)

        return weights

    def predict(
        self,
        X: np.ndarray,
        return_confidence: bool = True
    ) -> Tuple[float, float]:
        """
        Get weighted prediction from ensemble.

        Args:
            X: Input features (single sample or batch)
            return_confidence: Whether to return confidence score

        Returns:
            Tuple of (prediction, confidence) if return_confidence=True
            Otherwise just prediction
        """
        predictions = {}

        # Get prediction from each model
        for name, model in self.models.items():
            weight = self.weights.get(name, 0)

            if weight == 0:
                continue  # Skip excluded models

            # Get prediction
            try:
                pred = model.predict(X) if hasattr(model, 'predict') else model.predict_single(X)
                predictions[name] = float(pred) if not isinstance(pred, float) else pred
            except Exception as e:
                logger.warning(f"Error getting prediction from {name}: {e}")
                continue

        if not predictions:
            logger.error("No valid predictions from any model!")
            return (0.0, 0.0) if return_confidence else 0.0

        # Weighted average
        final_pred = sum(
            pred * self.weights[name]
            for name, pred in predictions.items()
        )

        if return_confidence:
            # Calculate ensemble confidence
            confidence = self.calculate_confidence(predictions)
            return final_pred, confidence
        else:
            return final_pred

    def calculate_confidence(
        self,
        predictions: Dict[str, float]
    ) -> float:
        """
        Calculate ensemble confidence based on model agreement.

        High confidence = models agree (low std deviation)
        Low confidence = models disagree (high std deviation)

        Args:
            predictions: Dictionary of model_name -> prediction

        Returns:
            Confidence score between 0 and 1
        """
        if len(predictions) < 2:
            return 0.5  # Neutral confidence if only one model

        preds = list(predictions.values())

        # Calculate disagreement
        std = np.std(preds)
        mean_abs = np.mean(np.abs(preds))

        if mean_abs == 0:
            return 0.0  # No signal

        # Disagreement ratio
        disagreement = std / (mean_abs + 1e-8)

        # Confidence = 1 - disagreement (capped at 0)
        confidence = max(0, min(1, 1 - disagreement))

        return confidence

    def get_model_contributions(
        self,
        X: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Get detailed breakdown of each model's contribution.

        Args:
            X: Input features

        Returns:
            Dictionary of model_name -> {
                'prediction': float,
                'weight': float,
                'contribution': float (prediction × weight)
            }
        """
        contributions = {}

        for name, model in self.models.items():
            weight = self.weights.get(name, 0)

            if weight == 0:
                contributions[name] = {
                    'prediction': 0.0,
                    'weight': 0.0,
                    'contribution': 0.0,
                    'excluded': True
                }
                continue

            try:
                pred = model.predict(X) if hasattr(model, 'predict') else model.predict_single(X)
                pred = float(pred) if not isinstance(pred, float) else pred

                contributions[name] = {
                    'prediction': pred,
                    'weight': weight,
                    'contribution': pred * weight,
                    'excluded': False
                }
            except Exception as e:
                logger.warning(f"Error getting prediction from {name}: {e}")
                contributions[name] = {
                    'prediction': 0.0,
                    'weight': weight,
                    'contribution': 0.0,
                    'error': str(e)
                }

        return contributions

    def _log_ensemble_configuration(self):
        """Log ensemble configuration for debugging."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("PERFORMANCE-WEIGHTED ENSEMBLE INITIALIZED")
        logger.info("=" * 70)
        logger.info(f"Total models: {len(self.models)}")
        logger.info(f"Active models: {sum(1 for w in self.weights.values() if w > 0)}")
        logger.info(f"Excluded models: {sum(1 for w in self.weights.values() if w == 0)}")
        logger.info(f"Min directional accuracy: {self.min_dir_acc:.2%}")
        logger.info(f"Min correlation: {self.min_correlation:.4f}")
        logger.info("")

    def update_weights(
        self,
        new_metrics: Dict[str, Dict[str, float]]
    ):
        """
        Update ensemble weights based on new performance metrics.

        Useful for adaptive weighting during live trading.

        Args:
            new_metrics: Updated performance metrics
        """
        logger.info("Updating ensemble weights with new metrics...")
        self.metrics = new_metrics
        self.weights = self.calculate_dynamic_weights(new_metrics)
        logger.info("Weights updated successfully!")


# Export for use in other modules
__all__ = ['PerformanceWeightedEnsemble']
