"""
Model Evaluator Module
=======================
Evaluates model performance with trading-specific metrics.

Usage:
    from models import ModelEvaluator
    from config import get_config

    config = get_config()
    evaluator = ModelEvaluator(config)

    # Evaluate model
    metrics = evaluator.evaluate(model, X_test, y_test, returns)

    # Generate report
    report = evaluator.generate_report(metrics)
"""

import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

from config.config_loader import Config
from models.model_baseline import BaselineModel

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluates ML models with both classification and trading metrics.

    Trading metrics include:
    - Sharpe ratio (simulated)
    - Win rate
    - Profit factor
    - Maximum drawdown
    - Risk-adjusted return
    """

    def __init__(self, config: Config):
        """
        Initialize evaluator.

        Args:
            config: Configuration object
        """
        self.config = config
        self.min_prob_threshold = config.model.min_probability_threshold

    def evaluate(
            self,
            model: BaselineModel,
            X: np.ndarray,
            y: np.ndarray,
            future_returns: Optional[np.ndarray] = None,
            timestamps: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.

        Args:
            model: Trained model
            X: Feature matrix
            y: True labels
            future_returns: Actual future returns for each sample
            timestamps: Timestamps for each sample

        Returns:
            Dict with classification and trading metrics
        """
        logger.info(f"Evaluating model on {len(X)} samples")

        metrics = {}

        # Get predictions
        y_pred = model.predict(X)
        proba = model.predict_proba(X)

        # Classification metrics
        metrics.update(self._classification_metrics(y, y_pred, proba))

        # Trading simulation metrics (if returns provided)
        if future_returns is not None:
            trading_metrics = self._trading_metrics(
                y_pred, proba, future_returns, timestamps
            )
            metrics.update(trading_metrics)

        # Confidence analysis
        metrics.update(self._confidence_analysis(y, y_pred, proba))

        return metrics

    def _classification_metrics(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            proba: np.ndarray
    ) -> Dict[str, float]:
        """Calculate classification metrics."""
        metrics = {}

        # Overall accuracy
        metrics['accuracy'] = (y_pred == y_true).mean()

        # Per-class metrics
        for label, name in [(-1, 'down'), (0, 'neutral'), (1, 'up')]:
            mask_pred = y_pred == label
            mask_true = y_true == label

            # Support
            support = mask_true.sum()
            metrics[f'support_{name}'] = int(support)

            # Precision
            if mask_pred.sum() > 0:
                precision = (y_true[mask_pred] == label).mean()
            else:
                precision = 0.0
            metrics[f'precision_{name}'] = precision

            # Recall
            if support > 0:
                recall = (y_pred[mask_true] == label).mean()
            else:
                recall = 0.0
            metrics[f'recall_{name}'] = recall

            # F1
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            metrics[f'f1_{name}'] = f1

        # Macro F1
        f1_scores = [metrics.get(f'f1_{name}', 0) for name in ['down', 'neutral', 'up']]
        metrics['f1_macro'] = np.mean(f1_scores)

        # Directional accuracy (ignoring neutral)
        directional_mask = (y_pred != 0) | (y_true != 0)
        if directional_mask.sum() > 0:
            metrics['directional_accuracy'] = (y_pred[directional_mask] == y_true[directional_mask]).mean()
        else:
            metrics['directional_accuracy'] = 0.0

        # Prediction distribution
        for label, name in [(-1, 'down'), (0, 'neutral'), (1, 'up')]:
            metrics[f'pred_ratio_{name}'] = (y_pred == label).mean()

        return metrics

    def _trading_metrics(
            self,
            y_pred: np.ndarray,
            proba: np.ndarray,
            future_returns: np.ndarray,
            timestamps: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Calculate trading-related metrics.

        Simulates trading based on predictions and calculates PnL metrics.
        """
        metrics = {}

        # Filter by probability threshold
        max_proba = np.max(proba, axis=1)
        high_conf_mask = max_proba >= self.min_prob_threshold

        # Only trade on directional predictions with high confidence
        trade_mask = (y_pred != 0) & high_conf_mask

        if trade_mask.sum() == 0:
            logger.warning("No trades generated (all filtered out)")
            metrics['num_trades'] = 0
            metrics['win_rate'] = 0.0
            metrics['profit_factor'] = 0.0
            metrics['sharpe_ratio'] = 0.0
            metrics['max_drawdown'] = 0.0
            return metrics

        # Calculate trade returns
        # Long (y_pred=1): profit if future_return > 0
        # Short (y_pred=-1): profit if future_return < 0
        trade_returns = np.where(
            trade_mask,
            y_pred * future_returns,  # Direction * actual return
            0.0
        )

        # Only consider actual trades
        actual_trade_returns = trade_returns[trade_mask]

        # Basic trading metrics
        metrics['num_trades'] = int(trade_mask.sum())
        metrics['trade_frequency'] = trade_mask.mean()

        # Win rate
        wins = (actual_trade_returns > 0).sum()
        metrics['win_rate'] = wins / len(actual_trade_returns) if len(actual_trade_returns) > 0 else 0.0

        # Average trade returns
        metrics['avg_trade_return'] = actual_trade_returns.mean()
        metrics['avg_win'] = actual_trade_returns[actual_trade_returns > 0].mean() if wins > 0 else 0.0
        metrics['avg_loss'] = actual_trade_returns[actual_trade_returns < 0].mean() if (len(actual_trade_returns) - wins) > 0 else 0.0

        # Profit factor
        gross_profit = actual_trade_returns[actual_trade_returns > 0].sum()
        gross_loss = abs(actual_trade_returns[actual_trade_returns < 0].sum())
        metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Total return
        metrics['total_return'] = actual_trade_returns.sum()
        metrics['cumulative_return'] = (1 + actual_trade_returns).prod() - 1

        # Sharpe ratio (annualized, assuming M15 = 4 trades per hour)
        if len(actual_trade_returns) > 1 and actual_trade_returns.std() > 0:
            # Annualization factor: sqrt(trades per year)
            # M15: 4 per hour * 24 hours * 252 trading days = 24,192
            annualization = np.sqrt(24192)
            sharpe = (actual_trade_returns.mean() / actual_trade_returns.std()) * annualization
            metrics['sharpe_ratio'] = sharpe
        else:
            metrics['sharpe_ratio'] = 0.0

        # Maximum drawdown
        cumulative = (1 + actual_trade_returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        metrics['max_drawdown'] = abs(drawdown.min())

        # Calmar ratio (return / max drawdown)
        if metrics['max_drawdown'] > 0:
            metrics['calmar_ratio'] = metrics['cumulative_return'] / metrics['max_drawdown']
        else:
            metrics['calmar_ratio'] = float('inf') if metrics['cumulative_return'] > 0 else 0.0

        # Expectancy
        metrics['expectancy'] = (
                metrics['win_rate'] * metrics['avg_win'] +
                (1 - metrics['win_rate']) * metrics['avg_loss']
        )

        return metrics

    def _confidence_analysis(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            proba: np.ndarray
    ) -> Dict[str, float]:
        """Analyze prediction confidence and calibration."""
        metrics = {}

        max_proba = np.max(proba, axis=1)

        # Average confidence
        metrics['avg_confidence'] = max_proba.mean()
        metrics['avg_confidence_correct'] = max_proba[y_pred == y_true].mean() if (y_pred == y_true).sum() > 0 else 0.0
        metrics['avg_confidence_wrong'] = max_proba[y_pred != y_true].mean() if (y_pred != y_true).sum() > 0 else 0.0

        # Accuracy at different confidence thresholds
        for threshold in [0.5, 0.55, 0.6, 0.65, 0.7]:
            mask = max_proba >= threshold
            if mask.sum() > 0:
                acc = (y_pred[mask] == y_true[mask]).mean()
                coverage = mask.mean()
                metrics[f'accuracy_at_{int(threshold*100)}'] = acc
                metrics[f'coverage_at_{int(threshold*100)}'] = coverage

        return metrics

    def compare_models(
            self,
            metrics_current: Dict[str, float],
            metrics_new: Dict[str, float]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Compare new model metrics against current model.

        Returns True if new model should replace current.

        Args:
            metrics_current: Current model metrics
            metrics_new: New model metrics

        Returns:
            Tuple[should_deploy, comparison_details]
        """
        comparison = {}

        # Key metrics to compare
        key_metrics = [
            ('sharpe_ratio', 'higher', self.config.retraining.sharpe_tolerance),
            ('max_drawdown', 'lower', self.config.retraining.drawdown_tolerance),
            ('win_rate', 'higher', None),
            ('profit_factor', 'higher', None),
            ('accuracy', 'higher', None),
        ]

        passes = []

        for metric, direction, tolerance in key_metrics:
            current_val = metrics_current.get(metric, 0)
            new_val = metrics_new.get(metric, 0)

            comparison[metric] = {
                'current': current_val,
                'new': new_val,
                'change': new_val - current_val,
                'change_pct': ((new_val - current_val) / current_val * 100) if current_val != 0 else 0
            }

            # Check if passes threshold
            if tolerance is not None:
                if direction == 'higher':
                    passes.append(new_val >= current_val * tolerance)
                else:
                    passes.append(new_val <= current_val * tolerance)

        # Additional absolute thresholds
        min_sharpe = metrics_new.get('sharpe_ratio', 0) >= self.config.retraining.min_sharpe_ratio
        max_dd = metrics_new.get('max_drawdown', 1) <= self.config.retraining.max_allowed_drawdown
        min_wr = metrics_new.get('win_rate', 0) >= self.config.retraining.min_win_rate
        min_pf = metrics_new.get('profit_factor', 0) >= self.config.retraining.min_profit_factor

        absolute_checks = [min_sharpe, max_dd, min_wr, min_pf]

        # Decision: all relative and absolute checks must pass
        should_deploy = all(passes) and all(absolute_checks)

        comparison['relative_checks_passed'] = sum(passes)
        comparison['absolute_checks_passed'] = sum(absolute_checks)
        comparison['should_deploy'] = should_deploy
        comparison['reason'] = "All criteria met" if should_deploy else "One or more criteria failed"

        return should_deploy, comparison

    def generate_report(self, metrics: Dict[str, Any]) -> str:
        """Generate human-readable evaluation report."""
        lines = [
            "=" * 60,
            "MODEL EVALUATION REPORT",
            "=" * 60,
            "",
            "CLASSIFICATION METRICS",
            "-" * 30,
            f"  Accuracy:              {metrics.get('accuracy', 0):.4f}",
            f"  Directional Accuracy:  {metrics.get('directional_accuracy', 0):.4f}",
            f"  Macro F1:              {metrics.get('f1_macro', 0):.4f}",
            "",
            "  Per-Class Performance:",
            f"    Up    - P: {metrics.get('precision_up', 0):.3f}, R: {metrics.get('recall_up', 0):.3f}, F1: {metrics.get('f1_up', 0):.3f}",
            f"    Neut  - P: {metrics.get('precision_neutral', 0):.3f}, R: {metrics.get('recall_neutral', 0):.3f}, F1: {metrics.get('f1_neutral', 0):.3f}",
            f"    Down  - P: {metrics.get('precision_down', 0):.3f}, R: {metrics.get('recall_down', 0):.3f}, F1: {metrics.get('f1_down', 0):.3f}",
            "",
        ]

        if 'num_trades' in metrics:
            lines.extend([
                "TRADING METRICS",
                "-" * 30,
                f"  Number of Trades:      {metrics.get('num_trades', 0)}",
                f"  Trade Frequency:       {metrics.get('trade_frequency', 0):.2%}",
                f"  Win Rate:              {metrics.get('win_rate', 0):.2%}",
                f"  Profit Factor:         {metrics.get('profit_factor', 0):.3f}",
                f"  Sharpe Ratio:          {metrics.get('sharpe_ratio', 0):.3f}",
                f"  Max Drawdown:          {metrics.get('max_drawdown', 0):.2%}",
                f"  Total Return:          {metrics.get('total_return', 0):.4f}",
                f"  Cumulative Return:     {metrics.get('cumulative_return', 0):.2%}",
                f"  Expectancy:            {metrics.get('expectancy', 0):.5f}",
                "",
            ])

        lines.extend([
            "CONFIDENCE ANALYSIS",
            "-" * 30,
            f"  Avg Confidence:        {metrics.get('avg_confidence', 0):.3f}",
            f"  Avg Conf (Correct):    {metrics.get('avg_confidence_correct', 0):.3f}",
            f"  Avg Conf (Wrong):      {metrics.get('avg_confidence_wrong', 0):.3f}",
            "",
            "=" * 60,
        ])

        return "\n".join(lines)


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from config import get_config
    from data.data_fetcher import DataFetcher
    from features.feature_engineering import FeatureEngineer
    from models.model_trainer import ModelTrainer

    config = get_config()
    fetcher = DataFetcher(config)
    engineer = FeatureEngineer(config)
    trainer = ModelTrainer(config)
    evaluator = ModelEvaluator(config)

    # Prepare data
    from datetime import datetime
    start = datetime(2022, 1, 1)
    end = datetime(2024, 6, 1)
    df = fetcher.fetch_historical(start, end)
    df_features = engineer.build_features(df)

    # Get future returns for evaluation
    future_returns = df_features['future_return'].values

    X, y, feature_names = engineer.prepare_training_data(df_features)

    # Also get the corresponding future returns (aligned with X, y)
    df_clean = df_features.dropna(subset=feature_names + ['label', 'future_return'])
    future_returns_clean = df_clean['future_return'].values

    # Split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    returns_test = future_returns_clean[split_idx:]

    # Train
    model, _ = trainer.train_simple(X_train, y_train, feature_names)

    # Evaluate
    metrics = evaluator.evaluate(model, X_test, y_test, returns_test)

    # Print report
    print(evaluator.generate_report(metrics))
