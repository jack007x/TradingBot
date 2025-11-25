"""
Custom Loss Functions for Trading Regression Models
====================================================

These loss functions address the critical problem of model conservatism in financial prediction.

PROBLEM: Standard MSE Loss Makes Models Too Conservative
---------------------------------------------------------
When using MSE loss, models learn to predict near the mean to minimize squared errors.
This creates predictions with MUCH smaller variance than actual returns:

- Actual returns std:     0.00343 (0.343%)
- Predicted returns std:  0.00016 (0.016%)  ← Only 5% of actual!

Why this happens:
1. MSE heavily penalizes large errors (squared term)
2. Safest strategy: predict close to mean (minimal variance)
3. Model sacrifices capturing large moves to reduce MSE
4. Good for statistical metrics, BAD for trading!

SOLUTIONS: Custom loss functions that encourage realistic variance and directional accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class TradingLoss(nn.Module):
    """
    Trading-Focused Loss: MSE + Directional Penalty

    WHY THIS FIXES CONSERVATISM:
    - Penalizes wrong direction MORE than magnitude errors
    - Encourages model to make bold predictions when confident
    - Better aligns with trading objectives (direction > magnitude)

    EXPECTED IMPROVEMENT:
    - Directional accuracy: 52% → 55-58%
    - Predicted std: 5% of actual → 40-60% of actual
    - Correlation: 0.11 → 0.15-0.20

    Args:
        mse_weight: Weight for MSE component (magnitude accuracy)
        direction_weight: Weight for directional component (sign accuracy)
        variance_weight: Weight for variance matching (anti-conservatism)
    """

    def __init__(
        self,
        mse_weight: float = 0.4,
        direction_weight: float = 0.4,
        variance_weight: float = 0.2
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.direction_weight = direction_weight
        self.variance_weight = variance_weight

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Calculate combined loss.

        Returns:
            total_loss: Combined weighted loss
            components: Dictionary of individual loss components (for logging)
        """
        batch_size = predictions.size(0)

        # Component 1: MSE Loss (magnitude accuracy)
        # Standard MSE for overall prediction accuracy
        mse_loss = F.mse_loss(predictions, targets)

        # Component 2: Directional Loss (sign accuracy)
        # Penalize wrong direction predictions MORE heavily
        # This is what traders care about most!

        # Check if prediction and target have same sign
        same_sign = (predictions * targets) > 0  # Both positive or both negative
        wrong_sign = ~same_sign

        # For wrong signs, use squared error heavily penalized
        # For correct signs, use smaller penalty
        direction_errors = torch.where(
            wrong_sign,
            (predictions - targets) ** 2 * 3.0,  # 3x penalty for wrong direction!
            (predictions - targets) ** 2 * 0.5   # 0.5x penalty for correct direction
        )
        direction_loss = direction_errors.mean()

        # Component 3: Variance Matching Loss (anti-conservatism)
        # Force model to match actual return variance
        # This prevents model from being too conservative!

        pred_std = predictions.std() + 1e-8  # Add small epsilon for stability
        target_std = targets.std() + 1e-8

        # Penalize if predicted variance is too small OR too large
        variance_loss = ((pred_std - target_std) ** 2) / (target_std ** 2)

        # Combined loss
        total_loss = (
            self.mse_weight * mse_loss +
            self.direction_weight * direction_loss +
            self.variance_weight * variance_loss
        )

        # Return components for logging
        components = {
            'mse': mse_loss.item(),
            'direction': direction_loss.item(),
            'variance': variance_loss.item(),
            'total': total_loss.item(),
            'pred_std': pred_std.item(),
            'target_std': target_std.item(),
            'std_ratio': (pred_std / target_std).item()
        }

        return total_loss, components


class QuantileLoss(nn.Module):
    """
    Quantile Regression Loss: Predict Distribution, Not Just Point Estimate

    WHY THIS FIXES CONSERVATISM:
    - Predicts multiple quantiles (10th, 50th, 90th percentile)
    - Captures full distribution of possible returns
    - Provides natural confidence intervals
    - No penalty for predicting high variance!

    EXPECTED IMPROVEMENT:
    - Uncertainty quantification (confidence intervals)
    - Better extreme event prediction
    - More realistic variance in predictions
    - Useful for position sizing (use 90th percentile for risk)

    Example Output:
    - 10th percentile: -0.015 (10% chance of worse)
    - 50th percentile: +0.005 (median prediction)
    - 90th percentile: +0.025 (10% chance of better)

    Args:
        quantiles: List of quantiles to predict (e.g., [0.1, 0.5, 0.9])
    """

    def __init__(self, quantiles: list = [0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)

    def forward(
        self,
        predictions: torch.Tensor,  # Shape: (batch, num_quantiles)
        targets: torch.Tensor       # Shape: (batch, 1)
    ) -> Tuple[torch.Tensor, dict]:
        """
        Calculate quantile loss.

        Quantile loss formula:
        - If actual > predicted: loss = quantile * error
        - If actual < predicted: loss = (1 - quantile) * error

        This asymmetric penalty encourages proper quantile estimation.
        """
        # Ensure targets have correct shape for broadcasting
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)

        # Calculate errors for each quantile
        # predictions: (batch, num_quantiles)
        # targets: (batch, 1)
        errors = targets - predictions  # Broadcasting: (batch, num_quantiles)

        # Quantile loss (asymmetric)
        quantile_losses = []
        for i, q in enumerate(self.quantiles):
            # For quantile q:
            # - If error > 0 (underestimation): penalize by q
            # - If error < 0 (overestimation): penalize by (1-q)
            q_loss = torch.max(
                q * errors[:, i],
                (q - 1) * errors[:, i]
            )
            quantile_losses.append(q_loss.mean())

        # Average across quantiles
        total_loss = sum(quantile_losses) / len(quantile_losses)

        # Calculate metrics for logging
        components = {
            'total': total_loss.item(),
        }

        # Add individual quantile losses
        for i, q in enumerate(self.quantiles):
            components[f'q{int(q*100)}'] = quantile_losses[i].item()

        # Calculate prediction spread (uncertainty measure)
        if self.num_quantiles >= 2:
            pred_spread = (predictions[:, -1] - predictions[:, 0]).mean()
            components['prediction_spread'] = pred_spread.item()

        return total_loss, components


class VarianceMatchingLoss(nn.Module):
    """
    Variance Matching Loss: Force Predictions to Match Target Variance

    WHY THIS FIXES CONSERVATISM:
    - Directly penalizes variance mismatch
    - Can be combined with any other loss (MSE, etc.)
    - Simple and effective anti-conservatism mechanism

    EXPECTED IMPROVEMENT:
    - Predicted std: 5% of actual → 70-90% of actual
    - Better capture of market volatility
    - More realistic predictions

    This is the SIMPLEST fix for conservatism!

    Args:
        mse_weight: Weight for MSE component
        variance_weight: Weight for variance matching
        mean_weight: Weight for mean matching (optional)
    """

    def __init__(
        self,
        mse_weight: float = 0.7,
        variance_weight: float = 0.2,
        mean_weight: float = 0.1
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.variance_weight = variance_weight
        self.mean_weight = mean_weight

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Calculate variance matching loss.
        """
        # Component 1: Standard MSE
        mse_loss = F.mse_loss(predictions, targets)

        # Component 2: Variance matching
        pred_var = predictions.var() + 1e-8
        target_var = targets.var() + 1e-8

        # Relative variance error (scale-invariant)
        variance_loss = ((pred_var - target_var) ** 2) / target_var

        # Component 3: Mean matching (ensure unbiased predictions)
        pred_mean = predictions.mean()
        target_mean = targets.mean()
        mean_loss = (pred_mean - target_mean) ** 2

        # Combined loss
        total_loss = (
            self.mse_weight * mse_loss +
            self.variance_weight * variance_loss +
            self.mean_weight * mean_loss
        )

        # Logging components
        pred_std = torch.sqrt(pred_var)
        target_std = torch.sqrt(target_var)

        components = {
            'mse': mse_loss.item(),
            'variance': variance_loss.item(),
            'mean': mean_loss.item(),
            'total': total_loss.item(),
            'pred_std': pred_std.item(),
            'target_std': target_std.item(),
            'std_ratio': (pred_std / target_std).item(),
            'pred_mean': pred_mean.item(),
            'target_mean': target_mean.item()
        }

        return total_loss, components


class SharpeRatioLoss(nn.Module):
    """
    Sharpe Ratio Loss: Optimize Directly for Trading Performance

    WHY THIS IS INTERESTING:
    - Maximizes risk-adjusted returns (what traders actually want!)
    - Considers both return AND volatility
    - Aligns model training with trading objective

    WARNING: Experimental! May be unstable.
    - Sharpe ratio is non-differentiable at zero
    - Requires careful tuning
    - Use as supplementary loss, not primary

    EXPECTED IMPROVEMENT:
    - Better risk-adjusted predictions
    - Natural position sizing signals
    - Trades quality over quantity

    Args:
        mse_weight: Weight for MSE (stability)
        sharpe_weight: Weight for Sharpe optimization
    """

    def __init__(
        self,
        mse_weight: float = 0.7,
        sharpe_weight: float = 0.3,
        risk_free_rate: float = 0.02
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.sharpe_weight = sharpe_weight
        self.risk_free_rate = risk_free_rate / 252  # Daily risk-free rate

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Calculate Sharpe-optimized loss.
        """
        # Component 1: Standard MSE
        mse_loss = F.mse_loss(predictions, targets)

        # Component 2: Negative Sharpe ratio (minimize = maximize Sharpe)
        # Treat predictions as "returns" and calculate their Sharpe

        pred_mean = predictions.mean()
        pred_std = predictions.std() + 1e-8  # Avoid division by zero

        # Sharpe ratio of predictions
        excess_return = pred_mean - self.risk_free_rate
        sharpe_ratio = excess_return / pred_std

        # We want to MAXIMIZE Sharpe, so MINIMIZE negative Sharpe
        # But also penalize if predictions don't match targets
        # This encourages high Sharpe predictions that are also accurate

        # Correlation between predictions and targets (quality measure)
        pred_centered = predictions - pred_mean
        target_centered = targets - targets.mean()
        correlation = (pred_centered * target_centered).mean() / (
            (pred_centered.std() + 1e-8) * (target_centered.std() + 1e-8)
        )

        # Sharpe loss: maximize Sharpe AND correlation
        sharpe_loss = -sharpe_ratio * correlation

        # Combined loss
        total_loss = (
            self.mse_weight * mse_loss +
            self.sharpe_weight * sharpe_loss
        )

        components = {
            'mse': mse_loss.item(),
            'sharpe': sharpe_ratio.item(),
            'correlation': correlation.item(),
            'sharpe_loss': sharpe_loss.item(),
            'total': total_loss.item()
        }

        return total_loss, components


class HuberLoss(nn.Module):
    """
    Huber Loss: Robust to Outliers

    WHY THIS HELPS:
    - MSE too sensitive to outliers (squared term)
    - MAE not sensitive enough to large errors
    - Huber combines both: quadratic near zero, linear for large errors

    EXPECTED IMPROVEMENT:
    - More stable training (less affected by extreme returns)
    - Better generalization
    - Can use higher learning rate

    This is a SIMPLE, PROVEN improvement over MSE!

    Args:
        delta: Threshold for switching from quadratic to linear
               (typically set to 1.0 for normalized data)
    """

    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Calculate Huber loss.

        Formula:
        - If |error| <= delta: loss = 0.5 * error^2
        - If |error| > delta:  loss = delta * (|error| - 0.5 * delta)
        """
        errors = predictions - targets
        abs_errors = torch.abs(errors)

        # Quadratic for small errors, linear for large
        quadratic = 0.5 * errors ** 2
        linear = self.delta * (abs_errors - 0.5 * self.delta)

        # Use quadratic if error <= delta, else use linear
        loss = torch.where(abs_errors <= self.delta, quadratic, linear)

        total_loss = loss.mean()

        components = {
            'total': total_loss.item(),
            'mean_abs_error': abs_errors.mean().item(),
            'quadratic_ratio': (abs_errors <= self.delta).float().mean().item()
        }

        return total_loss, components


# ============================================================================
# Loss Function Comparison Helper
# ============================================================================

class LossComparator:
    """
    Utility to compare different loss functions during training.

    Usage:
    ```python
    comparator = LossComparator()

    # During training
    for epoch in range(epochs):
        for X, y in train_loader:
            predictions = model(X)

            # Compare all losses
            comparison = comparator.compare_losses(predictions, y)

            # Use your preferred loss for training
            loss = comparison['trading']['loss']
            loss.backward()

            # Log all metrics
            logger.info(f"Epoch {epoch}: {comparison}")
    ```
    """

    def __init__(self):
        self.losses = {
            'mse': nn.MSELoss(),
            'trading': TradingLoss(),
            'variance': VarianceMatchingLoss(),
            'huber': HuberLoss(),
            'quantile': QuantileLoss()
        }

    def compare_losses(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> dict:
        """
        Calculate all losses and return comparison.

        Returns:
            Dictionary with loss values and metrics for each loss function
        """
        results = {}

        for name, loss_fn in self.losses.items():
            if name == 'mse':
                # Standard MSE doesn't return components
                loss_value = loss_fn(predictions, targets)
                results[name] = {
                    'loss': loss_value,
                    'components': {'total': loss_value.item()}
                }
            else:
                # Custom losses return (loss, components)
                if name == 'quantile':
                    # Quantile needs special handling (multiple outputs)
                    # Skip for now in comparison, or handle separately
                    continue

                loss_value, components = loss_fn(predictions, targets)
                results[name] = {
                    'loss': loss_value,
                    'components': components
                }

        return results


# ============================================================================
# Export all loss functions
# ============================================================================

__all__ = [
    'TradingLoss',
    'QuantileLoss',
    'VarianceMatchingLoss',
    'SharpeRatioLoss',
    'HuberLoss',
    'LossComparator'
]
