"""
MSE + Post-Scaling: Simple Alternative to TradingLoss
=====================================================

WHY THIS APPROACH:
- TradingLoss can be unstable (negative correlation!)
- This keeps training simple and stable
- Fixes conservatism in inference, not training

ADVANTAGES:
- Stable training (pure MSE)
- Good directional accuracy preserved
- Positive correlation maintained
- Conservatism fixed in post-processing
- No risk of variance loss dominating

HOW IT WORKS:
1. Train with standard MSE loss (stable, proven)
2. Model learns good patterns (positive correlation)
3. At inference, scale predictions to match target variance
4. Best of both worlds!
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from loguru import logger


class MSEWithPostScaling:
    """
    Simple MSE training + inference-time scaling.

    This is the SAFEST approach when TradingLoss causes issues.
    """

    def __init__(
        self,
        target_std: float = 0.003,  # Target std (update from training data)
        max_scale_factor: float = 2.0  # Max scaling to prevent explosion
    ):
        """
        Initialize MSE + scaling.

        Args:
            target_std: Expected std of returns (from training data)
            max_scale_factor: Maximum scaling allowed (safety limit)
        """
        self.criterion = nn.MSELoss()
        self.target_std = target_std
        self.max_scale_factor = max_scale_factor
        self.pred_std_history = []

    def calculate_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate standard MSE loss.

        Simple, stable, proven.
        """
        return self.criterion(predictions, targets)

    def scale_predictions(
        self,
        predictions: torch.Tensor,
        adaptive: bool = True
    ) -> torch.Tensor:
        """
        Scale predictions to match target variance.

        This is done at INFERENCE time, not training!

        Args:
            predictions: Raw model predictions
            adaptive: Use adaptive scaling based on recent history

        Returns:
            Scaled predictions with realistic variance
        """
        with torch.no_grad():
            # Calculate current prediction std
            pred_std = predictions.std().item()

            # Store in history (for adaptive scaling)
            self.pred_std_history.append(pred_std)
            if len(self.pred_std_history) > 100:
                self.pred_std_history.pop(0)

            # Calculate scale factor
            if adaptive and len(self.pred_std_history) > 10:
                # Use recent average std
                avg_pred_std = np.mean(self.pred_std_history[-20:])
                scale_factor = self.target_std / avg_pred_std
            else:
                # Use current std
                scale_factor = self.target_std / (pred_std + 1e-8)

            # Cap scaling (safety limit)
            scale_factor = min(scale_factor, self.max_scale_factor)
            scale_factor = max(scale_factor, 0.5)  # Don't scale down too much

            # Apply scaling
            mean_pred = predictions.mean()
            scaled_predictions = (predictions - mean_pred) * scale_factor + mean_pred

            logger.debug(f"Post-scaling: pred_std={pred_std:.6f}, "
                        f"target_std={self.target_std:.6f}, "
                        f"scale_factor={scale_factor:.3f}")

            return scaled_predictions

    def update_target_std(self, target_std: float):
        """Update target std from training data."""
        self.target_std = target_std
        logger.info(f"Updated target std to {target_std:.6f}")


class ProgressiveTradingLoss(nn.Module):
    """
    Progressive TradingLoss: Gradually increase variance weight.

    STRATEGY:
    - Start with pure MSE (epochs 1-20)
    - Gradually add variance component (epochs 20-50)
    - Full TradingLoss (epochs 50+)

    This prevents early variance-matching from ruining the model.
    """

    def __init__(
        self,
        base_mse_weight: float = 0.70,
        base_direction_weight: float = 0.25,
        base_variance_weight: float = 0.05,
        warmup_epochs: int = 20,
        full_variance_epoch: int = 50
    ):
        super().__init__()
        self.base_mse_weight = base_mse_weight
        self.base_direction_weight = base_direction_weight
        self.base_variance_weight = base_variance_weight
        self.warmup_epochs = warmup_epochs
        self.full_variance_epoch = full_variance_epoch

        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        """Update current epoch for progressive scaling."""
        self.current_epoch = epoch

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> tuple:
        """
        Calculate progressive loss.

        Variance weight increases gradually over epochs.
        """
        # MSE component
        mse_loss = torch.nn.functional.mse_loss(predictions, targets)

        # Direction component
        same_sign = (predictions * targets) > 0
        wrong_sign = ~same_sign
        direction_errors = torch.where(
            wrong_sign,
            (predictions - targets) ** 2 * 3.0,
            (predictions - targets) ** 2 * 0.5
        )
        direction_loss = direction_errors.mean()

        # Variance component (progressive)
        pred_std = predictions.std() + 1e-8
        target_std = targets.std() + 1e-8
        std_ratio = pred_std / target_std
        variance_loss = (torch.log(std_ratio)) ** 2
        variance_loss = torch.clamp(variance_loss, max=4.0)

        # Calculate progressive variance weight
        if self.current_epoch < self.warmup_epochs:
            # No variance penalty during warmup
            variance_weight = 0.0
        elif self.current_epoch < self.full_variance_epoch:
            # Gradually increase variance weight
            progress = (self.current_epoch - self.warmup_epochs) / \
                      (self.full_variance_epoch - self.warmup_epochs)
            variance_weight = self.base_variance_weight * progress
        else:
            # Full variance weight
            variance_weight = self.base_variance_weight

        # Combined loss
        total_loss = (
            self.base_mse_weight * mse_loss +
            self.base_direction_weight * direction_loss +
            variance_weight * variance_loss
        )

        # Components for logging
        components = {
            'mse': mse_loss.item(),
            'direction': direction_loss.item(),
            'variance': variance_loss.item(),
            'variance_weight': variance_weight,
            'total': total_loss.item(),
            'pred_std': pred_std.item(),
            'target_std': target_std.item(),
            'std_ratio': (pred_std / target_std).item()
        }

        return total_loss, components


# Example usage
if __name__ == "__main__":
    # Approach 1: MSE + Post-Scaling (SAFEST)
    scaler = MSEWithPostScaling(target_std=0.003)

    # During training
    predictions = torch.randn(64)
    targets = torch.randn(64) * 0.003  # Scale to realistic returns

    loss = scaler.calculate_loss(predictions, targets)
    loss.backward()

    # During inference
    scaled_preds = scaler.scale_predictions(predictions)
    print(f"Original std: {predictions.std():.6f}")
    print(f"Scaled std: {scaled_preds.std():.6f}")
    print(f"Target std: {scaler.target_std:.6f}")

    # Approach 2: Progressive TradingLoss
    progressive = ProgressiveTradingLoss()

    for epoch in range(60):
        progressive.set_epoch(epoch)
        loss, components = progressive(predictions, targets)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Variance weight = {components['variance_weight']:.3f}")
