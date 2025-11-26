"""
Time Series Data Augmentation

PROBLEM:
- Limited training data (few years of historical data)
- Model overfits to specific market conditions
- Need more diverse examples without breaking time series structure

SOLUTION - SAFE TIME SERIES AUGMENTATION:
1. Magnitude Warping: Scale values slightly (0.98-1.02x)
2. Jittering: Add small random noise (std=0.001)
3. Combined: Apply both transformations

IMPORTANT: Does NOT break temporal structure!
- No random shuffling of timestamps
- No window slicing that breaks causality
- Preserves sequence order and relationships

EXPECTED BENEFITS:
- 2-3x effective training data
- Better generalization (less overfitting)
- More robust to different market conditions
- Improved correlation and directional accuracy
"""

import numpy as np
from typing import Tuple, Dict
from loguru import logger


class TimeSeriesAugmenter:
    """
    Augment time series data while preserving temporal structure.

    Safe transformations that don't break causality:
    - Magnitude warping: Slight scaling of values
    - Jittering: Small random noise addition
    - Combined: Both transformations together

    Does NOT use:
    - Time warping (changes sequence length)
    - Window slicing (breaks temporal dependencies)
    - Random permutation (destroys order)
    """

    def __init__(
        self,
        magnitude_range: Tuple[float, float] = (0.98, 1.02),
        jitter_std: float = 0.001,
        augment_ratio: float = 1.5
    ):
        """
        Initialize time series augmenter.

        Args:
            magnitude_range: Range for magnitude warping (min, max) multipliers
            jitter_std: Standard deviation for Gaussian noise
            augment_ratio: How much to augment (1.5 = 50% more data)
        """
        self.magnitude_range = magnitude_range
        self.jitter_std = jitter_std
        self.augment_ratio = augment_ratio

        logger.info(f"TimeSeriesAugmenter initialized:")
        logger.info(f"  - Magnitude range: {magnitude_range}")
        logger.info(f"  - Jitter std: {jitter_std}")
        logger.info(f"  - Augmentation ratio: {augment_ratio}x")

    def augment_batch(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment a batch of sequences.

        Args:
            X: Training sequences (n_samples, seq_len, n_features)
            y: Training labels (n_samples,)

        Returns:
            Tuple of (augmented_X, augmented_y) with more samples
        """
        n_original = len(X)
        n_augment = int(n_original * self.augment_ratio)

        logger.info(f"Augmenting {n_original} samples → {n_original + n_augment} total")

        # Start with original data
        augmented_X = [X]
        augmented_y = [y]

        # Generate augmented samples
        for i in range(n_augment):
            # Randomly select samples to augment
            indices = np.random.choice(n_original, size=max(1, n_original // 2), replace=False)

            # Random augmentation type
            aug_type = np.random.choice(['magnitude', 'jitter', 'combined'])

            if aug_type == 'magnitude':
                aug_X = self.magnitude_warping(X[indices])
            elif aug_type == 'jitter':
                aug_X = self.add_jitter(X[indices])
            else:  # combined
                aug_X = self.add_jitter(
                    self.magnitude_warping(X[indices])
                )

            augmented_X.append(aug_X)
            augmented_y.append(y[indices])

        # Combine all data
        final_X = np.vstack(augmented_X)
        final_y = np.concatenate(augmented_y)

        logger.info(f"Augmentation complete: {final_X.shape[0]} total samples")
        logger.info(f"  - Original: {n_original}")
        logger.info(f"  - Augmented: {final_X.shape[0] - n_original}")
        logger.info(f"  - Increase: {(final_X.shape[0] / n_original - 1) * 100:.1f}%")

        return final_X, final_y

    def magnitude_warping(self, X: np.ndarray) -> np.ndarray:
        """
        Scale magnitude of sequences slightly.

        This simulates different market volatility regimes.
        Example: 0.98x = slightly calmer market, 1.02x = slightly more volatile

        Args:
            X: Sequences to transform (n_samples, seq_len, n_features)

        Returns:
            Scaled sequences with same shape
        """
        n_samples = len(X)

        # Random scale for each sample (same across all features)
        scales = np.random.uniform(
            self.magnitude_range[0],
            self.magnitude_range[1],
            size=(n_samples, 1, 1)
        )

        # Apply scaling
        X_warped = X * scales

        return X_warped

    def add_jitter(self, X: np.ndarray) -> np.ndarray:
        """
        Add small random noise to sequences.

        This simulates measurement noise and minor market fluctuations.

        Args:
            X: Sequences to transform (n_samples, seq_len, n_features)

        Returns:
            Jittered sequences with same shape
        """
        # Generate Gaussian noise with same shape
        noise = np.random.normal(0, self.jitter_std, X.shape)

        # Add noise to original data
        X_jittered = X + noise

        return X_jittered

    def validate_augmentation(
        self,
        X_original: np.ndarray,
        X_augmented: np.ndarray
    ) -> Dict[str, float]:
        """
        Validate that augmentation didn't corrupt data too much.

        Args:
            X_original: Original sequences
            X_augmented: Augmented sequences

        Returns:
            Dictionary of validation metrics
        """
        # Calculate correlation between original and augmented
        corr = np.corrcoef(
            X_original.flatten(),
            X_augmented.flatten()
        )[0, 1]

        # Calculate mean absolute difference
        mae = np.mean(np.abs(X_original - X_augmented))

        # Calculate relative difference
        rel_diff = mae / (np.mean(np.abs(X_original)) + 1e-8)

        metrics = {
            'correlation': corr,
            'mae': mae,
            'relative_difference': rel_diff
        }

        logger.info("Augmentation validation:")
        logger.info(f"  - Correlation: {corr:.4f} (should be >0.95)")
        logger.info(f"  - MAE: {mae:.6f}")
        logger.info(f"  - Relative diff: {rel_diff:.4f} (should be <0.05)")

        if corr < 0.95:
            logger.warning("⚠️  Low correlation - augmentation may be too aggressive!")
        if rel_diff > 0.05:
            logger.warning("⚠️  High relative difference - consider reducing magnitude/jitter!")

        return metrics


# Export for use in other modules
__all__ = ['TimeSeriesAugmenter']
