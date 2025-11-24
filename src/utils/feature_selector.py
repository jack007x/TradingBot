"""
Feature selection utilities for RL trading.
Reduces feature space to most important indicators.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
from loguru import logger


class SimpleFeatureSelector:
    """
    Select most important features for RL trading.
    Focuses on essential price action and technical indicators.
    """

    @staticmethod
    def get_essential_features() -> List[str]:
        """
        Get list of essential features for RL (10-15 features).

        These are hand-picked features that capture:
        - Price momentum
        - Trend direction
        - Volatility
        - Volume
        - Key support/resistance levels

        Returns:
            List of feature column names
        """
        return [
            # Price features (3)
            'returns',
            'log_returns',
            'close_open_ratio',

            # Trend indicators (3)
            'sma_cross_5_20',
            'dist_from_sma_20',
            'price_above_sma_200',

            # Momentum (3)
            'rsi_14',
            'macd',
            'macd_signal',

            # Volatility (2)
            'atr_14',
            'bb_width',

            # Volume (1)
            'volume_sma_ratio',

            # Additional momentum (1)
            'stoch_k'
        ]

    @staticmethod
    def select_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select essential features from dataframe.

        Args:
            df: Full dataframe with all features

        Returns:
            Tuple of (reduced dataframe, selected feature names)
        """
        essential = SimpleFeatureSelector.get_essential_features()

        # Check which features are available
        available = [f for f in essential if f in df.columns]

        if len(available) < 10:
            logger.warning(f"Only {len(available)} essential features found, expected 10-15")

        # Add close price (needed for environment)
        if 'close' not in available:
            available.insert(0, 'close')

        logger.info(f"Selected {len(available)} features for RL: {available}")

        return df[available], available

    @staticmethod
    def create_compact_state(df: pd.DataFrame, window_size: int = 10) -> np.ndarray:
        """
        Create compact state representation with limited window.

        Args:
            df: Dataframe with selected features
            window_size: Number of timesteps (default 10)

        Returns:
            Compact state array (samples, window_size, n_features)
        """
        # Get only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])

        # Normalize each feature
        normalized = (numeric_df - numeric_df.mean()) / (numeric_df.std() + 1e-8)

        # Create sequences
        data = normalized.values
        sequences = []

        for i in range(window_size, len(data)):
            sequences.append(data[i-window_size:i])

        return np.array(sequences, dtype=np.float32)
