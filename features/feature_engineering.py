"""
Feature Engineering Module
===========================
Builds ML features from OHLCV data for price prediction.

CRITICAL: All features are calculated using only past data to avoid look-ahead bias.

Usage:
    from features import FeatureEngineer
    from config import get_config

    config = get_config()
    engineer = FeatureEngineer(config)

    # Build features and labels
    df_features = engineer.build_features(df_ohlcv)

    # Get feature matrix for training
    X, y, feature_names = engineer.prepare_training_data(df_features)
"""

import logging
from typing import Tuple, List, Optional
import pandas as pd
import numpy as np

from config.config_loader import Config

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Builds features for ML model training and inference.

    Features include:
    - Price-based: returns, log-returns, range, body size
    - Technical indicators: EMA, RSI, ATR, Bollinger Bands, MACD, Stochastic
    - Time-based: hour, day of week, session indicators
    - Volatility: rolling std, ATR ratios

    All features use only past information (no look-ahead bias).
    """

    def __init__(self, config: Config):
        """
        Initialize feature engineer.

        Args:
            config: Configuration object
        """
        self.config = config
        self.lookback = config.features.lookback_window
        self.horizon = config.model.prediction_horizon_bars
        self.return_threshold = config.model.return_threshold_percent / 100

        # Technical indicator parameters
        self.ti = config.features.technical_indicators

        # Feature lists
        self._feature_names: List[str] = []

    def build_features(
            self,
            df: pd.DataFrame,
            include_labels: bool = True
    ) -> pd.DataFrame:
        """
        Build all features from OHLCV data.

        Args:
            df: OHLCV DataFrame with columns [time, open, high, low, close, volume]
            include_labels: Whether to add target labels (for training)

        Returns:
            pd.DataFrame: Data with all features added
        """
        if df.empty:
            logger.warning("Empty DataFrame, returning empty")
            return df

        df = df.copy()
        logger.info(f"Building features for {len(df)} bars")

        # Ensure sorted by time
        df = df.sort_values('time').reset_index(drop=True)

        # Build all feature groups
        df = self._add_price_features(df)
        df = self._add_technical_indicators(df)
        df = self._add_time_features(df)
        df = self._add_volatility_features(df)

        # Add labels if requested
        if include_labels:
            df = self._add_labels(df)

        # Store feature names (exclude non-feature columns)
        non_features = ['time', 'open', 'high', 'low', 'close', 'volume',
                        'spread', 'label', 'future_return']
        self._feature_names = [c for c in df.columns if c not in non_features]

        logger.info(f"Built {len(self._feature_names)} features")
        return df

    def prepare_training_data(
            self,
            df: pd.DataFrame,
            dropna: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare feature matrix X and target vector y for training.

        Args:
            df: DataFrame with features (output of build_features)
            dropna: Drop rows with NaN values

        Returns:
            Tuple[X, y, feature_names]: Feature matrix, labels, feature names
        """
        if 'label' not in df.columns:
            raise ValueError("DataFrame must have 'label' column. Run build_features with include_labels=True")

        # Get feature columns
        feature_cols = self.get_feature_names()
        if not feature_cols:
            feature_cols = [c for c in df.columns if c not in
                           ['time', 'open', 'high', 'low', 'close', 'volume',
                            'spread', 'label', 'future_return']]

        # Extract X and y
        df_clean = df[feature_cols + ['label']].copy()

        if dropna:
            df_clean = df_clean.dropna()

        X = df_clean[feature_cols].values
        y = df_clean['label'].values

        logger.info(f"Prepared training data: X shape {X.shape}, y shape {y.shape}")
        return X, y, feature_cols

    def prepare_inference_data(
            self,
            df: pd.DataFrame
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare feature matrix for inference (no labels needed).

        Args:
            df: DataFrame with features

        Returns:
            Tuple[X, feature_names]: Feature matrix and names
        """
        feature_cols = self.get_feature_names()
        if not feature_cols:
            feature_cols = [c for c in df.columns if c not in
                           ['time', 'open', 'high', 'low', 'close', 'volume',
                            'spread', 'label', 'future_return']]

        # Get last row for inference
        X = df[feature_cols].iloc[[-1]].values

        return X, feature_cols

    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self._feature_names.copy()

    # =========================================================================
    # Price-Based Features
    # =========================================================================

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""

        # Simple returns (percentage change)
        df['return_1'] = df['close'].pct_change(1)
        df['return_2'] = df['close'].pct_change(2)
        df['return_5'] = df['close'].pct_change(5)
        df['return_10'] = df['close'].pct_change(10)

        # Log returns
        df['log_return_1'] = np.log(df['close'] / df['close'].shift(1))
        df['log_return_5'] = np.log(df['close'] / df['close'].shift(5))

        # Price range (high - low)
        df['price_range'] = (df['high'] - df['low']) / df['close']

        # Body size (|close - open|)
        df['body_size'] = np.abs(df['close'] - df['open']) / df['close']

        # Body direction (1 = bullish, -1 = bearish)
        df['body_direction'] = np.sign(df['close'] - df['open'])

        # Upper shadow
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']

        # Lower shadow
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']

        # Distance from recent high/low
        df['dist_from_high_20'] = (df['close'] - df['high'].rolling(20).max()) / df['close']
        df['dist_from_low_20'] = (df['close'] - df['low'].rolling(20).min()) / df['close']

        # Gap (open vs previous close)
        df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)

        return df

    # =========================================================================
    # Technical Indicators
    # =========================================================================

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators."""

        close = df['close']
        high = df['high']
        low = df['low']

        # EMAs
        for period in self.ti['ema_periods']:
            df[f'ema_{period}'] = close.ewm(span=period, adjust=False).mean()
            # Price relative to EMA
            df[f'close_ema_{period}_ratio'] = close / df[f'ema_{period}'] - 1

        # EMA crossovers
        df['ema_8_21_cross'] = (df['ema_8'] > df['ema_21']).astype(int)
        df['ema_21_50_cross'] = (df['ema_21'] > df['ema_50']).astype(int)

        # RSI
        rsi_period = self.ti['rsi_period']
        df['rsi'] = self._calculate_rsi(close, rsi_period)
        df['rsi_normalized'] = (df['rsi'] - 50) / 50  # Normalize to [-1, 1]

        # ATR (Average True Range)
        atr_period = self.ti['atr_period']
        df['atr'] = self._calculate_atr(high, low, close, atr_period)
        df['atr_percent'] = df['atr'] / close * 100

        # Bollinger Bands
        bb_period = self.ti['bollinger_period']
        bb_std = self.ti['bollinger_std']
        df['bb_middle'] = close.rolling(bb_period).mean()
        bb_std_val = close.rolling(bb_period).std()
        df['bb_upper'] = df['bb_middle'] + bb_std * bb_std_val
        df['bb_lower'] = df['bb_middle'] - bb_std * bb_std_val
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # MACD
        macd_fast = self.ti['macd_fast']
        macd_slow = self.ti['macd_slow']
        macd_signal = self.ti['macd_signal']
        ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=macd_signal, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        df['macd_normalized'] = df['macd'] / close * 100

        # Stochastic
        stoch_k = self.ti['stochastic_k']
        stoch_d = self.ti['stochastic_d']
        lowest_low = low.rolling(stoch_k).min()
        highest_high = high.rolling(stoch_k).max()
        df['stoch_k'] = 100 * (close - lowest_low) / (highest_high - lowest_low)
        df['stoch_d'] = df['stoch_k'].rolling(stoch_d).mean()
        df['stoch_k_normalized'] = (df['stoch_k'] - 50) / 50

        # ADX (Average Directional Index) - simplified
        df['adx'] = self._calculate_adx(high, low, close, 14)

        # On-Balance Volume indicator (if volume available)
        if 'volume' in df.columns:
            df['obv'] = (np.sign(close.diff()) * df['volume']).cumsum()
            df['obv_ema'] = df['obv'].ewm(span=20).mean()
            df['obv_signal'] = df['obv'] - df['obv_ema']

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_atr(
            self,
            high: pd.Series,
            low: pd.Series,
            close: pd.Series,
            period: int
    ) -> pd.Series:
        """Calculate Average True Range."""
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(span=period, adjust=False).mean()

        return atr

    def _calculate_adx(
            self,
            high: pd.Series,
            low: pd.Series,
            close: pd.Series,
            period: int
    ) -> pd.Series:
        """Calculate Average Directional Index (simplified)."""
        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        atr = self._calculate_atr(high, low, close, period)

        plus_di = 100 * (plus_dm.ewm(span=period).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period).mean() / atr)

        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(span=period).mean()

        return adx

    # =========================================================================
    # Time-Based Features
    # =========================================================================

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""

        # Extract time components
        time = df['time']

        # Hour of day (0-23)
        df['hour'] = time.dt.hour

        # Cyclical encoding of hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # Day of week (0=Monday, 6=Sunday)
        df['day_of_week'] = time.dt.dayofweek

        # Cyclical encoding of day
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Trading sessions (approximate UTC times)
        hour = df['hour']

        # Asian session: 00:00 - 08:00 UTC
        df['is_asian_session'] = ((hour >= 0) & (hour < 8)).astype(int)

        # London session: 07:00 - 16:00 UTC
        df['is_london_session'] = ((hour >= 7) & (hour < 16)).astype(int)

        # New York session: 12:00 - 21:00 UTC
        df['is_ny_session'] = ((hour >= 12) & (hour < 21)).astype(int)

        # London-NY overlap: 12:00 - 16:00 UTC (most liquid)
        df['is_overlap_session'] = ((hour >= 12) & (hour < 16)).astype(int)

        # Weekend indicator (should be 0 for trading data)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # Is market open hour (high activity)
        df['is_active_hour'] = ((hour >= 7) & (hour < 21)).astype(int)

        return df

    # =========================================================================
    # Volatility Features
    # =========================================================================

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features."""

        close = df['close']
        returns = close.pct_change()

        # Rolling standard deviation of returns
        for period in self.config.features.volatility_features['rolling_std_periods']:
            df[f'volatility_{period}'] = returns.rolling(period).std()

        # Normalized volatility (current vs longer term)
        df['volatility_ratio'] = df['volatility_5'] / df['volatility_20']

        # ATR ratio (short vs long)
        atr_short = self._calculate_atr(df['high'], df['low'], close, 5)
        atr_long = self._calculate_atr(df['high'], df['low'], close, 20)
        df['atr_ratio'] = atr_short / atr_long

        # Parkinson volatility (based on high-low)
        log_hl = np.log(df['high'] / df['low'])
        df['parkinson_vol'] = np.sqrt(log_hl ** 2 / (4 * np.log(2)))

        # Realized volatility (rolling)
        df['realized_vol_10'] = np.sqrt((returns ** 2).rolling(10).sum())

        # Volatility percentile (current vol vs historical)
        df['vol_percentile'] = df['volatility_10'].rolling(100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5,
            raw=False
        )

        return df

    # =========================================================================
    # Label Generation
    # =========================================================================

    def _add_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add target labels for supervised learning.

        Label:
            1  = Up (return > threshold)
            0  = Neutral (|return| <= threshold)
            -1 = Down (return < -threshold)
        """
        # Future return over horizon bars
        df['future_return'] = df['close'].shift(-self.horizon) / df['close'] - 1

        # Convert to class labels
        df['label'] = 0  # Neutral by default
        df.loc[df['future_return'] > self.return_threshold, 'label'] = 1
        df.loc[df['future_return'] < -self.return_threshold, 'label'] = -1

        # Log label distribution
        label_counts = df['label'].value_counts()
        total = len(df.dropna(subset=['label']))
        if total > 0:
            logger.info(f"Label distribution:")
            for label, count in sorted(label_counts.items()):
                pct = count / total * 100
                label_name = {1: 'Up', 0: 'Neutral', -1: 'Down'}[label]
                logger.info(f"  {label_name} ({label}): {count} ({pct:.1f}%)")

        return df

    def get_minimum_bars_required(self) -> int:
        """
        Get minimum number of bars required to calculate all features.

        Returns:
            int: Minimum bars needed
        """
        # Maximum lookback from all indicators
        max_lookback = max(
            self.lookback,
            max(self.ti['ema_periods']),
            self.ti['bollinger_period'],
            self.ti['macd_slow'] + self.ti['macd_signal'],
            100,  # For percentile calculations
        )
        return max_lookback + self.horizon + 10  # Extra buffer


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from config import get_config
    from data.data_fetcher import DataFetcher

    config = get_config()
    engineer = FeatureEngineer(config)
    fetcher = DataFetcher(config)

    # Fetch demo data
    from datetime import datetime
    start = datetime(2024, 1, 1)
    end = datetime(2024, 6, 1)
    df = fetcher.fetch_historical(start, end)

    print(f"Raw data: {len(df)} rows")
    print(f"Columns: {list(df.columns)}")

    # Build features
    df_features = engineer.build_features(df, include_labels=True)

    print(f"\nWith features: {len(df_features)} rows")
    print(f"Total columns: {len(df_features.columns)}")
    print(f"Feature count: {len(engineer.get_feature_names())}")

    # Show feature names
    print("\nFeature names:")
    for name in engineer.get_feature_names()[:20]:
        print(f"  - {name}")
    if len(engineer.get_feature_names()) > 20:
        print(f"  ... and {len(engineer.get_feature_names()) - 20} more")

    # Prepare training data
    X, y, feature_names = engineer.prepare_training_data(df_features)
    print(f"\nTraining data shape: X={X.shape}, y={y.shape}")

    # Show some statistics
    print("\nFeature statistics (first 5):")
    df_stats = pd.DataFrame(X, columns=feature_names).describe()
    print(df_stats.iloc[:, :5])
