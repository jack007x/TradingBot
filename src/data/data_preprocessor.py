"""
Data preprocessing for AI Trading Bot.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
from loguru import logger


class DataPreprocessor:
    """
    Preprocesses market data for machine learning models.
    Handles feature engineering, normalization, and sequence creation.
    """

    def __init__(
        self,
        scaling_method: str = 'minmax',
        feature_range: Tuple[float, float] = (0, 1)
    ):
        """
        Initialize the data preprocessor.

        Args:
            scaling_method: Method for scaling ('minmax', 'standard', 'robust')
            feature_range: Range for MinMax scaling
        """
        self.scaling_method = scaling_method
        self.feature_range = feature_range
        self.scalers: Dict[str, object] = {}
        self.feature_columns: List[str] = []

    def _create_scaler(self) -> object:
        """Create a scaler based on the configured method."""
        if self.scaling_method == 'minmax':
            return MinMaxScaler(feature_range=self.feature_range)
        elif self.scaling_method == 'standard':
            return StandardScaler()
        elif self.scaling_method == 'robust':
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")

    def add_technical_indicators(
        self,
        df: pd.DataFrame,
        include_all: bool = True
    ) -> pd.DataFrame:
        """
        Add comprehensive technical indicators.

        Args:
            df: DataFrame with OHLCV data
            include_all: Include all available indicators

        Returns:
            DataFrame with technical indicators
        """
        df = df.copy()

        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")

        # ============== Price-based Features ==============
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Price ratios
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']

        # ============== Moving Averages ==============
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

        # Moving average crossovers
        df['sma_cross_5_20'] = (df['sma_5'] > df['sma_20']).astype(int)
        df['sma_cross_20_50'] = (df['sma_20'] > df['sma_50']).astype(int)
        df['price_above_sma_200'] = (df['close'] > df['sma_200']).astype(int)

        # Distance from moving averages
        df['dist_from_sma_20'] = (df['close'] - df['sma_20']) / df['sma_20']
        df['dist_from_sma_50'] = (df['close'] - df['sma_50']) / df['sma_50']

        # ============== Momentum Indicators ==============
        # RSI
        for period in [7, 14, 21]:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-10)
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # Stochastic Oscillator
        for period in [14]:
            low_min = df['low'].rolling(window=period).min()
            high_max = df['high'].rolling(window=period).max()
            df[f'stoch_k_{period}'] = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-10)
            df[f'stoch_d_{period}'] = df[f'stoch_k_{period}'].rolling(window=3).mean()

        # Williams %R
        df['williams_r'] = -100 * (df['high'].rolling(14).max() - df['close']) / \
                          (df['high'].rolling(14).max() - df['low'].rolling(14).min() + 1e-10)

        # CCI (Commodity Channel Index)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = typical_price.rolling(window=20).mean()
        mad = typical_price.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
        df['cci'] = (typical_price - sma_tp) / (0.015 * mad + 1e-10)

        # Rate of Change
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = df['close'].pct_change(periods=period) * 100

        # Momentum
        df['momentum_10'] = df['close'] - df['close'].shift(10)

        # ============== Volatility Indicators ==============
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        for period in [7, 14, 21]:
            df[f'atr_{period}'] = true_range.rolling(window=period).mean()
            df[f'atr_pct_{period}'] = df[f'atr_{period}'] / df['close']

        # Bollinger Bands
        for period in [20]:
            middle = df['close'].rolling(window=period).mean()
            std = df['close'].rolling(window=period).std()
            df[f'bb_upper_{period}'] = middle + 2 * std
            df[f'bb_lower_{period}'] = middle - 2 * std
            df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / middle
            df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / \
                                          (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'] + 1e-10)

        # Historical Volatility
        df['volatility_20'] = df['log_returns'].rolling(window=20).std() * np.sqrt(252)

        # ============== Trend Indicators ==============
        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # ADX (Average Directional Index)
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr = true_range
        atr_14 = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr_14)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr_14)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df['adx'] = dx.rolling(14).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di

        # Aroon
        period = 25
        df['aroon_up'] = 100 * df['high'].rolling(period + 1).apply(
            lambda x: x.argmax()) / period
        df['aroon_down'] = 100 * df['low'].rolling(period + 1).apply(
            lambda x: x.argmin()) / period
        df['aroon_oscillator'] = df['aroon_up'] - df['aroon_down']

        # ============== Volume Indicators ==============
        # Volume Moving Averages
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_sma_20'] + 1e-10)

        # OBV (On-Balance Volume)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

        # Volume Price Trend
        df['vpt'] = (df['volume'] * df['close'].pct_change()).fillna(0).cumsum()

        # Money Flow Index
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

        positive_mf = positive_flow.rolling(14).sum()
        negative_mf = negative_flow.rolling(14).sum()
        mfi = 100 - (100 / (1 + positive_mf / (negative_mf + 1e-10)))
        df['mfi'] = mfi

        # ============== Pattern Features ==============
        # Candlestick patterns (simple)
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['lower_shadow'] = df[['close', 'open']].min(axis=1) - df['low']
        df['is_bullish'] = (df['close'] > df['open']).astype(int)

        # Doji detection
        df['is_doji'] = (df['body_size'] < (df['high'] - df['low']) * 0.1).astype(int)

        # Price position in range
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)

        # ============== Time Features ==============
        if isinstance(df.index, pd.DatetimeIndex):
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['day_of_month'] = df.index.day
            df['month'] = df.index.month

            # Cyclical encoding
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        logger.info(f"Added {len(df.columns)} features to DataFrame")
        return df

    def prepare_sequences(
        self,
        df: pd.DataFrame,
        sequence_length: int = 60,
        target_column: str = 'close',
        prediction_horizon: int = 1,
        target_type: str = 'regression'
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare sequences for time series models.

        Args:
            df: DataFrame with features
            sequence_length: Number of timesteps in each sequence
            target_column: Column to predict
            prediction_horizon: Steps ahead to predict
            target_type: 'regression' or 'classification'

        Returns:
            Tuple of (X sequences, y targets, feature names)
        """
        # Remove non-numeric columns
        df = df.select_dtypes(include=[np.number])

        # Drop rows with NaN
        df = df.dropna()

        if len(df) < sequence_length + prediction_horizon:
            raise ValueError(f"Not enough data: {len(df)} rows, need {sequence_length + prediction_horizon}")

        # Get feature columns
        feature_cols = [col for col in df.columns
                       if col not in ['symbol', 'timeframe']]
        self.feature_columns = feature_cols

        # Create sequences
        X, y = [], []
        data = df[feature_cols].values
        target_idx = feature_cols.index(target_column)

        for i in range(len(data) - sequence_length - prediction_horizon + 1):
            X.append(data[i:i + sequence_length])

            if target_type == 'regression':
                y.append(data[i + sequence_length + prediction_horizon - 1, target_idx])
            else:
                # Classification: predict direction
                current_price = data[i + sequence_length - 1, target_idx]
                future_price = data[i + sequence_length + prediction_horizon - 1, target_idx]
                y.append(1 if future_price > current_price else 0)

        X = np.array(X)
        y = np.array(y)

        logger.info(f"Created {len(X)} sequences of shape {X.shape[1:]}")
        return X, y, feature_cols

    def scale_data(
        self,
        df: pd.DataFrame,
        fit: bool = True,
        scaler_name: str = 'default'
    ) -> pd.DataFrame:
        """
        Scale DataFrame features.

        Args:
            df: DataFrame to scale
            fit: Whether to fit the scaler
            scaler_name: Name for the scaler (for saving/loading)

        Returns:
            Scaled DataFrame
        """
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if fit:
            scaler = self._create_scaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            self.scalers[scaler_name] = scaler
        else:
            if scaler_name not in self.scalers:
                raise ValueError(f"Scaler '{scaler_name}' not found. Fit first.")
            df[numeric_cols] = self.scalers[scaler_name].transform(df[numeric_cols])

        return df

    def inverse_scale(
        self,
        data: np.ndarray,
        scaler_name: str = 'default',
        column_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Inverse transform scaled data.

        Args:
            data: Scaled data
            scaler_name: Name of scaler to use
            column_idx: If provided, only inverse transform specific column

        Returns:
            Original scale data
        """
        if scaler_name not in self.scalers:
            raise ValueError(f"Scaler '{scaler_name}' not found")

        scaler = self.scalers[scaler_name]

        if column_idx is not None:
            # Create dummy array with same number of features
            dummy = np.zeros((len(data), scaler.n_features_in_))
            dummy[:, column_idx] = data.flatten()
            result = scaler.inverse_transform(dummy)
            return result[:, column_idx]

        return scaler.inverse_transform(data)

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        shuffle: bool = True,
        stratify: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Split data into train, validation, and test sets with stratification.

        Args:
            X: Features
            y: Targets
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            shuffle: Whether to shuffle data
            stratify: Use stratified split to preserve class distribution (CRITICAL for imbalanced data)

        Returns:
            Dictionary with train/val/test splits
        """
        test_ratio = 1 - train_ratio - val_ratio

        # CRITICAL FIX: Use stratified split to maintain class distribution
        stratify_param = y if stratify else None

        # First split: train and temp (val + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=(val_ratio + test_ratio),
            shuffle=shuffle,
            stratify=stratify_param,
            random_state=42 if stratify else None
        )

        # Second split: val and test (also stratified)
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        stratify_temp = y_temp if stratify else None
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_ratio_adjusted),
            shuffle=shuffle,
            stratify=stratify_temp,
            random_state=42 if stratify else None
        )

        logger.info(f"Split data: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

        if stratify:
            # Log class distributions to verify stratification
            unique_train, counts_train = np.unique(y_train, return_counts=True)
            unique_val, counts_val = np.unique(y_val, return_counts=True)
            unique_test, counts_test = np.unique(y_test, return_counts=True)

            logger.info("Stratified class distributions:")
            logger.info(f"  Train: {dict(zip(unique_train, counts_train))}")
            logger.info(f"  Val:   {dict(zip(unique_val, counts_val))}")
            logger.info(f"  Test:  {dict(zip(unique_test, counts_test))}")

        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }

    def calculate_adaptive_threshold(
        self,
        df: pd.DataFrame,
        atr_column: str = 'atr_14',
        percentile: float = 50.0,
        min_threshold: float = 0.0005,
        max_threshold: float = 0.01
    ) -> float:
        """
        Calculate adaptive threshold based on ATR (Average True Range).

        Higher volatility (higher ATR) requires higher threshold to distinguish
        meaningful moves from noise.

        Args:
            df: DataFrame with ATR column
            atr_column: Name of ATR column
            percentile: Percentile of ATR to use (50 = median)
            min_threshold: Minimum threshold (0.05%)
            max_threshold: Maximum threshold (1%)

        Returns:
            Adaptive threshold as percentage
        """
        if atr_column not in df.columns:
            logger.warning(f"ATR column '{atr_column}' not found, using default threshold")
            return 0.002

        # Get ATR at specified percentile
        atr_value = np.percentile(df[atr_column].dropna(), percentile)

        # Get median close price
        median_close = df['close'].median()

        # Calculate threshold as ATR / close (as percentage)
        # CRITICAL FIX: Use 200% of ATR (was 70% - still too conservative!)
        # For volatile assets like XAUUSD, need higher threshold to reduce neutral class
        # Target: 0.3-0.5% threshold for XAUUSD to get balanced classes
        threshold = (atr_value / median_close) * 2.0

        # Clamp to min/max
        threshold = max(min_threshold, min(max_threshold, threshold))

        logger.info(f"Adaptive threshold calculated: {threshold:.6f} ({threshold*100:.4f}%)")
        logger.info(f"Based on ATR={atr_value:.4f}, median_close={median_close:.2f}")
        logger.info(f"Using 200% of ATR (increased from 70%) for meaningful price moves")

        return threshold

    def create_direction_labels(
        self,
        df: pd.DataFrame,
        prediction_horizon: int = 1,
        threshold: Optional[float] = None,
        target_col: str = 'close',
        adaptive: bool = True
    ) -> np.ndarray:
        """
        Create direction labels for classification.

        Labels:
        - 0: Down (price decreases by more than threshold)
        - 1: Neutral (price change within threshold)
        - 2: Up (price increases by more than threshold)

        Args:
            df: DataFrame with price data
            prediction_horizon: How many steps ahead to predict
            threshold: Minimum change to consider as up/down (as percentage)
                      If None and adaptive=True, will calculate from ATR
            target_col: Column name for target price
            adaptive: Use adaptive threshold based on ATR

        Returns:
            Array of direction labels (0=down, 1=neutral, 2=up)
        """
        # Calculate threshold adaptively if requested
        if threshold is None and adaptive:
            threshold = self.calculate_adaptive_threshold(df)
        elif threshold is None:
            threshold = 0.002  # Default 0.2%

        logger.info(f"Using threshold: {threshold:.6f} ({threshold*100:.4f}%)")

        prices = df[target_col].values
        labels = []

        for i in range(len(prices) - prediction_horizon):
            current = prices[i]
            future = prices[i + prediction_horizon]

            # Calculate percentage change
            pct_change = (future - current) / current

            # Classify
            if pct_change < -threshold:
                labels.append(0)  # Down
            elif pct_change > threshold:
                labels.append(2)  # Up
            else:
                labels.append(1)  # Neutral

        labels = np.array(labels)

        # Log class distribution
        unique, counts = np.unique(labels, return_counts=True)
        dist = dict(zip(unique, counts))
        total = len(labels)
        logger.info(f"Direction label distribution:")
        logger.info(f"  Down:    {dist.get(0, 0):5d} ({dist.get(0, 0)/total*100:5.1f}%)")
        logger.info(f"  Neutral: {dist.get(1, 0):5d} ({dist.get(1, 0)/total*100:5.1f}%)")
        logger.info(f"  Up:      {dist.get(2, 0):5d} ({dist.get(2, 0)/total*100:5.1f}%)")

        return labels

    def prepare_directional_sequences(
        self,
        df: pd.DataFrame,
        sequence_length: int = 60,
        prediction_horizon: int = 1,
        direction_threshold: Optional[float] = None,
        adaptive_threshold: bool = True,
        target_col: str = 'close',
        feature_cols: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare sequences with direction labels for classification.

        Args:
            df: DataFrame with features
            sequence_length: Length of input sequences
            prediction_horizon: How many steps ahead to predict
            direction_threshold: Minimum change for up/down classification
                                If None, will use adaptive threshold based on ATR
            adaptive_threshold: Use adaptive threshold if direction_threshold is None
            target_col: Target column name
            feature_cols: Feature columns to use (None = all numeric)

        Returns:
            Tuple of (X sequences, y direction labels, feature column names)
        """
        # Select features
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Remove target column from features
            if target_col in feature_cols:
                feature_cols.remove(target_col)

        # Convert to numpy
        data = df[feature_cols].values

        # Create direction labels with adaptive threshold
        # Need full df for ATR calculation
        direction_labels = self.create_direction_labels(
            df,
            prediction_horizon=prediction_horizon,
            threshold=direction_threshold,
            target_col=target_col,
            adaptive=adaptive_threshold
        )

        # Create sequences
        X = []
        y = []

        for i in range(sequence_length, len(data)):
            # Label is for bar AFTER the sequence window
            # Sequence: [i-sequence_length:i] (bars 0 to i-1)
            # Label: direction_labels[i] (comparing bar i to bar i+prediction_horizon)
            if i < len(direction_labels):
                X.append(data[i - sequence_length:i])
                y.append(direction_labels[i])

        X = np.array(X)
        y = np.array(y)

        logger.info(f"Created {len(X)} directional sequences of shape {X.shape[1:]}")
        logger.info(f"Label distribution: {np.bincount(y)}")

        return X, y, feature_cols

    def create_regression_labels(
        self,
        df: pd.DataFrame,
        horizons: List[int] = [1, 4, 12, 24],
        target_col: str = 'close'
    ) -> pd.DataFrame:
        """
        Create CONTINUOUS return labels for regression (NOT classification).

        WHY THIS IS BETTER:
        - NO class imbalance issues
        - NO threshold dependency
        - Captures magnitude of moves (0.5% vs 2% distinction)
        - Natural probability distribution
        - More information per label

        Args:
            df: DataFrame with price data
            horizons: List of forward-looking periods (e.g., [1, 4, 12, 24] for 1h, 4h, 12h, 24h)
            target_col: Price column to use

        Returns:
            DataFrame with continuous return labels added
        """
        df = df.copy()

        # Create forward returns for multiple horizons
        for h in horizons:
            # Forward percentage return
            df[f'target_return_{h}h'] = df[target_col].pct_change(h).shift(-h)

        # Primary target (typically mid-horizon, e.g., 4h)
        primary_horizon = horizons[1] if len(horizons) > 1 else horizons[0]
        df['target_return'] = df[f'target_return_{primary_horizon}h']

        # Log statistics
        logger.info(f"Created regression labels for horizons: {horizons}")
        for h in horizons:
            col = f'target_return_{h}h'
            returns = df[col].dropna()
            logger.info(f"  {h}h returns: Mean={returns.mean():.6f}, Std={returns.std():.6f}, "
                       f"Min={returns.min():.6f}, Max={returns.max():.6f}")

        # Log distribution info (continuous, not discrete!)
        logger.info(f"Primary target (target_return): {primary_horizon}h forward return")
        logger.info(f"  Positive returns: {(df['target_return'] > 0).sum()} "
                   f"({(df['target_return'] > 0).sum() / len(df) * 100:.1f}%)")
        logger.info(f"  Negative returns: {(df['target_return'] < 0).sum()} "
                   f"({(df['target_return'] < 0).sum() / len(df) * 100:.1f}%)")
        logger.info(f"  Near-zero returns: {(df['target_return'].abs() < 0.001).sum()} "
                   f"({(df['target_return'].abs() < 0.001).sum() / len(df) * 100:.1f}%)")

        return df

    def prepare_regression_sequences(
        self,
        df: pd.DataFrame,
        sequence_length: int = 60,
        target_col: str = 'target_return',
        feature_cols: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare sequences with CONTINUOUS return labels for regression.

        KEY DIFFERENCES from directional sequences:
        - Labels are continuous floats (not 0/1/2 classes)
        - NO threshold-based binning
        - NO class imbalance
        - NO need for SMOTE or balancing

        Args:
            df: DataFrame with features and regression labels
            sequence_length: Length of input sequences
            target_col: Target return column (continuous)
            feature_cols: Feature columns to use (None = all numeric except targets)

        Returns:
            Tuple of (X sequences, y continuous returns, feature column names)
        """
        # Select features (exclude target columns)
        if feature_cols is None:
            # Get all numeric columns
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            # Remove all target columns
            target_columns = [col for col in feature_cols if 'target_return' in col]
            for col in target_columns:
                if col in feature_cols:
                    feature_cols.remove(col)

            logger.info(f"Auto-selected {len(feature_cols)} feature columns")

        # Check target column exists
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")

        # Convert to numpy
        data = df[feature_cols].values
        targets = df[target_col].values

        # Create sequences
        X = []
        y = []

        for i in range(sequence_length, len(data)):
            # Sequence: [i-sequence_length:i] (bars 0 to i-1)
            # Label: targets[i] (forward return from bar i)
            if not np.isnan(targets[i]):  # Skip NaN targets
                X.append(data[i - sequence_length:i])
                y.append(targets[i])

        X = np.array(X)
        y = np.array(y)

        logger.info(f"Created {len(X)} regression sequences of shape {X.shape[1:]}")
        logger.info(f"Target statistics:")
        logger.info(f"  Mean: {y.mean():.6f}")
        logger.info(f"  Std:  {y.std():.6f}")
        logger.info(f"  Min:  {y.min():.6f}")
        logger.info(f"  Max:  {y.max():.6f}")
        logger.info(f"  Median: {np.median(y):.6f}")

        # Check for extreme outliers (might want to clip)
        outlier_threshold = 3 * y.std()
        outliers = np.abs(y) > outlier_threshold
        if outliers.sum() > 0:
            logger.warning(f"Found {outliers.sum()} outliers (>{outlier_threshold:.4f})")
            logger.warning(f"Consider clipping extreme values")

        return X, y, feature_cols

    def create_chart_images(
        self,
        df: pd.DataFrame,
        window_size: int = 60,
        image_size: Tuple[int, int] = (64, 64)
    ) -> np.ndarray:
        """
        Create chart images for CNN from OHLCV data.

        Args:
            df: DataFrame with OHLCV data
            window_size: Number of candles per image
            image_size: Output image dimensions

        Returns:
            Array of chart images
        """
        import matplotlib.pyplot as plt
        from io import BytesIO
        from PIL import Image

        images = []

        for i in range(window_size, len(df)):
            window = df.iloc[i - window_size:i]

            # Create figure
            fig, ax = plt.subplots(figsize=(2, 2), dpi=32)
            ax.set_facecolor('black')
            fig.patch.set_facecolor('black')

            # Plot candlesticks
            for j, (idx, row) in enumerate(window.iterrows()):
                color = 'green' if row['close'] >= row['open'] else 'red'

                # Body
                ax.plot([j, j], [row['open'], row['close']],
                       color=color, linewidth=2)
                # Wicks
                ax.plot([j, j], [row['low'], row['high']],
                       color=color, linewidth=0.5)

            ax.axis('off')
            plt.tight_layout(pad=0)

            # Convert to image array
            buf = BytesIO()
            fig.savefig(buf, format='png', facecolor='black',
                       edgecolor='none', bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            plt.close(fig)

            img = Image.open(buf).convert('L')  # Grayscale
            img = img.resize(image_size)
            images.append(np.array(img))

        return np.array(images).reshape(-1, 1, *image_size)

    def save_scaler(
        self,
        scaler_name: str,
        filepath: Union[str, Path]
    ) -> None:
        """Save a scaler to file."""
        if scaler_name not in self.scalers:
            raise ValueError(f"Scaler '{scaler_name}' not found")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scalers[scaler_name], filepath)
        logger.info(f"Saved scaler to {filepath}")

    def load_scaler(
        self,
        scaler_name: str,
        filepath: Union[str, Path]
    ) -> None:
        """Load a scaler from file."""
        filepath = Path(filepath)
        self.scalers[scaler_name] = joblib.load(filepath)
        logger.info(f"Loaded scaler from {filepath}")

    def get_feature_names(self) -> List[str]:
        """Get list of feature column names."""
        return self.feature_columns.copy()
