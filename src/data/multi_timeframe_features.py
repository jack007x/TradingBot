"""
Multi-Timeframe Feature Generator
Generates features from multiple timeframes for better context.

Higher timeframes provide:
- Trend direction
- Support/resistance levels
- Volatility regime
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from loguru import logger
import MetaTrader5 as mt5


class MultiTimeframeFeatureGenerator:
    """
    Generate features from multiple timeframes.

    Example:
    - M5: Entry timing
    - M15: Short-term momentum
    - H1: Intraday trend
    - H4: Swing trend
    - D1: Major trend
    """

    TIMEFRAME_MAP = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1,
        'W1': mt5.TIMEFRAME_W1,
    }

    def __init__(
        self,
        timeframes: List[str] = ['M15', 'H1', 'H4', 'D1'],
        bars_per_tf: int = 100
    ):
        """
        Initialize multi-timeframe feature generator.

        Args:
            timeframes: List of timeframes to use
            bars_per_tf: Number of bars to fetch per timeframe
        """
        self.timeframes = timeframes
        self.bars_per_tf = bars_per_tf

        logger.info(f"MultiTimeframeFeatureGenerator initialized")
        logger.info(f"  Timeframes: {timeframes}")
        logger.info(f"  Bars per TF: {bars_per_tf}")

    def fetch_mtf_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for all configured timeframes.

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary of timeframe -> DataFrame
        """
        mtf_data = {}

        for tf_name in self.timeframes:
            tf = self.TIMEFRAME_MAP.get(tf_name)
            if tf is None:
                logger.warning(f"Unknown timeframe: {tf_name}")
                continue

            try:
                # Fetch data from MT5
                rates = mt5.copy_rates_from_pos(symbol, tf, 0, self.bars_per_tf)

                if rates is None or len(rates) == 0:
                    logger.warning(f"Failed to fetch {tf_name} data for {symbol}")
                    continue

                # Convert to DataFrame
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)

                # Rename columns to standard names
                if 'tick_volume' in df.columns:
                    df = df.rename(columns={'tick_volume': 'volume'})

                mtf_data[tf_name] = df
                logger.debug(f"Fetched {len(df)} bars for {tf_name}")

            except Exception as e:
                logger.error(f"Error fetching {tf_name} data: {e}")
                continue

        return mtf_data

    def generate_features(self, symbol: str) -> np.ndarray:
        """
        Generate multi-timeframe features.

        Features per timeframe:
        1. Trend direction (SMA 20 > SMA 50)
        2. Trend strength (ADX)
        3. Momentum (RSI)
        4. Volatility regime (ATR percentile)
        5. Distance from key levels (BB position)
        6. Price vs MAs

        Returns:
            Feature array with MTF context
        """
        mtf_data = self.fetch_mtf_data(symbol)

        if not mtf_data:
            logger.error(f"No MTF data available for {symbol}")
            return np.array([])

        all_features = []

        for tf_name, df in mtf_data.items():
            tf_features = self._calculate_tf_features(df, tf_name)
            all_features.extend(tf_features)

        # Add cross-timeframe features
        cross_features = self._calculate_cross_tf_features(mtf_data)
        all_features.extend(cross_features)

        logger.debug(f"Generated {len(all_features)} MTF features")

        return np.array(all_features)

    def _calculate_tf_features(
        self,
        df: pd.DataFrame,
        tf_name: str
    ) -> List[float]:
        """Calculate features for single timeframe."""
        features = []

        close = df['close']
        high = df['high']
        low = df['low']

        # 1. Trend Direction
        sma_20 = close.rolling(20, min_periods=1).mean().iloc[-1]
        sma_50 = close.rolling(50, min_periods=1).mean().iloc[-1]
        trend_dir = 1 if sma_20 > sma_50 else -1
        features.append(trend_dir)

        # 2. Trend Strength (ADX)
        adx = self._calculate_adx(df, period=14)
        features.append(adx / 100)  # Normalize 0-1

        # 3. Momentum (RSI)
        rsi = self._calculate_rsi(close, period=14)
        features.append(rsi / 100)  # Normalize 0-1

        # 4. Volatility Regime (ATR percentile)
        atr = self._calculate_atr(df, period=14)
        atr_history = (df['high'].rolling(14, min_periods=1).max() -
                      df['low'].rolling(14, min_periods=1).min())
        atr_percentile = (atr_history < atr).mean()
        features.append(atr_percentile)

        # 5. Bollinger Band Position
        bb_middle = close.rolling(20, min_periods=1).mean().iloc[-1]
        bb_std = close.rolling(20, min_periods=1).std().iloc[-1]
        bb_upper = bb_middle + 2 * bb_std
        bb_lower = bb_middle - 2 * bb_std
        bb_position = (close.iloc[-1] - bb_lower) / (bb_upper - bb_lower + 1e-8)
        features.append(np.clip(bb_position, 0, 1))

        # 6. Price vs MAs
        dist_from_sma20 = (close.iloc[-1] - sma_20) / (sma_20 + 1e-8)
        features.append(np.clip(dist_from_sma20, -0.1, 0.1) * 10)  # Scale to ~-1 to 1

        return features

    def _calculate_cross_tf_features(
        self,
        mtf_data: Dict[str, pd.DataFrame]
    ) -> List[float]:
        """Calculate features that compare across timeframes."""
        features = []

        # Trend alignment score
        # +1 for each TF with bullish trend, -1 for bearish
        trend_alignment = 0
        for tf_name, df in mtf_data.items():
            sma_20 = df['close'].rolling(20, min_periods=1).mean().iloc[-1]
            sma_50 = df['close'].rolling(50, min_periods=1).mean().iloc[-1]
            if sma_20 > sma_50:
                trend_alignment += 1
            else:
                trend_alignment -= 1

        # Normalize by number of timeframes
        trend_alignment = trend_alignment / len(mtf_data)
        features.append(trend_alignment)

        # Volatility comparison (higher TF vs lower TF)
        tf_list = list(mtf_data.keys())
        if len(tf_list) >= 2:
            # Compare first and last timeframe volatility
            first_tf = tf_list[0]
            last_tf = tf_list[-1]

            first_vol = mtf_data[first_tf]['close'].pct_change().std()
            last_vol = mtf_data[last_tf]['close'].pct_change().std()
            vol_ratio = first_vol / (last_vol + 1e-8)
            features.append(np.clip(vol_ratio, 0, 5) / 5)
        else:
            features.append(0.5)

        return features

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average Directional Index."""
        high = df['high']
        low = df['low']
        close = df['close']

        # True Range
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)

        # Directional Movement
        dm_plus = (high - high.shift(1)).clip(lower=0)
        dm_minus = (low.shift(1) - low).clip(lower=0)

        # Smoothed values
        atr = tr.rolling(period, min_periods=1).mean()
        dm_plus_smooth = dm_plus.rolling(period, min_periods=1).mean()
        dm_minus_smooth = dm_minus.rolling(period, min_periods=1).mean()

        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / (atr + 1e-8)
        di_minus = 100 * dm_minus_smooth / (atr + 1e-8)

        # ADX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus + 1e-8)
        adx = dx.rolling(period, min_periods=1).mean()

        return adx.iloc[-1] if not np.isnan(adx.iloc[-1]) else 25.0

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period, min_periods=1).mean()

        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))

        return rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50.0

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close']

        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)

        atr = tr.rolling(period, min_periods=1).mean()

        return atr.iloc[-1] if not np.isnan(atr.iloc[-1]) else 0.0
