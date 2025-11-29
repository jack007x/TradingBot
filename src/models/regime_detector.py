"""
Market Regime Detector
Detects current market regime to adjust strategy.

Regimes:
- TRENDING_UP: Strong bullish trend (follow trend)
- TRENDING_DOWN: Strong bearish trend (follow trend)
- RANGING: Sideways (mean reversion)
- HIGH_VOLATILITY: Extreme moves (reduce size, wider stops)
- LOW_VOLATILITY: Quiet market (tighter stops)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from loguru import logger


class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_vol"
    LOW_VOLATILITY = "low_vol"
    UNKNOWN = "unknown"


@dataclass
class RegimeInfo:
    """Information about detected regime."""
    regime: MarketRegime
    confidence: float
    adx: float
    atr_percentile: float
    bb_width: float
    trend_strength: float
    recommendations: Dict


class MarketRegimeDetector:
    """
    Detects market regime using multiple indicators.

    Detection uses:
    - ADX for trend strength
    - ATR percentile for volatility regime
    - Bollinger Band width for ranging markets
    - MA crossovers for trend direction
    """

    def __init__(
        self,
        adx_threshold: float = 25.0,
        vol_high_percentile: float = 0.8,
        vol_low_percentile: float = 0.2,
        bb_width_threshold: float = 0.02
    ):
        """
        Initialize regime detector.

        Args:
            adx_threshold: ADX above this = trending
            vol_high_percentile: ATR percentile above this = high vol
            vol_low_percentile: ATR percentile below this = low vol
            bb_width_threshold: BB width below this = ranging
        """
        self.adx_threshold = adx_threshold
        self.vol_high_pct = vol_high_percentile
        self.vol_low_pct = vol_low_percentile
        self.bb_width_threshold = bb_width_threshold

        # Store regime history
        self.regime_history: list = []

        logger.info("MarketRegimeDetector initialized")
        logger.info(f"  ADX threshold: {adx_threshold}")
        logger.info(f"  Volatility thresholds: {vol_low_percentile}-{vol_high_percentile}")
        logger.info(f"  BB width threshold: {bb_width_threshold}")

    def detect_regime(self, df: pd.DataFrame) -> RegimeInfo:
        """
        Detect current market regime.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            RegimeInfo with detected regime and metadata
        """
        # Calculate indicators
        adx = self._calculate_adx(df)
        atr_pct = self._calculate_atr_percentile(df)
        bb_width = self._calculate_bb_width(df)
        trend_strength = self._calculate_trend_strength(df)

        # Detect regime with priority logic
        regime = MarketRegime.UNKNOWN
        confidence = 0.5

        # Priority 1: High/Low Volatility (most important for risk management)
        if atr_pct > self.vol_high_pct:
            regime = MarketRegime.HIGH_VOLATILITY
            confidence = atr_pct

        elif atr_pct < self.vol_low_pct:
            regime = MarketRegime.LOW_VOLATILITY
            confidence = 1 - atr_pct

        # Priority 2: Trending vs Ranging (for strategy selection)
        elif adx > self.adx_threshold:
            # Strong trend
            if trend_strength > 0:
                regime = MarketRegime.TRENDING_UP
            else:
                regime = MarketRegime.TRENDING_DOWN
            confidence = min(adx / 50, 1.0)  # Higher ADX = higher confidence

        elif bb_width < self.bb_width_threshold:
            regime = MarketRegime.RANGING
            confidence = 1 - (bb_width / self.bb_width_threshold)

        # Get trading recommendations for this regime
        recommendations = self._get_recommendations(regime)

        # Store in history
        info = RegimeInfo(
            regime=regime,
            confidence=confidence,
            adx=adx,
            atr_percentile=atr_pct,
            bb_width=bb_width,
            trend_strength=trend_strength,
            recommendations=recommendations
        )

        self.regime_history.append(info)
        if len(self.regime_history) > 100:
            self.regime_history.pop(0)

        logger.info(f"📊 Regime: {regime.value.upper()} (confidence: {confidence:.2%})")
        logger.info(f"   ADX: {adx:.1f} | ATR%: {atr_pct:.2f} | BB Width: {bb_width:.4f}")

        return info

    def _get_recommendations(self, regime: MarketRegime) -> Dict:
        """
        Get trading recommendations for regime.
        """
        recommendations = {
            MarketRegime.TRENDING_UP: {
                'bias': 'long',
                'entry_strategy': 'pullback_to_ma',
                'position_size_mult': 1.2,
                'tp_mult': 1.5,  # Wider TP in trends
                'sl_mult': 1.0,
                'hold_winners': True,
                'cut_losers_fast': True,
                'confidence_threshold_adj': 1.0  # No adjustment
            },
            MarketRegime.TRENDING_DOWN: {
                'bias': 'short',
                'entry_strategy': 'pullback_to_ma',
                'position_size_mult': 1.2,
                'tp_mult': 1.5,
                'sl_mult': 1.0,
                'hold_winners': True,
                'cut_losers_fast': True,
                'confidence_threshold_adj': 1.0
            },
            MarketRegime.RANGING: {
                'bias': 'neutral',
                'entry_strategy': 'mean_reversion',
                'position_size_mult': 0.7,  # Smaller in ranges
                'tp_mult': 0.7,  # Tighter TP
                'sl_mult': 0.8,
                'hold_winners': False,  # Take profits quickly
                'cut_losers_fast': True,
                'confidence_threshold_adj': 1.1  # Need higher confidence
            },
            MarketRegime.HIGH_VOLATILITY: {
                'bias': 'neutral',
                'entry_strategy': 'wait_for_setup',
                'position_size_mult': 0.5,  # Much smaller
                'tp_mult': 2.0,  # Wide TP for big moves
                'sl_mult': 1.5,  # Wider SL
                'hold_winners': True,
                'cut_losers_fast': False,  # Give room
                'confidence_threshold_adj': 1.3  # Much higher confidence needed
            },
            MarketRegime.LOW_VOLATILITY: {
                'bias': 'neutral',
                'entry_strategy': 'breakout',
                'position_size_mult': 1.0,
                'tp_mult': 0.8,
                'sl_mult': 0.8,
                'hold_winners': False,
                'cut_losers_fast': True,
                'confidence_threshold_adj': 1.0
            },
            MarketRegime.UNKNOWN: {
                'bias': 'neutral',
                'entry_strategy': 'conservative',
                'position_size_mult': 0.5,
                'tp_mult': 1.0,
                'sl_mult': 1.0,
                'hold_winners': False,
                'cut_losers_fast': True,
                'confidence_threshold_adj': 1.2
            }
        }

        return recommendations.get(regime, recommendations[MarketRegime.UNKNOWN])

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ADX."""
        high = df['high']
        low = df['low']
        close = df['close']

        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)

        dm_plus = (high - high.shift(1)).clip(lower=0)
        dm_minus = (low.shift(1) - low).clip(lower=0)

        atr = tr.rolling(period, min_periods=1).mean()
        dm_plus_smooth = dm_plus.rolling(period, min_periods=1).mean()
        dm_minus_smooth = dm_minus.rolling(period, min_periods=1).mean()

        di_plus = 100 * dm_plus_smooth / (atr + 1e-8)
        di_minus = 100 * dm_minus_smooth / (atr + 1e-8)

        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus + 1e-8)
        adx = dx.rolling(period, min_periods=1).mean()

        return adx.iloc[-1] if not np.isnan(adx.iloc[-1]) else 25.0

    def _calculate_atr_percentile(self, df: pd.DataFrame, period: int = 14, lookback: int = 100) -> float:
        """Calculate ATR as percentile of historical ATR."""
        high = df['high']
        low = df['low']
        close = df['close']

        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)

        atr = tr.rolling(period, min_periods=1).mean()
        current_atr = atr.iloc[-1]

        # Calculate percentile over lookback period
        atr_history = atr.iloc[-lookback:]
        percentile = (atr_history < current_atr).mean()

        return percentile if not np.isnan(percentile) else 0.5

    def _calculate_bb_width(self, df: pd.DataFrame, period: int = 20) -> float:
        """Calculate Bollinger Band width."""
        close = df['close']

        middle = close.rolling(period, min_periods=1).mean()
        std = close.rolling(period, min_periods=1).std()

        bb_width = (2 * std) / (middle + 1e-8)

        return bb_width.iloc[-1] if not np.isnan(bb_width.iloc[-1]) else 0.03

    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """
        Calculate trend strength.
        Positive = uptrend, Negative = downtrend.
        """
        close = df['close']

        sma_20 = close.rolling(20, min_periods=1).mean().iloc[-1]
        sma_50 = close.rolling(50, min_periods=1).mean().iloc[-1]

        # Trend strength based on MA separation
        trend_strength = (sma_20 - sma_50) / (sma_50 + 1e-8)

        return trend_strength if not np.isnan(trend_strength) else 0.0

    def get_regime_stability(self) -> float:
        """
        Calculate how stable the regime has been.
        High stability = same regime for many periods.
        """
        if len(self.regime_history) < 5:
            return 0.5

        recent = self.regime_history[-10:]
        regimes = [r.regime for r in recent]

        # Count most common regime
        from collections import Counter
        counter = Counter(regimes)
        most_common_count = counter.most_common(1)[0][1]

        stability = most_common_count / len(recent)

        return stability

    def get_regime_summary(self) -> Dict:
        """Get summary statistics of regime history."""
        if not self.regime_history:
            return {}

        recent = self.regime_history[-20:]

        from collections import Counter
        regime_counts = Counter([r.regime.value for r in recent])

        return {
            'current_regime': self.regime_history[-1].regime.value,
            'current_confidence': self.regime_history[-1].confidence,
            'stability': self.get_regime_stability(),
            'regime_distribution': dict(regime_counts),
            'avg_adx': np.mean([r.adx for r in recent]),
            'avg_atr_percentile': np.mean([r.atr_percentile for r in recent])
        }
