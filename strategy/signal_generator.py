"""
Signal Generator Module
========================
Converts model predictions to actionable trading signals.

Usage:
    from strategy import SignalGenerator
    from config import get_config

    config = get_config()
    signal_gen = SignalGenerator(config)

    # Generate signal for current bar
    signal = signal_gen.generate_signal(
        prediction=1,
        probability=0.65,
        market_conditions={'atr': 15.0, 'hour': 10}
    )
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as np

from config.config_loader import Config

logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """Represents a trading signal."""
    direction: int  # 1 = BUY, -1 = SELL, 0 = NO_TRADE
    confidence: float  # Model confidence (0-1)
    strength: float  # Signal strength (0-1)
    reason: str  # Reason for signal/rejection
    filters_passed: Dict[str, bool]  # Which filters passed/failed

    @property
    def is_tradeable(self) -> bool:
        return self.direction != 0

    def __repr__(self):
        dir_str = {1: 'BUY', -1: 'SELL', 0: 'NO_TRADE'}[self.direction]
        return f"Signal({dir_str}, conf={self.confidence:.2f}, str={self.strength:.2f})"


class SignalGenerator:
    """
    Generates trading signals from model predictions.

    Applies filters:
    - Probability threshold
    - Trading hours
    - Volatility conditions
    - Market regime
    """

    def __init__(self, config: Config):
        """
        Initialize signal generator.

        Args:
            config: Configuration object
        """
        self.config = config

        # Thresholds
        self.min_probability = config.model.min_probability_threshold
        self.blocked_hours = config.trading.blocked_hours_utc

        # Volatility thresholds
        self.min_atr_percentile = 0.1  # Don't trade if volatility too low
        self.max_atr_percentile = 0.95  # Don't trade if volatility too high

        # Signal strength weights
        self.prob_weight = 0.7
        self.trend_weight = 0.3

    def generate_signal(
            self,
            prediction: int,
            probability: float,
            market_conditions: Dict[str, Any]
    ) -> TradingSignal:
        """
        Generate trading signal from prediction.

        Args:
            prediction: Model prediction (-1, 0, 1)
            probability: Probability of predicted class
            market_conditions: Dict with keys like 'atr', 'hour', 'ema_trend', etc.

        Returns:
            TradingSignal: Signal object
        """
        filters_passed = {}

        # Filter 1: Neutral prediction
        if prediction == 0:
            return TradingSignal(
                direction=0,
                confidence=probability,
                strength=0.0,
                reason="Neutral prediction",
                filters_passed=filters_passed
            )

        # Filter 2: Probability threshold
        prob_pass = probability >= self.min_probability
        filters_passed['probability'] = prob_pass
        if not prob_pass:
            return TradingSignal(
                direction=0,
                confidence=probability,
                strength=0.0,
                reason=f"Low confidence: {probability:.2f} < {self.min_probability}",
                filters_passed=filters_passed
            )

        # Filter 3: Trading hours
        hour = market_conditions.get('hour')
        if hour is not None:
            hour_pass = hour not in self.blocked_hours
            filters_passed['trading_hours'] = hour_pass
            if not hour_pass:
                return TradingSignal(
                    direction=0,
                    confidence=probability,
                    strength=0.0,
                    reason=f"Blocked trading hour: {hour}",
                    filters_passed=filters_passed
                )

        # Filter 4: Volatility (ATR)
        atr = market_conditions.get('atr')
        atr_percentile = market_conditions.get('atr_percentile')
        if atr_percentile is not None:
            vol_pass = self.min_atr_percentile <= atr_percentile <= self.max_atr_percentile
            filters_passed['volatility'] = vol_pass
            if not vol_pass:
                return TradingSignal(
                    direction=0,
                    confidence=probability,
                    strength=0.0,
                    reason=f"Extreme volatility: percentile={atr_percentile:.2f}",
                    filters_passed=filters_passed
                )

        # Filter 5: Trend alignment (optional)
        ema_trend = market_conditions.get('ema_trend')  # 1 = bullish, -1 = bearish
        if ema_trend is not None:
            trend_aligned = (prediction == 1 and ema_trend == 1) or \
                            (prediction == -1 and ema_trend == -1)
            filters_passed['trend_alignment'] = trend_aligned
            # Don't reject, but reduce strength if against trend

        # Calculate signal strength
        strength = self._calculate_strength(
            probability,
            market_conditions,
            prediction
        )

        # All filters passed
        return TradingSignal(
            direction=prediction,
            confidence=probability,
            strength=strength,
            reason="All filters passed",
            filters_passed=filters_passed
        )

    def _calculate_strength(
            self,
            probability: float,
            conditions: Dict[str, Any],
            direction: int
    ) -> float:
        """
        Calculate signal strength (0-1).

        Higher strength = more favorable conditions.
        """
        strength = 0.0

        # Probability contribution (normalized from threshold to 1)
        prob_range = 1.0 - self.min_probability
        prob_normalized = (probability - self.min_probability) / prob_range
        strength += self.prob_weight * prob_normalized

        # Trend alignment bonus
        ema_trend = conditions.get('ema_trend')
        if ema_trend is not None:
            if ema_trend == direction:
                strength += self.trend_weight * 1.0
            elif ema_trend == 0:
                strength += self.trend_weight * 0.5
            else:
                strength += self.trend_weight * 0.0

        # Session bonus (overlap session is best)
        is_overlap = conditions.get('is_overlap_session', False)
        if is_overlap:
            strength *= 1.1

        return min(1.0, max(0.0, strength))

    def batch_generate_signals(
            self,
            predictions: np.ndarray,
            probabilities: np.ndarray,
            market_df: 'pd.DataFrame'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate signals for multiple bars (vectorized where possible).

        Args:
            predictions: Array of predictions
            probabilities: Array of max probabilities
            market_df: DataFrame with market conditions

        Returns:
            Tuple[signals, strengths]: Arrays of signal directions and strengths
        """
        n = len(predictions)
        signals = np.zeros(n, dtype=int)
        strengths = np.zeros(n)

        for i in range(n):
            conditions = {
                'hour': market_df.iloc[i].get('hour'),
                'atr': market_df.iloc[i].get('atr'),
                'atr_percentile': market_df.iloc[i].get('vol_percentile'),
                'ema_trend': self._get_ema_trend(market_df.iloc[i]),
                'is_overlap_session': market_df.iloc[i].get('is_overlap_session', 0) == 1,
            }

            signal = self.generate_signal(
                prediction=predictions[i],
                probability=probabilities[i],
                market_conditions=conditions
            )

            signals[i] = signal.direction
            strengths[i] = signal.strength

        return signals, strengths

    def _get_ema_trend(self, row) -> Optional[int]:
        """Determine EMA trend from row data."""
        if 'ema_8_21_cross' in row and 'ema_21_50_cross' in row:
            short_trend = row['ema_8_21_cross']
            medium_trend = row['ema_21_50_cross']
            if short_trend == 1 and medium_trend == 1:
                return 1  # Bullish
            elif short_trend == 0 and medium_trend == 0:
                return -1  # Bearish
            else:
                return 0  # Mixed
        return None


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from config import get_config

    config = get_config()
    signal_gen = SignalGenerator(config)

    # Test various scenarios
    test_cases = [
        {"prediction": 1, "probability": 0.65, "conditions": {"hour": 10, "atr_percentile": 0.5}},
        {"prediction": 1, "probability": 0.45, "conditions": {"hour": 10}},  # Low prob
        {"prediction": -1, "probability": 0.70, "conditions": {"hour": 23}},  # Blocked hour
        {"prediction": 0, "probability": 0.90, "conditions": {}},  # Neutral
        {"prediction": 1, "probability": 0.80, "conditions": {"atr_percentile": 0.98}},  # High vol
    ]

    print("Signal Generation Test Cases:")
    print("=" * 60)

    for i, tc in enumerate(test_cases):
        signal = signal_gen.generate_signal(
            tc['prediction'],
            tc['probability'],
            tc['conditions']
        )
        print(f"\nCase {i + 1}:")
        print(f"  Input: pred={tc['prediction']}, prob={tc['probability']:.2f}")
        print(f"  Output: {signal}")
        print(f"  Reason: {signal.reason}")
