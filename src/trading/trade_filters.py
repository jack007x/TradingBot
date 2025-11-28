"""
Trade Filtering System
Comprehensive filters to ensure only high-quality trades are executed.
"""

from datetime import datetime
from typing import Tuple, Dict, Optional
from loguru import logger
import numpy as np


class TradeFilter:
    """
    Multi-layer trade filtering system.

    Filters trades based on:
    - Confidence threshold
    - Spread limits
    - Trading session (for specific instruments)
    - Volatility conditions
    - Market hours
    """

    def __init__(
        self,
        min_confidence: float = 0.5,
        max_spread_pips: Optional[Dict[str, float]] = None
    ):
        """
        Initialize trade filter.

        Args:
            min_confidence: Minimum confidence threshold (0.0 - 1.0)
            max_spread_pips: Max allowed spread per symbol (pips)
        """
        self.min_confidence = min_confidence

        # Default max spreads (in pips) for common symbols
        self.max_spread_pips = max_spread_pips or {
            'XAUUSD': 35.0,   # Gold - wider spread acceptable
            'EURUSD': 2.0,    # Major pair - tight spread
            'GBPUSD': 3.0,    # Major pair
            'USDJPY': 2.0,    # Major pair
            'BTCUSD': 50.0,   # Crypto - volatile
            'default': 20.0   # Fallback
        }

        logger.info(f"TradeFilter initialized")
        logger.info(f"   Min confidence: {self.min_confidence:.2f}")
        logger.info(f"   Max spreads: {self.max_spread_pips}")

    def should_trade(
        self,
        symbol: str,
        signal: Dict,
        spread: float = 0.0,
        current_price: float = 0.0
    ) -> Tuple[bool, str]:
        """
        Comprehensive trade filtering.

        Args:
            symbol: Trading symbol
            signal: Signal dict with confidence, direction, etc.
            spread: Current spread in pips
            current_price: Current market price

        Returns:
            Tuple of (should_trade: bool, reason: str)
        """
        # Filter 1: Confidence check
        confidence = signal.get('confidence', 0.0)
        if confidence < self.min_confidence:
            return False, f"Low confidence: {confidence:.2f} < {self.min_confidence:.2f}"

        # Filter 2: Spread check
        max_spread = self.max_spread_pips.get(symbol, self.max_spread_pips['default'])
        if spread > max_spread:
            return False, f"High spread: {spread:.1f} pips > {max_spread:.1f} pips"

        # Filter 3: Session check for Gold
        if 'XAU' in symbol:
            hour_utc = datetime.utcnow().hour
            # Avoid Asian session (22:00 - 07:00 UTC) - low liquidity
            if hour_utc < 7 or hour_utc > 21:
                return False, "Asian session - low Gold liquidity"

        # Filter 4: Signal direction check
        if signal.get('signal') == 'hold':
            return False, "HOLD signal - no trade"

        # Filter 5: Prediction magnitude check (avoid tiny movements)
        prediction = signal.get('prediction', 0.0)
        if abs(prediction) < 0.0005:  # Less than 0.05% expected return
            return False, f"Prediction too small: {prediction:.4f}"

        # All filters passed
        logger.info(f"✅ Trade filters passed for {symbol}")
        logger.info(f"   Confidence: {confidence:.2f} ✓")
        logger.info(f"   Spread: {spread:.1f} pips ✓")
        logger.info(f"   Prediction: {prediction:.4f} ✓")

        return True, "All filters passed"

    def check_spread(self, symbol: str, spread: float) -> Tuple[bool, str]:
        """
        Check if spread is acceptable for this symbol.

        Args:
            symbol: Trading symbol
            spread: Current spread in pips

        Returns:
            Tuple of (is_acceptable: bool, reason: str)
        """
        max_spread = self.max_spread_pips.get(symbol, self.max_spread_pips['default'])

        if spread > max_spread:
            return False, f"Spread {spread:.1f} > max {max_spread:.1f} pips"

        return True, f"Spread OK: {spread:.1f} pips"

    def check_session(self, symbol: str) -> Tuple[bool, str]:
        """
        Check if current session is good for trading this symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Tuple of (is_good_session: bool, reason: str)
        """
        hour_utc = datetime.utcnow().hour

        # Gold - avoid Asian session
        if 'XAU' in symbol:
            if hour_utc < 7:
                return False, f"Asian session (UTC {hour_utc}:00) - low Gold liquidity"
            elif hour_utc > 21:
                return False, f"Late session (UTC {hour_utc}:00) - low Gold liquidity"
            elif 7 <= hour_utc < 16:
                return True, f"London session (UTC {hour_utc}:00) - good for Gold"
            elif 13 <= hour_utc <= 21:
                return True, f"NY session (UTC {hour_utc}:00) - good for Gold"
            else:
                return True, f"Active session (UTC {hour_utc}:00)"

        # EUR pairs - best during London session
        elif 'EUR' in symbol:
            if 7 <= hour_utc < 16:
                return True, f"London session (UTC {hour_utc}:00) - best for EUR"
            else:
                return True, f"Active session (UTC {hour_utc}:00)"

        # USD pairs - best during NY session
        elif 'USD' in symbol:
            if 13 <= hour_utc <= 21:
                return True, f"NY session (UTC {hour_utc}:00) - best for USD"
            else:
                return True, f"Active session (UTC {hour_utc}:00)"

        # Default - all sessions OK
        return True, f"Session OK (UTC {hour_utc}:00)"

    def check_volatility(
        self,
        symbol: str,
        current_atr: float,
        avg_atr: float,
        max_atr_multiplier: float = 3.0
    ) -> Tuple[bool, str]:
        """
        Check if volatility is within acceptable range.

        Args:
            symbol: Trading symbol
            current_atr: Current ATR value
            avg_atr: Average ATR value
            max_atr_multiplier: Max acceptable ATR vs average

        Returns:
            Tuple of (is_acceptable: bool, reason: str)
        """
        if avg_atr == 0:
            return True, "No ATR data - skip volatility check"

        atr_ratio = current_atr / avg_atr

        if atr_ratio > max_atr_multiplier:
            return False, f"Excessive volatility: ATR ratio {atr_ratio:.2f}x > {max_atr_multiplier}x"

        if atr_ratio < 0.3:
            return False, f"Very low volatility: ATR ratio {atr_ratio:.2f}x < 0.3x"

        return True, f"Volatility OK: ATR ratio {atr_ratio:.2f}x"

    def update_min_confidence(self, new_confidence: float):
        """Update minimum confidence threshold."""
        old_confidence = self.min_confidence
        self.min_confidence = new_confidence
        logger.info(f"Min confidence updated: {old_confidence:.2f} → {new_confidence:.2f}")

    def update_max_spread(self, symbol: str, max_spread: float):
        """Update max spread for specific symbol."""
        old_spread = self.max_spread_pips.get(symbol, self.max_spread_pips['default'])
        self.max_spread_pips[symbol] = max_spread
        logger.info(f"Max spread for {symbol} updated: {old_spread:.1f} → {max_spread:.1f} pips")


class VolumeFilter:
    """Filter trades based on volume conditions."""

    def __init__(self, min_volume_ratio: float = 0.5):
        """
        Initialize volume filter.

        Args:
            min_volume_ratio: Minimum volume vs average (0.5 = 50% of avg)
        """
        self.min_volume_ratio = min_volume_ratio
        logger.info(f"VolumeFilter initialized (min ratio: {min_volume_ratio})")

    def check_volume(
        self,
        current_volume: float,
        avg_volume: float
    ) -> Tuple[bool, str]:
        """
        Check if current volume is sufficient.

        Args:
            current_volume: Current bar volume
            avg_volume: Average volume

        Returns:
            Tuple of (is_sufficient: bool, reason: str)
        """
        if avg_volume == 0:
            return True, "No volume data - skip check"

        volume_ratio = current_volume / avg_volume

        if volume_ratio < self.min_volume_ratio:
            return False, f"Low volume: {volume_ratio:.2f}x < {self.min_volume_ratio}x avg"

        return True, f"Volume OK: {volume_ratio:.2f}x avg"
