"""
Position Sizing Module
=======================
Calculates position sizes based on risk management rules.

Usage:
    from strategy import PositionSizer
    from config import get_config

    config = get_config()
    sizer = PositionSizer(config)

    # Calculate lot size
    lot_size = sizer.calculate_lot_size(
        balance=10000,
        sl_distance=15.0,  # In price units
        signal_strength=0.8
    )
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import numpy as np

from config.config_loader import Config

logger = logging.getLogger(__name__)


@dataclass
class PositionSize:
    """Represents a calculated position size."""
    lot_size: float
    risk_amount: float  # Dollar amount at risk
    sl_distance: float  # Stop loss distance in price
    sl_pips: float  # Stop loss in pips
    risk_percent: float  # Actual risk percentage
    is_valid: bool
    rejection_reason: Optional[str] = None


class PositionSizer:
    """
    Calculates position sizes based on risk parameters.

    Features:
    - Fixed fractional position sizing
    - ATR-based stop loss calculation
    - Signal strength adjustment
    - Maximum position limits
    """

    def __init__(self, config: Config):
        """
        Initialize position sizer.

        Args:
            config: Configuration object
        """
        self.config = config

        # Risk parameters
        self.base_risk_percent = config.risk.risk_per_trade_percent / 100
        self.max_lot_size = config.risk.max_lot_size
        self.min_lot_size = config.risk.min_lot_size

        # SL parameters
        self.sl_atr_mult = config.risk.sl_atr_multiplier
        self.tp_atr_mult = config.risk.tp_atr_multiplier
        self.max_sl_pips = config.risk.max_sl_pips
        self.min_sl_pips = config.risk.min_sl_pips

        # Gold specifications
        self.pip_value = 0.01  # 1 pip = $0.01 for XAUUSD
        self.contract_size = 100  # 1 lot = 100 oz
        self.pip_value_per_lot = self.contract_size * self.pip_value  # $1 per pip per lot

    def calculate_lot_size(
            self,
            balance: float,
            sl_distance: float,
            signal_strength: float = 1.0,
            max_risk_override: Optional[float] = None
    ) -> PositionSize:
        """
        Calculate lot size based on risk parameters.

        Args:
            balance: Account balance in USD
            sl_distance: Stop loss distance in price units
            signal_strength: Signal strength (0-1) for position scaling
            max_risk_override: Override max risk percent

        Returns:
            PositionSize: Calculated position size
        """
        # Convert SL distance to pips
        sl_pips = sl_distance / self.pip_value

        # Validate SL is within bounds
        if sl_pips < self.min_sl_pips:
            return PositionSize(
                lot_size=0,
                risk_amount=0,
                sl_distance=sl_distance,
                sl_pips=sl_pips,
                risk_percent=0,
                is_valid=False,
                rejection_reason=f"SL too tight: {sl_pips:.1f} < {self.min_sl_pips} pips"
            )

        if sl_pips > self.max_sl_pips:
            return PositionSize(
                lot_size=0,
                risk_amount=0,
                sl_distance=sl_distance,
                sl_pips=sl_pips,
                risk_percent=0,
                is_valid=False,
                rejection_reason=f"SL too wide: {sl_pips:.1f} > {self.max_sl_pips} pips"
            )

        # Calculate risk amount
        risk_percent = max_risk_override if max_risk_override else self.base_risk_percent

        # Adjust risk based on signal strength
        adjusted_risk = risk_percent * signal_strength
        risk_amount = balance * adjusted_risk

        # Calculate lot size
        # lot_size = risk_amount / (sl_pips * pip_value_per_lot)
        lot_size = risk_amount / (sl_pips * self.pip_value_per_lot)

        # Apply constraints
        lot_size = max(self.min_lot_size, lot_size)
        lot_size = min(self.max_lot_size, lot_size)

        # Round to nearest 0.01
        lot_size = round(lot_size, 2)

        # Recalculate actual risk
        actual_risk_amount = lot_size * sl_pips * self.pip_value_per_lot
        actual_risk_percent = actual_risk_amount / balance

        return PositionSize(
            lot_size=lot_size,
            risk_amount=actual_risk_amount,
            sl_distance=sl_distance,
            sl_pips=sl_pips,
            risk_percent=actual_risk_percent,
            is_valid=True
        )

    def calculate_sl_tp(
            self,
            entry_price: float,
            direction: int,
            atr: float
    ) -> Tuple[float, float]:
        """
        Calculate Stop Loss and Take Profit levels.

        Args:
            entry_price: Entry price
            direction: 1 for long, -1 for short
            atr: Current ATR value

        Returns:
            Tuple[sl_price, tp_price]
        """
        # SL distance based on ATR
        sl_distance = atr * self.sl_atr_mult

        # Clamp SL distance
        min_sl_distance = self.min_sl_pips * self.pip_value
        max_sl_distance = self.max_sl_pips * self.pip_value
        sl_distance = np.clip(sl_distance, min_sl_distance, max_sl_distance)

        # TP distance (using configured ratio)
        tp_distance = sl_distance * (self.tp_atr_mult / self.sl_atr_mult)

        # Calculate prices
        if direction == 1:  # Long
            sl_price = entry_price - sl_distance
            tp_price = entry_price + tp_distance
        else:  # Short
            sl_price = entry_price + sl_distance
            tp_price = entry_price - tp_distance

        return sl_price, tp_price

    def calculate_risk_reward(
            self,
            entry_price: float,
            sl_price: float,
            tp_price: float,
            direction: int
    ) -> float:
        """
        Calculate risk/reward ratio.

        Args:
            entry_price: Entry price
            sl_price: Stop loss price
            tp_price: Take profit price
            direction: 1 for long, -1 for short

        Returns:
            float: Risk/reward ratio (e.g., 1.5 means 1.5:1 R:R)
        """
        if direction == 1:
            risk = entry_price - sl_price
            reward = tp_price - entry_price
        else:
            risk = sl_price - entry_price
            reward = entry_price - tp_price

        if risk <= 0:
            return 0

        return reward / risk

    def get_max_position_for_balance(
            self,
            balance: float,
            current_open_lots: float = 0
    ) -> float:
        """
        Get maximum allowed additional position size.

        Args:
            balance: Current balance
            current_open_lots: Currently open lot size

        Returns:
            float: Maximum additional lots allowed
        """
        # Maximum total exposure based on balance
        max_total_lots = self.max_lot_size

        # Available room
        available = max_total_lots - current_open_lots

        return max(0, available)


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from config import get_config

    config = get_config()
    sizer = PositionSizer(config)

    # Test cases
    print("Position Sizing Test Cases:")
    print("=" * 60)

    test_cases = [
        {"balance": 10000, "sl_distance": 5.0, "strength": 1.0},
        {"balance": 10000, "sl_distance": 10.0, "strength": 1.0},
        {"balance": 10000, "sl_distance": 5.0, "strength": 0.5},
        {"balance": 5000, "sl_distance": 5.0, "strength": 1.0},
        {"balance": 10000, "sl_distance": 0.3, "strength": 1.0},  # SL too tight
        {"balance": 10000, "sl_distance": 10.0, "strength": 1.0},  # Normal
    ]

    for i, tc in enumerate(test_cases):
        pos = sizer.calculate_lot_size(
            balance=tc['balance'],
            sl_distance=tc['sl_distance'],
            signal_strength=tc['strength']
        )
        print(f"\nCase {i + 1}: Balance=${tc['balance']}, SL={tc['sl_distance']}, Strength={tc['strength']}")
        print(f"  Lot Size: {pos.lot_size}")
        print(f"  Risk: ${pos.risk_amount:.2f} ({pos.risk_percent:.2%})")
        print(f"  SL Pips: {pos.sl_pips:.1f}")
        print(f"  Valid: {pos.is_valid}")
        if not pos.is_valid:
            print(f"  Reason: {pos.rejection_reason}")

    # Test SL/TP calculation
    print("\n" + "=" * 60)
    print("SL/TP Calculation:")
    entry = 2000.0
    atr = 5.0

    sl, tp = sizer.calculate_sl_tp(entry, 1, atr)  # Long
    rr = sizer.calculate_risk_reward(entry, sl, tp, 1)
    print(f"\nLong @ {entry}: SL={sl:.2f}, TP={tp:.2f}, R:R={rr:.2f}")

    sl, tp = sizer.calculate_sl_tp(entry, -1, atr)  # Short
    rr = sizer.calculate_risk_reward(entry, sl, tp, -1)
    print(f"Short @ {entry}: SL={sl:.2f}, TP={tp:.2f}, R:R={rr:.2f}")
