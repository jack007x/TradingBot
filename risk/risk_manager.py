"""
Risk Manager Module
====================
Enforces risk management rules and circuit breakers.

Usage:
    from risk import RiskManager
    from config import get_config

    config = get_config()
    risk_mgr = RiskManager(config)

    # Check if trading is allowed
    can_trade, reason = risk_mgr.can_open_trade(
        account_state=account_state,
        proposed_trade=trade_params
    )
"""

import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
import numpy as np

from config.config_loader import Config

logger = logging.getLogger(__name__)


@dataclass
class AccountState:
    """Current account state."""
    balance: float
    equity: float
    margin_used: float
    open_positions: int
    open_lots: float
    daily_pnl: float
    weekly_pnl: float
    consecutive_losses: int
    last_trade_time: Optional[datetime] = None
    starting_balance_today: float = 0
    starting_balance_week: float = 0


@dataclass
class TradeProposal:
    """Proposed trade parameters."""
    direction: int  # 1 = BUY, -1 = SELL
    lot_size: float
    sl_distance: float
    entry_price: float
    atr: float


@dataclass
class RiskCheckResult:
    """Result of risk check."""
    approved: bool
    reason: str
    checks_passed: Dict[str, bool] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class RiskManager:
    """
    Central risk management system.

    Implements circuit breakers:
    - Maximum daily loss
    - Maximum weekly loss
    - Maximum drawdown
    - Maximum open positions
    - Maximum consecutive losses
    - Blocked trading hours
    """

    def __init__(self, config: Config):
        """
        Initialize risk manager.

        Args:
            config: Configuration object
        """
        self.config = config

        # Limits
        self.max_daily_loss_pct = config.risk.max_daily_loss_percent / 100
        self.max_weekly_loss_pct = config.risk.max_weekly_loss_percent / 100
        self.max_drawdown_pct = config.risk.max_drawdown_percent / 100
        self.max_open_trades = config.risk.max_open_trades
        self.max_lot_size = config.risk.max_lot_size
        self.max_consecutive_losses = config.risk.max_consecutive_losses
        self.cooldown_minutes = config.risk.cooldown_after_losses_minutes
        self.blocked_hours = config.trading.blocked_hours_utc

        # State tracking
        self._trading_paused = False
        self._pause_reason = ""
        self._pause_until: Optional[datetime] = None
        self._peak_equity = 0
        self._trade_history: List[Dict] = []

    def can_open_trade(
            self,
            account_state: AccountState,
            proposed_trade: Optional[TradeProposal] = None
    ) -> RiskCheckResult:
        """
        Check if a new trade can be opened.

        Args:
            account_state: Current account state
            proposed_trade: Proposed trade parameters

        Returns:
            RiskCheckResult: Approval status with details
        """
        checks = {}
        warnings = []

        # Check 1: Trading pause
        if self._trading_paused:
            if self._pause_until and datetime.now() >= self._pause_until:
                self._trading_paused = False
                self._pause_reason = ""
                logger.info("Trading pause lifted")
            else:
                return RiskCheckResult(
                    approved=False,
                    reason=f"Trading paused: {self._pause_reason}",
                    checks_passed=checks
                )

        # Check 2: Trading hours
        current_hour = datetime.utcnow().hour
        hour_ok = current_hour not in self.blocked_hours
        checks['trading_hours'] = hour_ok
        if not hour_ok:
            return RiskCheckResult(
                approved=False,
                reason=f"Blocked trading hour: {current_hour} UTC",
                checks_passed=checks
            )

        # Check 3: Maximum open positions
        positions_ok = account_state.open_positions < self.max_open_trades
        checks['max_positions'] = positions_ok
        if not positions_ok:
            return RiskCheckResult(
                approved=False,
                reason=f"Max positions reached: {account_state.open_positions}/{self.max_open_trades}",
                checks_passed=checks
            )

        # Check 4: Daily loss limit
        daily_loss_pct = abs(min(0, account_state.daily_pnl)) / account_state.starting_balance_today \
            if account_state.starting_balance_today > 0 else 0
        daily_ok = daily_loss_pct < self.max_daily_loss_pct
        checks['daily_loss'] = daily_ok
        if not daily_ok:
            self._pause_trading(f"Daily loss limit hit: {daily_loss_pct:.2%}")
            return RiskCheckResult(
                approved=False,
                reason=f"Daily loss limit exceeded: {daily_loss_pct:.2%}",
                checks_passed=checks
            )

        # Check 5: Weekly loss limit
        weekly_loss_pct = abs(min(0, account_state.weekly_pnl)) / account_state.starting_balance_week \
            if account_state.starting_balance_week > 0 else 0
        weekly_ok = weekly_loss_pct < self.max_weekly_loss_pct
        checks['weekly_loss'] = weekly_ok
        if not weekly_ok:
            self._pause_trading(f"Weekly loss limit hit: {weekly_loss_pct:.2%}")
            return RiskCheckResult(
                approved=False,
                reason=f"Weekly loss limit exceeded: {weekly_loss_pct:.2%}",
                checks_passed=checks
            )

        # Check 6: Maximum drawdown
        if self._peak_equity == 0:
            self._peak_equity = account_state.equity
        else:
            self._peak_equity = max(self._peak_equity, account_state.equity)

        current_dd = (self._peak_equity - account_state.equity) / self._peak_equity
        dd_ok = current_dd < self.max_drawdown_pct
        checks['max_drawdown'] = dd_ok
        if not dd_ok:
            self._pause_trading(f"Max drawdown hit: {current_dd:.2%}")
            return RiskCheckResult(
                approved=False,
                reason=f"Max drawdown exceeded: {current_dd:.2%}",
                checks_passed=checks
            )

        # Check 7: Consecutive losses
        consec_ok = account_state.consecutive_losses < self.max_consecutive_losses
        checks['consecutive_losses'] = consec_ok
        if not consec_ok:
            self._pause_trading(
                f"Consecutive losses: {account_state.consecutive_losses}",
                cooldown_minutes=self.cooldown_minutes
            )
            return RiskCheckResult(
                approved=False,
                reason=f"Too many consecutive losses: {account_state.consecutive_losses}",
                checks_passed=checks
            )

        # Validate proposed trade if provided
        if proposed_trade is not None:
            # Check lot size
            if proposed_trade.lot_size > self.max_lot_size:
                checks['lot_size'] = False
                return RiskCheckResult(
                    approved=False,
                    reason=f"Lot size too large: {proposed_trade.lot_size} > {self.max_lot_size}",
                    checks_passed=checks
                )
            checks['lot_size'] = True

            # Check total exposure
            total_lots = account_state.open_lots + proposed_trade.lot_size
            exposure_ok = total_lots <= self.max_lot_size * self.max_open_trades
            checks['total_exposure'] = exposure_ok
            if not exposure_ok:
                warnings.append(f"High exposure: {total_lots:.2f} lots")

        # Add warnings for approaching limits
        if daily_loss_pct > self.max_daily_loss_pct * 0.7:
            warnings.append(f"Approaching daily loss limit: {daily_loss_pct:.2%}")

        if current_dd > self.max_drawdown_pct * 0.7:
            warnings.append(f"Approaching max drawdown: {current_dd:.2%}")

        return RiskCheckResult(
            approved=True,
            reason="All risk checks passed",
            checks_passed=checks,
            warnings=warnings
        )

    def _pause_trading(
            self,
            reason: str,
            cooldown_minutes: Optional[int] = None
    ):
        """Pause trading with optional cooldown."""
        self._trading_paused = True
        self._pause_reason = reason

        if cooldown_minutes:
            self._pause_until = datetime.now() + timedelta(minutes=cooldown_minutes)
        else:
            self._pause_until = None  # Manual reset required

        logger.warning(f"Trading PAUSED: {reason}")

    def resume_trading(self) -> bool:
        """
        Manually resume trading.

        Returns:
            bool: True if trading resumed
        """
        if self._trading_paused:
            self._trading_paused = False
            self._pause_reason = ""
            self._pause_until = None
            logger.info("Trading manually resumed")
            return True
        return False

    def is_trading_paused(self) -> Tuple[bool, str]:
        """
        Check if trading is paused.

        Returns:
            Tuple[is_paused, reason]
        """
        return self._trading_paused, self._pause_reason

    def record_trade(self, trade_result: Dict[str, Any]) -> None:
        """
        Record a completed trade for tracking.

        Args:
            trade_result: Trade result dict with 'pnl', 'exit_time', etc.
        """
        self._trade_history.append({
            **trade_result,
            'recorded_at': datetime.now()
        })

        # Keep only recent trades
        max_history = 500
        if len(self._trade_history) > max_history:
            self._trade_history = self._trade_history[-max_history:]

    def get_daily_stats(self, date: Optional[datetime] = None) -> Dict[str, float]:
        """Get trading statistics for a day."""
        if date is None:
            date = datetime.now().date()

        day_trades = [
            t for t in self._trade_history
            if t.get('exit_time', t.get('recorded_at')).date() == date
        ]

        if not day_trades:
            return {'pnl': 0, 'trades': 0, 'wins': 0, 'losses': 0}

        pnls = [t.get('pnl', 0) for t in day_trades]

        return {
            'pnl': sum(pnls),
            'trades': len(day_trades),
            'wins': sum(1 for p in pnls if p > 0),
            'losses': sum(1 for p in pnls if p < 0),
            'win_rate': sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0,
        }

    def validate_sl_tp(
            self,
            entry_price: float,
            sl_price: float,
            tp_price: float,
            direction: int
    ) -> Tuple[bool, str]:
        """
        Validate stop loss and take profit levels.

        Returns:
            Tuple[is_valid, reason]
        """
        # Check direction consistency
        if direction == 1:  # Long
            if sl_price >= entry_price:
                return False, "SL must be below entry for long"
            if tp_price <= entry_price:
                return False, "TP must be above entry for long"
        else:  # Short
            if sl_price <= entry_price:
                return False, "SL must be above entry for short"
            if tp_price >= entry_price:
                return False, "TP must be below entry for short"

        # Check minimum SL distance
        sl_distance = abs(entry_price - sl_price)
        min_distance = self.config.risk.min_sl_pips * 0.01  # Convert to price
        if sl_distance < min_distance:
            return False, f"SL distance too small: {sl_distance:.2f} < {min_distance:.2f}"

        return True, "Valid"

    def get_status(self) -> Dict[str, Any]:
        """Get current risk manager status."""
        return {
            'trading_paused': self._trading_paused,
            'pause_reason': self._pause_reason,
            'pause_until': self._pause_until.isoformat() if self._pause_until else None,
            'peak_equity': self._peak_equity,
            'trade_history_count': len(self._trade_history),
            'limits': {
                'max_daily_loss_pct': self.max_daily_loss_pct,
                'max_weekly_loss_pct': self.max_weekly_loss_pct,
                'max_drawdown_pct': self.max_drawdown_pct,
                'max_open_trades': self.max_open_trades,
                'max_consecutive_losses': self.max_consecutive_losses,
            }
        }


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from config import get_config

    config = get_config()
    risk_mgr = RiskManager(config)

    # Create test account state
    account = AccountState(
        balance=10000,
        equity=10000,
        margin_used=0,
        open_positions=0,
        open_lots=0,
        daily_pnl=0,
        weekly_pnl=0,
        consecutive_losses=0,
        starting_balance_today=10000,
        starting_balance_week=10000
    )

    # Test normal trade
    print("Test 1: Normal conditions")
    result = risk_mgr.can_open_trade(account)
    print(f"  Approved: {result.approved}")
    print(f"  Reason: {result.reason}")

    # Test with losses
    print("\nTest 2: After daily loss")
    account.daily_pnl = -400  # 4% loss
    result = risk_mgr.can_open_trade(account)
    print(f"  Approved: {result.approved}")
    print(f"  Reason: {result.reason}")

    # Reset and test consecutive losses
    account.daily_pnl = 0
    risk_mgr.resume_trading()

    print("\nTest 3: Consecutive losses")
    account.consecutive_losses = 6
    result = risk_mgr.can_open_trade(account)
    print(f"  Approved: {result.approved}")
    print(f"  Reason: {result.reason}")

    # Check status
    print("\nRisk Manager Status:")
    status = risk_mgr.get_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
