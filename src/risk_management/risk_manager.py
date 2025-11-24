"""
Risk Management System for AI Trading Bot.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class Position:
    """Represents an open trading position."""
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    size: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    trailing_stop: Optional[float] = None
    highest_price: Optional[float] = None
    lowest_price: Optional[float] = None


@dataclass
class RiskMetrics:
    """Current risk metrics."""
    total_exposure: float = 0.0
    daily_pnl: float = 0.0
    daily_drawdown: float = 0.0
    max_drawdown: float = 0.0
    var_95: float = 0.0
    current_risk: float = 0.0


class RiskManager:
    """
    Comprehensive risk management system with dynamic position sizing
    and real-time risk monitoring.
    """

    def __init__(
        self,
        initial_balance: float = 10000.0,
        max_risk_per_trade: float = 0.02,
        max_daily_drawdown: float = 0.05,
        max_total_drawdown: float = 0.15,
        max_position_size: float = 0.3,
        max_correlation: float = 0.7,
        max_open_positions: int = 5,
        use_trailing_stop: bool = True,
        trailing_stop_pct: float = 0.02
    ):
        """
        Initialize risk manager.

        Args:
            initial_balance: Starting account balance
            max_risk_per_trade: Maximum risk per trade (fraction)
            max_daily_drawdown: Maximum allowed daily drawdown
            max_total_drawdown: Maximum total drawdown before stopping
            max_position_size: Maximum position size (fraction of balance)
            max_correlation: Maximum correlation between positions
            max_open_positions: Maximum number of open positions
            use_trailing_stop: Whether to use trailing stops
            trailing_stop_pct: Trailing stop percentage
        """
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_drawdown = max_daily_drawdown
        self.max_total_drawdown = max_total_drawdown
        self.max_position_size = max_position_size
        self.max_correlation = max_correlation
        self.max_open_positions = max_open_positions
        self.use_trailing_stop = use_trailing_stop
        self.trailing_stop_pct = trailing_stop_pct

        # State tracking
        self.positions: Dict[str, Position] = {}
        self.daily_pnl = 0.0
        self.daily_start_balance = initial_balance
        self.peak_balance = initial_balance
        self.trade_history: List[Dict] = []

        # Risk metrics
        self.metrics = RiskMetrics()

        logger.info("RiskManager initialized")

    def can_open_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        stop_loss: float,
        proposed_size: float
    ) -> Tuple[bool, str, float]:
        """
        Check if a new position can be opened.

        Args:
            symbol: Trading symbol
            side: 'long' or 'short'
            entry_price: Entry price
            stop_loss: Stop loss price
            proposed_size: Proposed position size

        Returns:
            Tuple of (allowed, reason, adjusted_size)
        """
        # Check max positions
        if len(self.positions) >= self.max_open_positions:
            return False, "Maximum open positions reached", 0.0

        # Check if already in this symbol
        if symbol in self.positions:
            return False, f"Already have position in {symbol}", 0.0

        # Check daily drawdown
        if self._check_daily_drawdown_exceeded():
            return False, "Daily drawdown limit exceeded", 0.0

        # Check total drawdown
        if self._check_total_drawdown_exceeded():
            return False, "Total drawdown limit exceeded", 0.0

        # Calculate risk for proposed position
        risk_amount = self._calculate_position_risk(entry_price, stop_loss, proposed_size)
        max_allowed_risk = self.current_balance * self.max_risk_per_trade

        # Adjust size if risk too high
        if risk_amount > max_allowed_risk:
            adjusted_size = self._calculate_size_for_risk(
                entry_price, stop_loss, max_allowed_risk
            )
            if adjusted_size < proposed_size * 0.1:
                return False, "Position size would be too small", 0.0
            return True, "Size adjusted for risk", adjusted_size

        # Check position size limit
        position_value = entry_price * proposed_size
        max_position_value = self.current_balance * self.max_position_size

        if position_value > max_position_value:
            adjusted_size = max_position_value / entry_price
            return True, "Size adjusted for max position limit", adjusted_size

        return True, "Position approved", proposed_size

    def open_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        size: float,
        stop_loss: float,
        take_profit: float
    ) -> Optional[Position]:
        """
        Open a new position.

        Args:
            symbol: Trading symbol
            side: 'long' or 'short'
            entry_price: Entry price
            size: Position size
            stop_loss: Stop loss price
            take_profit: Take profit price

        Returns:
            Position object if opened, None otherwise
        """
        can_open, reason, adjusted_size = self.can_open_position(
            symbol, side, entry_price, stop_loss, size
        )

        if not can_open:
            logger.warning(f"Cannot open position: {reason}")
            return None

        position = Position(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            size=adjusted_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_time=datetime.utcnow(),
            highest_price=entry_price if side == 'long' else None,
            lowest_price=entry_price if side == 'short' else None
        )

        self.positions[symbol] = position
        self._update_metrics()

        logger.info(f"Opened {side} position: {symbol} @ {entry_price}, size: {adjusted_size:.4f}")
        return position

    def update_position(
        self,
        symbol: str,
        current_price: float
    ) -> Optional[str]:
        """
        Update position with current price and check for exits.

        Args:
            symbol: Trading symbol
            current_price: Current market price

        Returns:
            Exit reason if position should be closed, None otherwise
        """
        if symbol not in self.positions:
            return None

        position = self.positions[symbol]

        # Update tracking prices
        if position.side == 'long':
            if position.highest_price is None or current_price > position.highest_price:
                position.highest_price = current_price
                # Update trailing stop
                if self.use_trailing_stop:
                    new_trailing = current_price * (1 - self.trailing_stop_pct)
                    if position.trailing_stop is None or new_trailing > position.trailing_stop:
                        position.trailing_stop = new_trailing
        else:  # short
            if position.lowest_price is None or current_price < position.lowest_price:
                position.lowest_price = current_price
                if self.use_trailing_stop:
                    new_trailing = current_price * (1 + self.trailing_stop_pct)
                    if position.trailing_stop is None or new_trailing < position.trailing_stop:
                        position.trailing_stop = new_trailing

        # Check exit conditions
        if position.side == 'long':
            if current_price <= position.stop_loss:
                return 'stop_loss'
            if current_price >= position.take_profit:
                return 'take_profit'
            if position.trailing_stop and current_price <= position.trailing_stop:
                return 'trailing_stop'
        else:  # short
            if current_price >= position.stop_loss:
                return 'stop_loss'
            if current_price <= position.take_profit:
                return 'take_profit'
            if position.trailing_stop and current_price >= position.trailing_stop:
                return 'trailing_stop'

        return None

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        reason: str = 'manual'
    ) -> Optional[Dict]:
        """
        Close a position.

        Args:
            symbol: Trading symbol
            exit_price: Exit price
            reason: Reason for closing

        Returns:
            Trade result dictionary
        """
        if symbol not in self.positions:
            return None

        position = self.positions[symbol]

        # Calculate PnL
        if position.side == 'long':
            pnl = (exit_price - position.entry_price) * position.size
            pnl_pct = (exit_price - position.entry_price) / position.entry_price
        else:
            pnl = (position.entry_price - exit_price) * position.size
            pnl_pct = (position.entry_price - exit_price) / position.entry_price

        # Update balance
        self.current_balance += pnl
        self.daily_pnl += pnl

        # Update peak balance
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance

        # Record trade
        trade_record = {
            'symbol': symbol,
            'side': position.side,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'size': position.size,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'entry_time': position.entry_time.isoformat(),
            'exit_time': datetime.utcnow().isoformat(),
            'reason': reason
        }

        self.trade_history.append(trade_record)
        del self.positions[symbol]
        self._update_metrics()

        logger.info(f"Closed {position.side} {symbol}: PnL = {pnl:.2f} ({pnl_pct:.2%}), Reason: {reason}")
        return trade_record

    def _calculate_position_risk(
        self,
        entry_price: float,
        stop_loss: float,
        size: float
    ) -> float:
        """Calculate risk amount for a position."""
        price_risk = abs(entry_price - stop_loss)
        return price_risk * size

    def _calculate_size_for_risk(
        self,
        entry_price: float,
        stop_loss: float,
        max_risk: float
    ) -> float:
        """Calculate position size for given risk amount."""
        price_risk = abs(entry_price - stop_loss)
        if price_risk == 0:
            return 0.0
        return max_risk / price_risk

    def _check_daily_drawdown_exceeded(self) -> bool:
        """Check if daily drawdown limit exceeded."""
        daily_dd = (self.daily_start_balance - self.current_balance) / self.daily_start_balance
        return daily_dd >= self.max_daily_drawdown

    def _check_total_drawdown_exceeded(self) -> bool:
        """Check if total drawdown limit exceeded."""
        total_dd = (self.peak_balance - self.current_balance) / self.peak_balance
        return total_dd >= self.max_total_drawdown

    def _update_metrics(self) -> None:
        """Update risk metrics."""
        # Total exposure
        total_exposure = sum(
            p.entry_price * p.size for p in self.positions.values()
        )
        self.metrics.total_exposure = total_exposure / self.current_balance if self.current_balance > 0 else 0

        # Daily PnL and drawdown
        self.metrics.daily_pnl = self.daily_pnl
        self.metrics.daily_drawdown = (self.daily_start_balance - self.current_balance) / self.daily_start_balance

        # Max drawdown
        self.metrics.max_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance

        # VaR (simplified)
        if len(self.trade_history) >= 20:
            returns = [t['pnl_pct'] for t in self.trade_history[-100:]]
            self.metrics.var_95 = np.percentile(returns, 5)

        # Current risk
        total_risk = sum(
            self._calculate_position_risk(p.entry_price, p.stop_loss, p.size)
            for p in self.positions.values()
        )
        self.metrics.current_risk = total_risk / self.current_balance if self.current_balance > 0 else 0

    def reset_daily(self) -> None:
        """Reset daily tracking (call at start of each trading day)."""
        self.daily_pnl = 0.0
        self.daily_start_balance = self.current_balance
        logger.info("Daily risk metrics reset")

    def get_risk_report(self) -> Dict:
        """Get comprehensive risk report."""
        self._update_metrics()

        return {
            'current_balance': self.current_balance,
            'initial_balance': self.initial_balance,
            'total_return': (self.current_balance - self.initial_balance) / self.initial_balance,
            'open_positions': len(self.positions),
            'total_exposure': self.metrics.total_exposure,
            'daily_pnl': self.metrics.daily_pnl,
            'daily_drawdown': self.metrics.daily_drawdown,
            'max_drawdown': self.metrics.max_drawdown,
            'current_risk': self.metrics.current_risk,
            'var_95': self.metrics.var_95,
            'positions': {
                symbol: {
                    'side': p.side,
                    'entry_price': p.entry_price,
                    'size': p.size,
                    'unrealized_pnl': self._calculate_unrealized_pnl(p, p.entry_price)
                }
                for symbol, p in self.positions.items()
            }
        }

    def _calculate_unrealized_pnl(self, position: Position, current_price: float) -> float:
        """Calculate unrealized PnL for a position."""
        if position.side == 'long':
            return (current_price - position.entry_price) * position.size
        else:
            return (position.entry_price - current_price) * position.size

    def adjust_for_volatility(
        self,
        symbol: str,
        current_volatility: float,
        baseline_volatility: float = 0.02
    ) -> float:
        """
        Get volatility-adjusted risk factor.

        Args:
            symbol: Trading symbol
            current_volatility: Current market volatility
            baseline_volatility: Baseline volatility for comparison

        Returns:
            Adjustment factor (< 1 means reduce risk)
        """
        if current_volatility <= 0:
            return 1.0

        ratio = baseline_volatility / current_volatility
        return min(max(ratio, 0.25), 2.0)  # Clamp between 0.25x and 2x
