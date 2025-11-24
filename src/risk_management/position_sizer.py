"""
Dynamic Position Sizing for AI Trading Bot.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger


class PositionSizer:
    """
    Dynamic position sizing based on various methods including
    Kelly Criterion, fixed fractional, and ATR-based sizing.
    """

    def __init__(
        self,
        method: str = 'dynamic',
        base_risk: float = 0.02,
        max_risk: float = 0.05,
        min_risk: float = 0.005,
        kelly_fraction: float = 0.5
    ):
        """
        Initialize position sizer.

        Args:
            method: Sizing method ('fixed', 'kelly', 'atr', 'dynamic')
            base_risk: Base risk per trade (fraction)
            max_risk: Maximum risk per trade
            min_risk: Minimum risk per trade
            kelly_fraction: Fraction of Kelly to use (for safety)
        """
        self.method = method
        self.base_risk = base_risk
        self.max_risk = max_risk
        self.min_risk = min_risk
        self.kelly_fraction = kelly_fraction

        # Performance tracking
        self.trade_results: List[Dict] = []
        self.win_rate = 0.5
        self.avg_win = 0.0
        self.avg_loss = 0.0

        logger.info(f"PositionSizer initialized with method: {method}")

    def calculate_size(
        self,
        balance: float,
        entry_price: float,
        stop_loss: float,
        confidence: float = 0.5,
        volatility: Optional[float] = None,
        atr: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Calculate optimal position size.

        Args:
            balance: Current account balance
            entry_price: Entry price
            stop_loss: Stop loss price
            confidence: Model confidence (0-1)
            volatility: Current market volatility
            atr: Average True Range

        Returns:
            Tuple of (position_size, risk_amount)
        """
        if self.method == 'fixed':
            return self._fixed_fractional(balance, entry_price, stop_loss)
        elif self.method == 'kelly':
            return self._kelly_criterion(balance, entry_price, stop_loss)
        elif self.method == 'atr':
            return self._atr_based(balance, entry_price, stop_loss, atr)
        else:  # dynamic
            return self._dynamic_sizing(
                balance, entry_price, stop_loss, confidence, volatility, atr
            )

    def _fixed_fractional(
        self,
        balance: float,
        entry_price: float,
        stop_loss: float
    ) -> Tuple[float, float]:
        """Fixed fractional position sizing."""
        risk_amount = balance * self.base_risk
        price_risk = abs(entry_price - stop_loss)

        if price_risk == 0:
            return 0.0, 0.0

        size = risk_amount / price_risk
        return size, risk_amount

    def _kelly_criterion(
        self,
        balance: float,
        entry_price: float,
        stop_loss: float
    ) -> Tuple[float, float]:
        """Kelly Criterion position sizing."""
        # Calculate Kelly percentage
        if self.avg_loss == 0 or self.win_rate <= 0:
            return self._fixed_fractional(balance, entry_price, stop_loss)

        win_loss_ratio = self.avg_win / self.avg_loss if self.avg_loss > 0 else 1.0
        kelly = self.win_rate - ((1 - self.win_rate) / win_loss_ratio)

        # Apply fraction for safety
        kelly = max(0, kelly * self.kelly_fraction)

        # Clamp to risk limits
        risk_pct = min(max(kelly, self.min_risk), self.max_risk)
        risk_amount = balance * risk_pct
        price_risk = abs(entry_price - stop_loss)

        if price_risk == 0:
            return 0.0, 0.0

        size = risk_amount / price_risk
        return size, risk_amount

    def _atr_based(
        self,
        balance: float,
        entry_price: float,
        stop_loss: float,
        atr: Optional[float]
    ) -> Tuple[float, float]:
        """ATR-based position sizing."""
        if atr is None or atr == 0:
            return self._fixed_fractional(balance, entry_price, stop_loss)

        # Adjust risk based on ATR
        baseline_atr_pct = 0.02  # 2% baseline
        current_atr_pct = atr / entry_price

        volatility_factor = baseline_atr_pct / current_atr_pct
        volatility_factor = min(max(volatility_factor, 0.5), 2.0)

        adjusted_risk = self.base_risk * volatility_factor
        adjusted_risk = min(max(adjusted_risk, self.min_risk), self.max_risk)

        risk_amount = balance * adjusted_risk
        price_risk = abs(entry_price - stop_loss)

        if price_risk == 0:
            return 0.0, 0.0

        size = risk_amount / price_risk
        return size, risk_amount

    def _dynamic_sizing(
        self,
        balance: float,
        entry_price: float,
        stop_loss: float,
        confidence: float,
        volatility: Optional[float],
        atr: Optional[float]
    ) -> Tuple[float, float]:
        """
        Dynamic position sizing based on multiple factors.
        Combines Kelly, volatility adjustment, and confidence scaling.
        """
        # Start with Kelly-based risk
        base_size, base_risk = self._kelly_criterion(balance, entry_price, stop_loss)

        # Confidence adjustment (0.5 to 1.5x)
        confidence_factor = 0.5 + confidence

        # Volatility adjustment
        volatility_factor = 1.0
        if volatility is not None and volatility > 0:
            baseline_vol = 0.02
            volatility_factor = baseline_vol / volatility
            volatility_factor = min(max(volatility_factor, 0.5), 1.5)

        # Recent performance adjustment
        performance_factor = self._calculate_performance_factor()

        # Combine factors
        total_factor = confidence_factor * volatility_factor * performance_factor

        adjusted_risk = base_risk * total_factor
        adjusted_risk = min(max(adjusted_risk, balance * self.min_risk), balance * self.max_risk)

        price_risk = abs(entry_price - stop_loss)
        if price_risk == 0:
            return 0.0, 0.0

        size = adjusted_risk / price_risk
        return size, adjusted_risk

    def _calculate_performance_factor(self) -> float:
        """Calculate performance-based adjustment factor."""
        if len(self.trade_results) < 5:
            return 1.0

        recent_trades = self.trade_results[-20:]
        recent_wins = sum(1 for t in recent_trades if t.get('pnl', 0) > 0)
        recent_win_rate = recent_wins / len(recent_trades)

        # Scale factor based on recent performance
        if recent_win_rate > 0.6:
            return 1.2
        elif recent_win_rate < 0.4:
            return 0.8
        return 1.0

    def update_stats(self, trade_result: Dict) -> None:
        """
        Update trading statistics after a trade.

        Args:
            trade_result: Trade result dictionary with 'pnl' and 'pnl_pct'
        """
        self.trade_results.append(trade_result)

        # Calculate updated statistics
        wins = [t for t in self.trade_results if t.get('pnl', 0) > 0]
        losses = [t for t in self.trade_results if t.get('pnl', 0) < 0]

        self.win_rate = len(wins) / len(self.trade_results) if self.trade_results else 0.5

        if wins:
            self.avg_win = np.mean([t['pnl_pct'] for t in wins])
        if losses:
            self.avg_loss = abs(np.mean([t['pnl_pct'] for t in losses]))

        logger.debug(f"Updated stats: Win Rate={self.win_rate:.2%}, Avg Win={self.avg_win:.4f}")

    def calculate_stop_loss(
        self,
        entry_price: float,
        side: str,
        atr: Optional[float] = None,
        support_resistance: Optional[float] = None,
        multiplier: float = 2.0
    ) -> float:
        """
        Calculate optimal stop loss level.

        Args:
            entry_price: Entry price
            side: 'long' or 'short'
            atr: Average True Range
            support_resistance: Nearest S/R level
            multiplier: ATR multiplier

        Returns:
            Stop loss price
        """
        # ATR-based stop
        atr_stop = None
        if atr is not None:
            if side == 'long':
                atr_stop = entry_price - (atr * multiplier)
            else:
                atr_stop = entry_price + (atr * multiplier)

        # S/R-based stop
        sr_stop = support_resistance

        # Default percentage stop
        pct_stop = entry_price * (1 - 0.02) if side == 'long' else entry_price * (1 + 0.02)

        # Choose the best stop
        if side == 'long':
            stops = [s for s in [atr_stop, sr_stop, pct_stop] if s is not None and s < entry_price]
            return max(stops) if stops else pct_stop
        else:
            stops = [s for s in [atr_stop, sr_stop, pct_stop] if s is not None and s > entry_price]
            return min(stops) if stops else pct_stop

    def calculate_take_profit(
        self,
        entry_price: float,
        stop_loss: float,
        side: str,
        risk_reward: float = 2.0
    ) -> float:
        """
        Calculate take profit level based on risk/reward ratio.

        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            side: 'long' or 'short'
            risk_reward: Desired risk/reward ratio

        Returns:
            Take profit price
        """
        risk = abs(entry_price - stop_loss)
        reward = risk * risk_reward

        if side == 'long':
            return entry_price + reward
        else:
            return entry_price - reward

    def get_sizing_recommendation(
        self,
        balance: float,
        entry_price: float,
        stop_loss: float,
        confidence: float,
        volatility: Optional[float] = None,
        atr: Optional[float] = None
    ) -> Dict:
        """
        Get comprehensive sizing recommendation.

        Args:
            balance: Current balance
            entry_price: Entry price
            stop_loss: Stop loss price
            confidence: Model confidence
            volatility: Market volatility
            atr: Average True Range

        Returns:
            Detailed sizing recommendation
        """
        size, risk_amount = self.calculate_size(
            balance, entry_price, stop_loss, confidence, volatility, atr
        )

        risk_pct = risk_amount / balance if balance > 0 else 0
        position_value = size * entry_price
        position_pct = position_value / balance if balance > 0 else 0

        return {
            'recommended_size': size,
            'risk_amount': risk_amount,
            'risk_percentage': risk_pct,
            'position_value': position_value,
            'position_percentage': position_pct,
            'method_used': self.method,
            'confidence_applied': confidence,
            'current_win_rate': self.win_rate,
            'kelly_percentage': self._calculate_kelly_pct() if self.avg_loss > 0 else None
        }

    def _calculate_kelly_pct(self) -> float:
        """Calculate raw Kelly percentage."""
        if self.avg_loss == 0:
            return 0.0
        win_loss_ratio = self.avg_win / self.avg_loss
        return self.win_rate - ((1 - self.win_rate) / win_loss_ratio)
