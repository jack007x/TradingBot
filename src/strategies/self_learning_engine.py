"""
Self-Learning Engine for AI Trading Bot.
Continuously learns and adapts from trading results.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
from loguru import logger


@dataclass
class LearningState:
    """Current learning state."""
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    best_strategy_params: Dict = field(default_factory=dict)
    model_performances: Dict = field(default_factory=dict)
    last_update: Optional[datetime] = None
    adaptation_count: int = 0


class SelfLearningEngine:
    """
    Self-learning engine that continuously improves trading strategies
    based on real-time performance feedback.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        adaptation_threshold: float = 0.1,
        min_trades_for_update: int = 10,
        performance_window: int = 100,
        enable_parameter_evolution: bool = True,
        enable_model_selection: bool = True,
        enable_strategy_adaptation: bool = True
    ):
        """
        Initialize self-learning engine.

        Args:
            learning_rate: Rate of parameter updates
            adaptation_threshold: Performance change threshold for adaptation
            min_trades_for_update: Minimum trades before updating
            performance_window: Number of trades for performance evaluation
            enable_parameter_evolution: Enable parameter optimization
            enable_model_selection: Enable dynamic model selection
            enable_strategy_adaptation: Enable strategy adaptation
        """
        self.learning_rate = learning_rate
        self.adaptation_threshold = adaptation_threshold
        self.min_trades_for_update = min_trades_for_update
        self.performance_window = performance_window
        self.enable_parameter_evolution = enable_parameter_evolution
        self.enable_model_selection = enable_model_selection
        self.enable_strategy_adaptation = enable_strategy_adaptation

        # State tracking
        self.state = LearningState()
        self.trade_history: List[Dict] = []
        self.parameter_history: List[Dict] = []
        self.model_weights: Dict[str, float] = {}

        # Current strategy parameters
        self.strategy_params = {
            'take_profit': 0.03,
            'stop_loss': 0.02,
            'trailing_stop_pct': 0.01,
            'position_size_pct': 0.1,
            'entry_threshold': 0.6,
            'exit_threshold': 0.4,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'ma_fast': 10,
            'ma_slow': 50
        }

        # Performance baselines
        self.baseline_metrics = {
            'win_rate': 0.5,
            'profit_factor': 1.0,
            'sharpe_ratio': 0.0,
            'avg_trade_pnl': 0.0
        }

        logger.info("SelfLearningEngine initialized")

    def record_trade(
        self,
        trade: Dict,
        model_predictions: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Record a completed trade for learning.

        Args:
            trade: Trade result dictionary
            model_predictions: Dictionary of model predictions and their accuracy
        """
        self.trade_history.append({
            **trade,
            'timestamp': datetime.utcnow().isoformat(),
            'model_predictions': model_predictions or {}
        })

        self.state.total_trades += 1
        if trade.get('pnl', 0) > 0:
            self.state.winning_trades += 1
        self.state.total_pnl += trade.get('pnl', 0)

        # Update model performance tracking
        if model_predictions:
            for model_name, was_correct in model_predictions.items():
                if model_name not in self.state.model_performances:
                    self.state.model_performances[model_name] = {'correct': 0, 'total': 0}
                self.state.model_performances[model_name]['total'] += 1
                if was_correct:
                    self.state.model_performances[model_name]['correct'] += 1

        # Check if adaptation needed
        if self.state.total_trades % self.min_trades_for_update == 0:
            self._check_and_adapt()

    def _check_and_adapt(self) -> None:
        """Check performance and trigger adaptations if needed."""
        current_metrics = self._calculate_current_metrics()

        # Check for significant performance change
        performance_change = self._calculate_performance_change(current_metrics)

        if abs(performance_change) > self.adaptation_threshold:
            logger.info(f"Performance change detected: {performance_change:.2%}")

            if self.enable_parameter_evolution:
                self._evolve_parameters(current_metrics)

            if self.enable_model_selection:
                self._update_model_weights()

            if self.enable_strategy_adaptation:
                self._adapt_strategy(current_metrics)

            self.state.adaptation_count += 1
            self.state.last_update = datetime.utcnow()

            # Update baseline
            self.baseline_metrics = current_metrics.copy()

    def _calculate_current_metrics(self) -> Dict[str, float]:
        """Calculate current performance metrics."""
        recent_trades = self.trade_history[-self.performance_window:]

        if not recent_trades:
            return self.baseline_metrics.copy()

        wins = [t for t in recent_trades if t.get('pnl', 0) > 0]
        losses = [t for t in recent_trades if t.get('pnl', 0) < 0]

        win_rate = len(wins) / len(recent_trades) if recent_trades else 0

        gross_profit = sum(t['pnl'] for t in wins) if wins else 0
        gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 0.001
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        pnls = [t.get('pnl', 0) for t in recent_trades]
        avg_pnl = np.mean(pnls) if pnls else 0
        std_pnl = np.std(pnls) if len(pnls) > 1 else 0.001
        sharpe = (avg_pnl / std_pnl) * np.sqrt(252) if std_pnl > 0 else 0

        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'avg_trade_pnl': avg_pnl
        }

    def _calculate_performance_change(self, current_metrics: Dict) -> float:
        """Calculate overall performance change from baseline."""
        changes = []

        for metric, baseline in self.baseline_metrics.items():
            current = current_metrics.get(metric, baseline)
            if baseline != 0:
                change = (current - baseline) / abs(baseline)
                changes.append(change)

        return np.mean(changes) if changes else 0.0

    def _evolve_parameters(self, current_metrics: Dict) -> None:
        """Evolve strategy parameters based on performance."""
        logger.info("Evolving strategy parameters...")

        # Analyze recent trades to determine adjustments
        recent_trades = self.trade_history[-self.performance_window:]

        # Analyze take profit effectiveness
        tp_hits = [t for t in recent_trades if t.get('reason') == 'take_profit']
        sl_hits = [t for t in recent_trades if t.get('reason') == 'stop_loss']

        if tp_hits and sl_hits:
            tp_ratio = len(tp_hits) / (len(tp_hits) + len(sl_hits))

            # If hitting TP too rarely, consider widening it
            if tp_ratio < 0.3:
                self.strategy_params['take_profit'] *= (1 + self.learning_rate)
            elif tp_ratio > 0.7:
                self.strategy_params['take_profit'] *= (1 - self.learning_rate * 0.5)

        # Analyze stop loss effectiveness
        avg_sl_loss = np.mean([t['pnl'] for t in sl_hits]) if sl_hits else 0
        avg_win = np.mean([t['pnl'] for t in tp_hits]) if tp_hits else 0

        if avg_sl_loss != 0 and avg_win != 0:
            risk_reward = abs(avg_win / avg_sl_loss)
            if risk_reward < 1.5:
                # Tighten stop loss
                self.strategy_params['stop_loss'] *= (1 - self.learning_rate * 0.5)
            elif risk_reward > 3:
                # Can afford wider stop loss
                self.strategy_params['stop_loss'] *= (1 + self.learning_rate * 0.3)

        # Clamp parameters to reasonable ranges
        self.strategy_params['take_profit'] = np.clip(self.strategy_params['take_profit'], 0.01, 0.15)
        self.strategy_params['stop_loss'] = np.clip(self.strategy_params['stop_loss'], 0.005, 0.05)

        # Record parameter change
        self.parameter_history.append({
            'timestamp': datetime.utcnow().isoformat(),
            'params': self.strategy_params.copy(),
            'metrics': current_metrics
        })

        logger.info(f"Parameters evolved: TP={self.strategy_params['take_profit']:.4f}, SL={self.strategy_params['stop_loss']:.4f}")

    def _update_model_weights(self) -> None:
        """Update weights for ensemble model selection."""
        if not self.state.model_performances:
            return

        logger.info("Updating model weights...")

        total_accuracy = 0
        accuracies = {}

        for model_name, perf in self.state.model_performances.items():
            if perf['total'] > 0:
                acc = perf['correct'] / perf['total']
                accuracies[model_name] = acc
                total_accuracy += acc

        # Normalize to get weights
        if total_accuracy > 0:
            for model_name, acc in accuracies.items():
                # Apply softmax-like transformation
                self.model_weights[model_name] = np.exp(acc * 2) / sum(np.exp(a * 2) for a in accuracies.values())

        logger.info(f"Model weights updated: {self.model_weights}")

    def _adapt_strategy(self, current_metrics: Dict) -> None:
        """Adapt overall strategy based on market conditions."""
        logger.info("Adapting strategy...")

        # Analyze recent market behavior
        recent_trades = self.trade_history[-self.performance_window:]

        if len(recent_trades) < 10:
            return

        # Calculate volatility of recent PnLs
        pnls = [t.get('pnl_pct', 0) for t in recent_trades]
        volatility = np.std(pnls)

        # Adjust entry threshold based on accuracy
        current_win_rate = current_metrics.get('win_rate', 0.5)

        if current_win_rate < 0.45:
            # Being less selective might help
            self.strategy_params['entry_threshold'] = min(0.8, self.strategy_params['entry_threshold'] + 0.05)
        elif current_win_rate > 0.6:
            # Can be more aggressive
            self.strategy_params['entry_threshold'] = max(0.4, self.strategy_params['entry_threshold'] - 0.05)

        # Adjust position size based on volatility
        if volatility > 0.05:
            self.strategy_params['position_size_pct'] = max(0.05, self.strategy_params['position_size_pct'] * 0.9)
        elif volatility < 0.02:
            self.strategy_params['position_size_pct'] = min(0.2, self.strategy_params['position_size_pct'] * 1.1)

        logger.info(f"Strategy adapted: Entry threshold={self.strategy_params['entry_threshold']:.2f}")

    def get_current_params(self) -> Dict:
        """Get current strategy parameters."""
        return self.strategy_params.copy()

    def get_model_weights(self) -> Dict[str, float]:
        """Get current model weights for ensemble."""
        return self.model_weights.copy()

    def get_learning_stats(self) -> Dict:
        """Get learning statistics."""
        return {
            'total_trades': self.state.total_trades,
            'win_rate': self.state.winning_trades / max(1, self.state.total_trades),
            'total_pnl': self.state.total_pnl,
            'adaptation_count': self.state.adaptation_count,
            'last_update': self.state.last_update.isoformat() if self.state.last_update else None,
            'current_params': self.strategy_params,
            'model_weights': self.model_weights,
            'model_performances': {
                name: perf['correct'] / max(1, perf['total'])
                for name, perf in self.state.model_performances.items()
            }
        }

    def suggest_action(
        self,
        predictions: Dict[str, Dict],
        current_price: float,
        features: Optional[Dict] = None
    ) -> Dict:
        """
        Suggest trading action based on weighted model predictions.

        Args:
            predictions: Dictionary of model predictions {model_name: {signal, confidence}}
            current_price: Current market price
            features: Additional market features

        Returns:
            Action recommendation
        """
        if not predictions:
            return {'action': 'hold', 'confidence': 0.0, 'reason': 'No predictions available'}

        # Weighted voting
        buy_score = 0.0
        sell_score = 0.0
        hold_score = 0.0

        for model_name, pred in predictions.items():
            weight = self.model_weights.get(model_name, 1.0 / len(predictions))
            confidence = pred.get('confidence', 0.5)
            signal = pred.get('signal', 'hold')

            weighted_confidence = weight * confidence

            if signal == 'buy':
                buy_score += weighted_confidence
            elif signal == 'sell':
                sell_score += weighted_confidence
            else:
                hold_score += weighted_confidence

        # Determine action
        total_score = buy_score + sell_score + hold_score
        if total_score == 0:
            return {'action': 'hold', 'confidence': 0.0, 'reason': 'Zero total score'}

        buy_pct = buy_score / total_score
        sell_pct = sell_score / total_score

        # Apply entry threshold
        threshold = self.strategy_params['entry_threshold']

        if buy_pct > threshold and buy_pct > sell_pct:
            return {
                'action': 'buy',
                'confidence': buy_pct,
                'take_profit': current_price * (1 + self.strategy_params['take_profit']),
                'stop_loss': current_price * (1 - self.strategy_params['stop_loss']),
                'position_size_pct': self.strategy_params['position_size_pct'],
                'reason': f'Buy signal with {buy_pct:.1%} confidence'
            }
        elif sell_pct > threshold and sell_pct > buy_pct:
            return {
                'action': 'sell',
                'confidence': sell_pct,
                'take_profit': current_price * (1 - self.strategy_params['take_profit']),
                'stop_loss': current_price * (1 + self.strategy_params['stop_loss']),
                'position_size_pct': self.strategy_params['position_size_pct'],
                'reason': f'Sell signal with {sell_pct:.1%} confidence'
            }

        return {
            'action': 'hold',
            'confidence': max(buy_pct, sell_pct),
            'reason': f'Below threshold ({threshold:.1%}). Buy: {buy_pct:.1%}, Sell: {sell_pct:.1%}'
        }

    def save(self, filepath: Path) -> None:
        """Save learning state to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'strategy_params': self.strategy_params,
            'model_weights': self.model_weights,
            'baseline_metrics': self.baseline_metrics,
            'state': {
                'total_trades': self.state.total_trades,
                'winning_trades': self.state.winning_trades,
                'total_pnl': self.state.total_pnl,
                'adaptation_count': self.state.adaptation_count,
                'model_performances': self.state.model_performances
            },
            'parameter_history': self.parameter_history[-100:]  # Keep last 100
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Learning state saved to {filepath}")

    def load(self, filepath: Path) -> None:
        """Load learning state from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.strategy_params = data.get('strategy_params', self.strategy_params)
        self.model_weights = data.get('model_weights', {})
        self.baseline_metrics = data.get('baseline_metrics', self.baseline_metrics)
        self.parameter_history = data.get('parameter_history', [])

        state_data = data.get('state', {})
        self.state.total_trades = state_data.get('total_trades', 0)
        self.state.winning_trades = state_data.get('winning_trades', 0)
        self.state.total_pnl = state_data.get('total_pnl', 0)
        self.state.adaptation_count = state_data.get('adaptation_count', 0)
        self.state.model_performances = state_data.get('model_performances', {})

        logger.info(f"Learning state loaded from {filepath}")
