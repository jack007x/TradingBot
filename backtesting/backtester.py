"""
Backtester Module
==================
Vectorized backtesting engine for strategy evaluation.

Usage:
    from backtesting import Backtester
    from config import get_config

    config = get_config()
    backtester = Backtester(config)

    # Run backtest
    results = backtester.run(df_features, model)

    # Generate report
    report = backtester.generate_report(results)
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd

from config.config_loader import Config
from models.model_baseline import BaselineModel

logger = logging.getLogger(__name__)


class Backtester:
    """
    Vectorized backtesting engine.

    Simulates trading based on model predictions with:
    - Transaction costs (spread, commission, slippage)
    - Position sizing based on risk
    - Stop loss and take profit
    - Trading session filters
    """

    def __init__(self, config: Config):
        """
        Initialize backtester.

        Args:
            config: Configuration object
        """
        self.config = config

        # Backtest parameters
        self.initial_balance = config.backtesting.initial_balance
        self.spread_pips = config.backtesting.spread_pips
        self.commission_per_lot = config.backtesting.commission_per_lot
        self.slippage_pips = config.backtesting.slippage_pips

        # Risk parameters
        self.risk_per_trade = config.risk.risk_per_trade_percent / 100
        self.sl_atr_mult = config.risk.sl_atr_multiplier
        self.tp_atr_mult = config.risk.tp_atr_multiplier
        self.max_sl_pips = config.risk.max_sl_pips
        self.min_sl_pips = config.risk.min_sl_pips

        # Signal parameters
        self.min_probability = config.model.min_probability_threshold

        # Gold pip value (for XAUUSD, 1 pip = 0.01)
        self.pip_value = 0.01
        self.contract_size = 100  # 1 lot = 100 oz

    def run(
            self,
            df: pd.DataFrame,
            model: BaselineModel,
            feature_names: List[str],
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Run backtest simulation.

        Args:
            df: DataFrame with OHLCV and features
            model: Trained model for predictions
            feature_names: List of feature column names
            start_date: Start date for backtest
            end_date: End date for backtest

        Returns:
            Dict with backtest results
        """
        logger.info("Starting backtest simulation...")

        # Filter date range
        if start_date is not None:
            df = df[df['time'] >= start_date]
        if end_date is not None:
            df = df[df['time'] <= end_date]

        df = df.copy().reset_index(drop=True)

        if len(df) < 100:
            raise ValueError(f"Not enough data for backtest: {len(df)} bars")

        # Ensure features are available
        missing_features = set(feature_names) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        # Get predictions
        X = df[feature_names].values
        valid_mask = ~np.isnan(X).any(axis=1)

        predictions = np.zeros(len(df))
        probabilities = np.zeros((len(df), 3))

        if valid_mask.sum() > 0:
            predictions[valid_mask] = model.predict(X[valid_mask])
            probabilities[valid_mask] = model.predict_proba(X[valid_mask])

        df['prediction'] = predictions
        df['prob_down'] = probabilities[:, 0]
        df['prob_neutral'] = probabilities[:, 1]
        df['prob_up'] = probabilities[:, 2]
        df['max_prob'] = np.max(probabilities, axis=1)

        # Calculate ATR for position sizing
        if 'atr' not in df.columns:
            df['atr'] = self._calculate_atr(df, period=14)

        # Generate trade signals
        df = self._generate_signals(df)

        # Simulate trades
        trades, equity_curve = self._simulate_trades(df)

        # Calculate metrics
        metrics = self._calculate_metrics(trades, equity_curve)

        results = {
            'trades': trades,
            'equity_curve': equity_curve,
            'metrics': metrics,
            'df': df,
            'backtest_params': {
                'start_date': df['time'].min().isoformat(),
                'end_date': df['time'].max().isoformat(),
                'initial_balance': self.initial_balance,
                'total_bars': len(df),
            }
        }

        logger.info(f"Backtest complete: {len(trades)} trades, "
                    f"Final balance: ${equity_curve[-1]:,.2f}")

        return results

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(span=period, adjust=False).mean()

        return atr

    def _generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from predictions."""
        df = df.copy()

        # Base signal from prediction
        df['signal'] = 0

        # Buy signal: prediction = 1 (up) with high confidence
        buy_condition = (
                (df['prediction'] == 1) &
                (df['max_prob'] >= self.min_probability)
        )
        df.loc[buy_condition, 'signal'] = 1

        # Sell signal: prediction = -1 (down) with high confidence
        sell_condition = (
                (df['prediction'] == -1) &
                (df['max_prob'] >= self.min_probability)
        )
        df.loc[sell_condition, 'signal'] = -1

        # Filter by trading hours (if time features available)
        if 'hour' in df.columns:
            blocked_hours = self.config.trading.blocked_hours_utc
            blocked_mask = df['hour'].isin(blocked_hours)
            df.loc[blocked_mask, 'signal'] = 0

        # Filter extreme volatility (ATR too high or too low)
        if 'atr' in df.columns:
            atr_mean = df['atr'].rolling(100).mean()
            atr_std = df['atr'].rolling(100).std()
            extreme_vol = (df['atr'] > atr_mean + 3 * atr_std) | (df['atr'] < atr_mean - 2 * atr_std)
            df.loc[extreme_vol, 'signal'] = 0

        return df

    def _simulate_trades(
            self,
            df: pd.DataFrame
    ) -> Tuple[List[Dict], np.ndarray]:
        """
        Simulate trades based on signals.

        Returns:
            Tuple[trade_list, equity_curve]
        """
        trades = []
        equity_curve = np.zeros(len(df))
        equity_curve[0] = self.initial_balance

        balance = self.initial_balance
        position = None  # Current open position

        for i in range(1, len(df)):
            current_bar = df.iloc[i]
            prev_bar = df.iloc[i - 1]

            # Check if we have an open position
            if position is not None:
                # Check exit conditions
                exit_price, exit_reason = self._check_exit(
                    position, current_bar
                )

                if exit_price is not None:
                    # Close position
                    pnl = self._calculate_pnl(
                        position['direction'],
                        position['entry_price'],
                        exit_price,
                        position['lot_size']
                    )

                    balance += pnl

                    # Record trade
                    trade = {
                        'entry_time': position['entry_time'],
                        'exit_time': current_bar['time'],
                        'direction': position['direction'],
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'lot_size': position['lot_size'],
                        'sl': position['sl'],
                        'tp': position['tp'],
                        'pnl': pnl,
                        'pnl_pips': (exit_price - position['entry_price']) * position['direction'] / self.pip_value,
                        'exit_reason': exit_reason,
                        'balance_after': balance,
                    }
                    trades.append(trade)
                    position = None

            # Check for new entry signal (only if no open position)
            if position is None and prev_bar['signal'] != 0:
                signal = prev_bar['signal']
                atr = prev_bar['atr'] if 'atr' in prev_bar and not np.isnan(prev_bar['atr']) else 1.0

                # Calculate entry, SL, TP
                entry_price = current_bar['open']

                # Adjust for spread and slippage
                if signal == 1:  # Buy
                    entry_price += (self.spread_pips + self.slippage_pips) * self.pip_value
                    sl_distance = atr * self.sl_atr_mult
                    sl_distance = np.clip(sl_distance, self.min_sl_pips * self.pip_value, self.max_sl_pips * self.pip_value)
                    sl = entry_price - sl_distance
                    tp = entry_price + sl_distance * (self.tp_atr_mult / self.sl_atr_mult)
                else:  # Sell
                    entry_price -= self.slippage_pips * self.pip_value
                    sl_distance = atr * self.sl_atr_mult
                    sl_distance = np.clip(sl_distance, self.min_sl_pips * self.pip_value, self.max_sl_pips * self.pip_value)
                    sl = entry_price + sl_distance
                    tp = entry_price - sl_distance * (self.tp_atr_mult / self.sl_atr_mult)

                # Position sizing
                lot_size = self._calculate_lot_size(balance, sl_distance)

                if lot_size >= self.config.risk.min_lot_size:
                    position = {
                        'entry_time': current_bar['time'],
                        'direction': signal,
                        'entry_price': entry_price,
                        'lot_size': lot_size,
                        'sl': sl,
                        'tp': tp,
                        'bars_held': 0,
                    }

            # Update position bars held
            if position is not None:
                position['bars_held'] += 1

            equity_curve[i] = balance

        # Close any remaining position at end
        if position is not None:
            exit_price = df.iloc[-1]['close']
            pnl = self._calculate_pnl(
                position['direction'],
                position['entry_price'],
                exit_price,
                position['lot_size']
            )
            balance += pnl

            trade = {
                'entry_time': position['entry_time'],
                'exit_time': df.iloc[-1]['time'],
                'direction': position['direction'],
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'lot_size': position['lot_size'],
                'sl': position['sl'],
                'tp': position['tp'],
                'pnl': pnl,
                'pnl_pips': (exit_price - position['entry_price']) * position['direction'] / self.pip_value,
                'exit_reason': 'end_of_data',
                'balance_after': balance,
            }
            trades.append(trade)
            equity_curve[-1] = balance

        return trades, equity_curve

    def _check_exit(
            self,
            position: Dict,
            current_bar: pd.Series
    ) -> Tuple[Optional[float], Optional[str]]:
        """
        Check if position should be closed.

        Returns:
            Tuple[exit_price, exit_reason] or (None, None) if no exit
        """
        direction = position['direction']
        sl = position['sl']
        tp = position['tp']

        high = current_bar['high']
        low = current_bar['low']

        # Check stop loss
        if direction == 1:  # Long
            if low <= sl:
                return sl, 'stop_loss'
            if high >= tp:
                return tp, 'take_profit'
        else:  # Short
            if high >= sl:
                return sl, 'stop_loss'
            if low <= tp:
                return tp, 'take_profit'

        # Time-based exit (max bars held)
        max_bars = self.config.model.prediction_horizon_bars * 3
        if position['bars_held'] >= max_bars:
            return current_bar['close'], 'time_exit'

        return None, None

    def _calculate_lot_size(
            self,
            balance: float,
            sl_distance: float
    ) -> float:
        """Calculate lot size based on risk."""
        risk_amount = balance * self.risk_per_trade

        # For gold: pip value per lot = $1 per 0.01 move per 1 lot (100 oz)
        pip_value_per_lot = self.contract_size * self.pip_value

        # Lot size = risk_amount / (sl_pips * pip_value_per_lot)
        sl_pips = sl_distance / self.pip_value
        lot_size = risk_amount / (sl_pips * pip_value_per_lot)

        # Apply constraints
        lot_size = max(self.config.risk.min_lot_size, lot_size)
        lot_size = min(self.config.risk.max_lot_size, lot_size)

        # Round to lot step
        lot_size = round(lot_size, 2)

        return lot_size

    def _calculate_pnl(
            self,
            direction: int,
            entry_price: float,
            exit_price: float,
            lot_size: float
    ) -> float:
        """Calculate profit/loss for a trade."""
        price_diff = (exit_price - entry_price) * direction
        pips = price_diff / self.pip_value

        # PnL = pips * pip_value_per_lot * lot_size
        pip_value_per_lot = self.contract_size * self.pip_value
        pnl = pips * pip_value_per_lot * lot_size

        # Subtract commission
        pnl -= self.commission_per_lot * lot_size

        return pnl

    def _calculate_metrics(
            self,
            trades: List[Dict],
            equity_curve: np.ndarray
    ) -> Dict[str, float]:
        """Calculate performance metrics."""
        metrics = {}

        if len(trades) == 0:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_pnl': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'total_return': 0,
                'total_return_pct': 0,
                'max_drawdown': 0,
                'max_drawdown_pct': 0,
                'sharpe_ratio': 0,
                'calmar_ratio': 0,
                'avg_trade_duration_hours': 0,
            }

        # Trade statistics
        pnls = np.array([t['pnl'] for t in trades])
        wins = pnls > 0
        losses = pnls < 0

        metrics['total_trades'] = len(trades)
        metrics['winning_trades'] = wins.sum()
        metrics['losing_trades'] = losses.sum()
        metrics['win_rate'] = wins.mean()

        # PnL metrics
        metrics['total_pnl'] = pnls.sum()
        metrics['avg_pnl'] = pnls.mean()
        metrics['avg_win'] = pnls[wins].mean() if wins.sum() > 0 else 0
        metrics['avg_loss'] = pnls[losses].mean() if losses.sum() > 0 else 0

        # Profit factor
        gross_profit = pnls[wins].sum() if wins.sum() > 0 else 0
        gross_loss = abs(pnls[losses].sum()) if losses.sum() > 0 else 0
        metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Return metrics
        metrics['total_return'] = (equity_curve[-1] / equity_curve[0]) - 1
        metrics['total_return_pct'] = metrics['total_return'] * 100

        # Drawdown
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        metrics['max_drawdown'] = abs(drawdown.min())
        metrics['max_drawdown_pct'] = metrics['max_drawdown'] * 100

        # Sharpe ratio (assuming daily returns for equity curve)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        if len(returns) > 1 and returns.std() > 0:
            # Annualized (assuming 252 trading days, 96 M15 bars per day)
            bars_per_year = 252 * 96
            sharpe = (returns.mean() / returns.std()) * np.sqrt(bars_per_year)
            metrics['sharpe_ratio'] = sharpe
        else:
            metrics['sharpe_ratio'] = 0

        # Calmar ratio
        if metrics['max_drawdown'] > 0:
            metrics['calmar_ratio'] = metrics['total_return'] / metrics['max_drawdown']
        else:
            metrics['calmar_ratio'] = float('inf') if metrics['total_return'] > 0 else 0

        # Trade duration
        durations = [(t['exit_time'] - t['entry_time']).total_seconds() / 3600 for t in trades
                     if isinstance(t['entry_time'], pd.Timestamp)]
        if durations:
            metrics['avg_trade_duration_hours'] = np.mean(durations)
        else:
            metrics['avg_trade_duration_hours'] = 0

        # Exit reason distribution
        exit_reasons = [t['exit_reason'] for t in trades]
        for reason in set(exit_reasons):
            metrics[f'exit_{reason}_count'] = exit_reasons.count(reason)

        return metrics

    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate text report of backtest results."""
        metrics = results['metrics']
        params = results['backtest_params']

        lines = [
            "=" * 60,
            "BACKTEST REPORT",
            "=" * 60,
            "",
            "PARAMETERS",
            "-" * 30,
            f"  Period: {params['start_date'][:10]} to {params['end_date'][:10]}",
            f"  Initial Balance: ${params['initial_balance']:,.2f}",
            f"  Total Bars: {params['total_bars']:,}",
            "",
            "PERFORMANCE SUMMARY",
            "-" * 30,
            f"  Total Trades: {metrics['total_trades']}",
            f"  Win Rate: {metrics['win_rate']:.2%}",
            f"  Profit Factor: {metrics['profit_factor']:.3f}",
            "",
            f"  Total PnL: ${metrics['total_pnl']:,.2f}",
            f"  Total Return: {metrics['total_return_pct']:.2f}%",
            f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}",
            f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%",
            f"  Calmar Ratio: {metrics['calmar_ratio']:.3f}",
            "",
            "TRADE STATISTICS",
            "-" * 30,
            f"  Winning Trades: {metrics['winning_trades']}",
            f"  Losing Trades: {metrics['losing_trades']}",
            f"  Avg Win: ${metrics['avg_win']:,.2f}",
            f"  Avg Loss: ${metrics['avg_loss']:,.2f}",
            f"  Avg Trade Duration: {metrics.get('avg_trade_duration_hours', 0):.1f} hours",
            "",
        ]

        # Exit reasons
        exit_counts = {k: v for k, v in metrics.items() if k.startswith('exit_') and k.endswith('_count')}
        if exit_counts:
            lines.append("EXIT REASONS")
            lines.append("-" * 30)
            for reason, count in exit_counts.items():
                reason_name = reason.replace('exit_', '').replace('_count', '')
                lines.append(f"  {reason_name}: {count}")
            lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)

    def save_results(
            self,
            results: Dict[str, Any],
            output_dir: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Save backtest results to files.

        Returns:
            Dict with paths to saved files
        """
        if output_dir is None:
            output_dir = self.config.get_reports_path()

        from pathlib import Path
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = {}

        # Save trade log
        if results['trades']:
            trades_df = pd.DataFrame(results['trades'])
            trades_path = output_dir / f"trades_{timestamp}.csv"
            trades_df.to_csv(trades_path, index=False)
            saved_files['trades'] = str(trades_path)

        # Save equity curve
        equity_df = pd.DataFrame({
            'equity': results['equity_curve']
        })
        equity_path = output_dir / f"equity_{timestamp}.csv"
        equity_df.to_csv(equity_path, index=False)
        saved_files['equity'] = str(equity_path)

        # Save report
        report = self.generate_report(results)
        report_path = output_dir / f"report_{timestamp}.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        saved_files['report'] = str(report_path)

        logger.info(f"Saved backtest results to {output_dir}")

        return saved_files


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from config import get_config
    from data.data_fetcher import DataFetcher
    from features.feature_engineering import FeatureEngineer
    from models.model_trainer import ModelTrainer

    config = get_config()
    fetcher = DataFetcher(config)
    engineer = FeatureEngineer(config)
    trainer = ModelTrainer(config)
    backtester = Backtester(config)

    # Prepare data
    from datetime import datetime
    start = datetime(2022, 1, 1)
    end = datetime(2024, 6, 1)
    df = fetcher.fetch_historical(start, end)
    df_features = engineer.build_features(df)
    X, y, feature_names = engineer.prepare_training_data(df_features)

    # Train model on first portion
    train_end = int(len(X) * 0.7)
    X_train, y_train = X[:train_end], y[:train_end]
    model, _ = trainer.train_simple(X_train, y_train, feature_names)

    # Backtest on remaining data
    df_test = df_features.iloc[train_end:].copy()
    print(f"\nBacktesting on {len(df_test)} bars...")

    results = backtester.run(df_test, model, feature_names)

    # Print report
    print("\n" + backtester.generate_report(results))

    # Save results
    saved = backtester.save_results(results)
    print("\nSaved files:")
    for name, path in saved.items():
        print(f"  {name}: {path}")
