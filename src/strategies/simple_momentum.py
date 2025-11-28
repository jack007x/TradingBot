"""
Simple Momentum Strategy - Fallback when ML models fail.

Proven technical analysis approach for Gold (XAUUSD).
Gold tends to trend rather than mean-revert, making momentum strategies effective.

Research shows momentum strategies work well for commodities:
- Gold respects technical levels
- Trending behavior during major sessions
- RSI and MACD provide reliable signals

This strategy serves as fallback when ML models:
- Have low confidence (<30%)
- Disagree on direction
- All fail minimum thresholds
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from loguru import logger


class SimpleMomentumStrategy:
    """
    Simple momentum-based strategy as fallback/confirmation.

    Trading Rules:
    ============
    BUY when (score >= 2):
    - Price > SMA20 (+1)
    - RSI > 55 and < 70 (+1)
    - MACD > Signal (+1)
    - MACD crossover up (+2, strong signal)
    - Near BB lower band (+1)

    SELL when (score >= 2):
    - Price < SMA20 (+1)
    - RSI < 45 and > 30 (+1)
    - MACD < Signal (+1)
    - MACD crossover down (+2, strong signal)
    - Near BB upper band (+1)

    HOLD otherwise

    Confidence:
    - Score 2-3: 40-60% confidence
    - Score 4-5: 80-100% confidence
    - Weak trend (ADX < 20): Reduce confidence by 50%
    """

    def __init__(
        self,
        rsi_buy_threshold: float = 55,
        rsi_sell_threshold: float = 45,
        min_score_to_trade: int = 2
    ):
        """
        Initialize momentum strategy.

        Args:
            rsi_buy_threshold: RSI level for buy signal (default 55)
            rsi_sell_threshold: RSI level for sell signal (default 45)
            min_score_to_trade: Minimum score needed to generate signal (default 2)
        """
        self.rsi_buy_threshold = rsi_buy_threshold
        self.rsi_sell_threshold = rsi_sell_threshold
        self.min_score_to_trade = min_score_to_trade

        logger.info("SimpleMomentumStrategy initialized")
        logger.info(f"  RSI thresholds: {rsi_sell_threshold}-{rsi_buy_threshold}")
        logger.info(f"  Min score: {min_score_to_trade}")

    def get_signal(self, df: pd.DataFrame) -> Dict:
        """
        Generate trading signal from latest data.

        Args:
            df: DataFrame with technical indicators

        Returns:
            Signal dict with:
                - signal: 'buy', 'sell', or 'hold'
                - confidence: 0.0 to 1.0
                - reasons: List of contributing factors
                - buy_score: Raw buy score
                - sell_score: Raw sell score
        """
        if len(df) < 2:
            return {
                'signal': 'hold',
                'confidence': 0,
                'reasons': ['Insufficient data'],
                'buy_score': 0,
                'sell_score': 0
            }

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        buy_signals = []
        sell_signals = []
        reasons = []

        # Signal 1: Price vs SMA20
        if 'sma_20' in df.columns and pd.notna(latest['sma_20']):
            if latest['close'] > latest['sma_20']:
                buy_signals.append(1)
                reasons.append('Price > SMA20')
            elif latest['close'] < latest['sma_20']:
                sell_signals.append(1)
                reasons.append('Price < SMA20')

        # Signal 2: RSI
        if 'rsi_14' in df.columns and pd.notna(latest['rsi_14']):
            rsi = latest['rsi_14']
            if rsi > self.rsi_buy_threshold and rsi < 70:  # Bullish but not overbought
                buy_signals.append(1)
                reasons.append(f'RSI={rsi:.1f} (bullish)')
            elif rsi < self.rsi_sell_threshold and rsi > 30:  # Bearish but not oversold
                sell_signals.append(1)
                reasons.append(f'RSI={rsi:.1f} (bearish)')

        # Signal 3: MACD position
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            if pd.notna(latest['macd']) and pd.notna(latest['macd_signal']):
                if latest['macd'] > latest['macd_signal']:
                    buy_signals.append(1)
                    reasons.append('MACD > Signal')
                elif latest['macd'] < latest['macd_signal']:
                    sell_signals.append(1)
                    reasons.append('MACD < Signal')

        # Signal 4: MACD Crossover (STRONG SIGNAL - double weight)
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            if (pd.notna(latest['macd']) and pd.notna(latest['macd_signal']) and
                pd.notna(prev['macd']) and pd.notna(prev['macd_signal'])):

                macd_cross_up = (latest['macd'] > latest['macd_signal']) and \
                               (prev['macd'] <= prev['macd_signal'])
                macd_cross_down = (latest['macd'] < latest['macd_signal']) and \
                                 (prev['macd'] >= prev['macd_signal'])

                if macd_cross_up:
                    buy_signals.append(2)  # Double weight!
                    reasons.append('🔥 MACD Crossover UP')
                elif macd_cross_down:
                    sell_signals.append(2)  # Double weight!
                    reasons.append('🔥 MACD Crossover DOWN')

        # Signal 5: Bollinger Band position
        if 'bb_position_20' in df.columns and pd.notna(latest['bb_position_20']):
            bb_pos = latest['bb_position_20']
            if bb_pos < 0.2:  # Near lower band - potential bounce
                buy_signals.append(1)
                reasons.append('Near BB lower (oversold)')
            elif bb_pos > 0.8:  # Near upper band - potential reversal
                sell_signals.append(1)
                reasons.append('Near BB upper (overbought)')

        # Calculate scores
        buy_score = sum(buy_signals)
        sell_score = sum(sell_signals)
        total_score = buy_score + sell_score

        # Check trend strength (ADX)
        trend_multiplier = 1.0
        if 'adx' in df.columns and pd.notna(latest['adx']):
            adx = latest['adx']
            if adx < 20:  # Weak trend - reduce confidence
                trend_multiplier = 0.5
                reasons.append(f'⚠️  Weak trend (ADX={adx:.1f})')
            elif adx > 40:  # Strong trend - boost confidence
                trend_multiplier = 1.2
                reasons.append(f'✅ Strong trend (ADX={adx:.1f})')

        # Generate final signal
        if total_score == 0:
            return {
                'signal': 'hold',
                'confidence': 0,
                'reasons': ['No clear signals'],
                'buy_score': buy_score,
                'sell_score': sell_score
            }

        # Determine direction
        if buy_score > sell_score and buy_score >= self.min_score_to_trade:
            # Calculate confidence: score/5 * trend_multiplier
            base_confidence = min(buy_score / 5, 1.0)
            final_confidence = min(base_confidence * trend_multiplier, 1.0)

            return {
                'signal': 'buy',
                'confidence': final_confidence,
                'reasons': reasons,
                'buy_score': buy_score,
                'sell_score': sell_score
            }

        elif sell_score > buy_score and sell_score >= self.min_score_to_trade:
            base_confidence = min(sell_score / 5, 1.0)
            final_confidence = min(base_confidence * trend_multiplier, 1.0)

            return {
                'signal': 'sell',
                'confidence': final_confidence,
                'reasons': reasons,
                'buy_score': buy_score,
                'sell_score': sell_score
            }
        else:
            # Mixed signals or score too low
            return {
                'signal': 'hold',
                'confidence': 0.3,
                'reasons': reasons + ['Mixed signals or low score'],
                'buy_score': buy_score,
                'sell_score': sell_score
            }

    def backtest(self, df: pd.DataFrame, verbose: bool = False) -> Dict:
        """
        Quick backtest to validate strategy performance.

        Args:
            df: DataFrame with OHLC and indicators
            verbose: Print detailed results

        Returns:
            Dict with performance metrics:
                - win_rate: Percentage of profitable trades
                - avg_return: Average return per trade
                - sharpe: Sharpe ratio
                - total_trades: Number of trades executed
                - total_signals: Number of signals generated
        """
        if len(df) < 60:
            logger.warning("Insufficient data for backtest (need >=60 bars)")
            return {
                'win_rate': 0,
                'avg_return': 0,
                'sharpe': 0,
                'total_trades': 0,
                'total_signals': 0
            }

        signals = []
        returns = []

        # Start from bar 50 to ensure indicators are stable
        for i in range(50, len(df) - 1):
            window = df.iloc[:i+1]
            signal = self.get_signal(window)
            signals.append(signal['signal'])

            # Calculate next bar return
            current_price = df.iloc[i]['close']
            next_price = df.iloc[i+1]['close']
            future_return = (next_price - current_price) / current_price

            # Record P&L based on signal
            if signal['signal'] == 'buy':
                returns.append(future_return)
            elif signal['signal'] == 'sell':
                returns.append(-future_return)  # Inverse for short
            else:
                returns.append(0)  # Hold = no P&L

        returns = np.array(returns)

        # Calculate metrics
        trades_mask = returns != 0
        if trades_mask.sum() > 0:
            trade_returns = returns[trades_mask]
            win_rate = (trade_returns > 0).sum() / len(trade_returns)
            avg_return = trade_returns.mean()
            sharpe = trade_returns.mean() / (trade_returns.std() + 1e-8) * np.sqrt(252 * 24)  # Assuming 1h bars
        else:
            win_rate = 0
            avg_return = 0
            sharpe = 0

        total_trades = trades_mask.sum()
        total_signals = len(signals)

        results = {
            'win_rate': win_rate,
            'avg_return': avg_return,
            'sharpe': sharpe,
            'total_trades': int(total_trades),
            'total_signals': total_signals,
            'signal_distribution': {
                'buy': (np.array(signals) == 'buy').sum(),
                'sell': (np.array(signals) == 'sell').sum(),
                'hold': (np.array(signals) == 'hold').sum()
            }
        }

        if verbose:
            logger.info("=" * 60)
            logger.info("MOMENTUM STRATEGY BACKTEST RESULTS")
            logger.info("=" * 60)
            logger.info(f"Win Rate: {win_rate*100:.2f}%")
            logger.info(f"Avg Return: {avg_return*100:.4f}%")
            logger.info(f"Sharpe Ratio: {sharpe:.2f}")
            logger.info(f"Total Trades: {total_trades}")
            logger.info(f"Total Signals: {total_signals}")
            logger.info(f"Signal Distribution:")
            logger.info(f"  BUY:  {results['signal_distribution']['buy']} ({results['signal_distribution']['buy']/total_signals*100:.1f}%)")
            logger.info(f"  SELL: {results['signal_distribution']['sell']} ({results['signal_distribution']['sell']/total_signals*100:.1f}%)")
            logger.info(f"  HOLD: {results['signal_distribution']['hold']} ({results['signal_distribution']['hold']/total_signals*100:.1f}%)")
            logger.info("=" * 60)

        return results
