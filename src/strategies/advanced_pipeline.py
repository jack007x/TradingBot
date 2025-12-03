"""
Advanced Trading Pipeline v2.0
==============================

CRITICAL COMPONENTS:
1. ModelHealthMonitor - Detects stuck/degenerate predictions
2. TechnicalAnalysisEngine - Pure TA signal generation
3. TradeGuard - Anti-flip-flop, cooldown, trend alignment
4. AdvancedRiskCalculator - Swing-based SL, 1:3 RR enforcement
5. HybridSignalGenerator - Combines ML + TA signals
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


class SignalStrength(Enum):
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    NONE = "none"


class TrendDirection(Enum):
    UP = "up"
    DOWN = "down"
    SIDEWAYS = "sideways"


@dataclass
class TradeGuardState:
    """Tracks trading state for guard conditions"""
    last_trade_time: Optional[datetime] = None
    last_signal: Optional[str] = None
    signal_history: List[str] = field(default_factory=list)
    daily_trades: int = 0
    daily_pnl: float = 0.0
    consecutive_losses: int = 0
    last_daily_reset: Optional[datetime] = None

    def reset_daily(self):
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.last_daily_reset = datetime.utcnow()


@dataclass
class HybridSignal:
    """Complete trading signal with all metadata"""
    signal: str  # 'buy', 'sell', 'hold'
    confidence: float
    strength: SignalStrength

    # Component signals
    ml_signal: str
    ml_confidence: float
    ta_signal: str
    ta_confidence: float

    # Market context
    trend_direction: TrendDirection
    trend_strength: float
    regime: str

    # Risk parameters
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    risk_reward_ratio: float

    # Metadata
    confirmations: Dict[str, bool] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.utcnow)


class ModelHealthMonitor:
    """
    CRITICAL: Monitors ML model health and detects degenerate states.

    Detects when models produce stuck/constant predictions
    which was the main cause of previous trading failures.
    """

    def __init__(
        self,
        prediction_window: int = 20,
        variance_threshold: float = 1e-6,
        stuck_detection_count: int = 10
    ):
        self.prediction_window = prediction_window
        self.variance_threshold = variance_threshold
        self.stuck_detection_count = stuck_detection_count
        self.prediction_history: Dict[str, List[float]] = {}
        self.health_status: Dict[str, Dict] = {}

    def record_prediction(self, model_name: str, prediction: float):
        """Record a prediction for health monitoring"""
        if model_name not in self.prediction_history:
            self.prediction_history[model_name] = []
        self.prediction_history[model_name].append(prediction)
        if len(self.prediction_history[model_name]) > self.prediction_window:
            self.prediction_history[model_name] = self.prediction_history[model_name][-self.prediction_window:]

    def check_model_health(self, model_name: str) -> Dict:
        """
        Check if model is producing healthy predictions.

        Returns dict with:
        - healthy: bool
        - reason: str (if unhealthy)
        - variance: float
        - warnings: List[str]
        """
        if model_name not in self.prediction_history:
            return {'healthy': True, 'reason': 'No history yet'}

        preds = self.prediction_history[model_name]
        if len(preds) < self.stuck_detection_count:
            return {'healthy': True, 'reason': 'Insufficient history'}

        recent = preds[-self.stuck_detection_count:]
        variance = np.var(recent)
        mean = np.mean(recent)

        health = {
            'healthy': True,
            'variance': variance,
            'mean': mean,
            'recent_predictions': recent[-5:],
            'warnings': []
        }

        # Check for stuck predictions (near-zero variance)
        if variance < self.variance_threshold:
            health['healthy'] = False
            health['reason'] = f'STUCK: Variance {variance:.2e} < threshold'
            health['warnings'].append('Model producing constant predictions!')
            logger.warning(f"⚠️ {model_name} STUCK DETECTED: predictions = {mean:.6f} ± {np.sqrt(variance):.6f}")

        # Check for NaN/Inf
        if np.any(np.isnan(recent)) or np.any(np.isinf(recent)):
            health['healthy'] = False
            health['reason'] = 'NaN/Inf predictions detected'

        self.health_status[model_name] = health
        return health

    def get_healthy_models(self, model_names: List[str]) -> List[str]:
        """Return list of models that are currently healthy"""
        return [name for name in model_names if self.check_model_health(name)['healthy']]


class TechnicalAnalysisEngine:
    """
    Pure technical analysis signal generator.
    Used as secondary/fallback when ML models fail.

    Strategy: Trade with the trend on pullbacks
    - In uptrend: Buy on RSI oversold + price near support
    - In downtrend: Sell on RSI overbought + price near resistance
    """

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        df = df.copy()

        # Moving Averages
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()

        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))

        # ATR
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()

        # ADX
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        atr_14 = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / (atr_14 + 1e-10))
        minus_di = 100 * (minus_dm.rolling(14).mean() / (atr_14 + 1e-10))
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df['adx'] = dx.rolling(14).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * bb_std
        df['bb_lower'] = df['bb_middle'] - 2 * bb_std
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)

        # Stochastic
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14 + 1e-10)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()

        return df

    def get_trend_direction(self, df: pd.DataFrame) -> Tuple[TrendDirection, float]:
        """Determine current trend direction and strength (ADX)"""
        if len(df) < 50:
            return TrendDirection.SIDEWAYS, 0.0

        latest = df.iloc[-1]

        confirmations = {
            'price_above_sma20': latest['close'] > latest['sma_20'],
            'price_above_sma50': latest['close'] > latest['sma_50'],
            'sma20_above_sma50': latest['sma_20'] > latest['sma_50'],
            'macd_positive': latest['macd'] > 0,
            'macd_above_signal': latest['macd'] > latest['macd_signal'],
            'plus_di_above_minus': latest['plus_di'] > latest['minus_di'],
        }

        bullish_count = sum(confirmations.values())
        adx = latest['adx'] if not np.isnan(latest['adx']) else 20

        if adx < 20:
            return TrendDirection.SIDEWAYS, adx
        elif bullish_count >= 4:
            return TrendDirection.UP, adx
        elif bullish_count <= 2:
            return TrendDirection.DOWN, adx
        else:
            return TrendDirection.SIDEWAYS, adx

    def generate_signal(self, df: pd.DataFrame, trend: TrendDirection) -> Tuple[str, float, Dict]:
        """
        Generate trading signal from technical analysis.

        Returns: (signal, confidence, confirmations)
        """
        if len(df) < 50:
            return 'hold', 0.0, {}

        df = self.calculate_all_indicators(df)
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        confirmations = {}
        buy_score = 0
        sell_score = 0

        # BUY CONDITIONS
        rsi_oversold = latest['rsi'] < 40 and latest['rsi'] > prev['rsi']
        confirmations['rsi_oversold_turning'] = rsi_oversold
        if rsi_oversold: buy_score += 2

        bb_bounce = latest['bb_position'] < 0.3 and latest['close'] > prev['close']
        confirmations['bb_lower_bounce'] = bb_bounce
        if bb_bounce: buy_score += 1.5

        macd_turn_up = latest['macd_histogram'] > prev['macd_histogram']
        confirmations['macd_turning_up'] = macd_turn_up
        if macd_turn_up: buy_score += 1

        stoch_buy = latest['stoch_k'] < 30 and latest['stoch_k'] > latest['stoch_d']
        confirmations['stoch_oversold_cross'] = stoch_buy
        if stoch_buy: buy_score += 1.5

        # SELL CONDITIONS
        rsi_overbought = latest['rsi'] > 60 and latest['rsi'] < prev['rsi']
        confirmations['rsi_overbought_turning'] = rsi_overbought
        if rsi_overbought: sell_score += 2

        bb_reject = latest['bb_position'] > 0.7 and latest['close'] < prev['close']
        confirmations['bb_upper_reject'] = bb_reject
        if bb_reject: sell_score += 1.5

        macd_turn_down = latest['macd_histogram'] < prev['macd_histogram']
        confirmations['macd_turning_down'] = macd_turn_down
        if macd_turn_down: sell_score += 1

        stoch_sell = latest['stoch_k'] > 70 and latest['stoch_k'] < latest['stoch_d']
        confirmations['stoch_overbought_cross'] = stoch_sell
        if stoch_sell: sell_score += 1.5

        # TREND ALIGNMENT BONUS
        if trend == TrendDirection.UP:
            buy_score += 2
            sell_score *= 0.5
        elif trend == TrendDirection.DOWN:
            sell_score += 2
            buy_score *= 0.5

        # DETERMINE SIGNAL
        max_score = 10
        if buy_score > sell_score and buy_score >= 3:
            return 'buy', min(buy_score / max_score, 0.85), confirmations
        elif sell_score > buy_score and sell_score >= 3:
            return 'sell', min(sell_score / max_score, 0.85), confirmations
        else:
            return 'hold', 0.0, confirmations


class TradeGuard:
    """
    CRITICAL: Trade filtering and guard system to prevent bad trades.

    Guards:
    1. Anti-flip-flop: Prevents rapid BUY->SELL->BUY sequences
    2. Cooldown: Minimum time between trades
    3. Trend alignment: Only trade with the trend
    4. Loss limits: Stop trading after consecutive losses
    5. Signal consistency: Require consistent signals over time
    """

    def __init__(
        self,
        cooldown_minutes: int = 30,
        max_daily_trades: int = 5,
        max_consecutive_losses: int = 3,
        signal_consistency_required: int = 3,
        signal_history_size: int = 5
    ):
        self.cooldown_minutes = cooldown_minutes
        self.max_daily_trades = max_daily_trades
        self.max_consecutive_losses = max_consecutive_losses
        self.signal_consistency_required = signal_consistency_required
        self.signal_history_size = signal_history_size
        self.state = TradeGuardState()

    def check_daily_reset(self):
        now = datetime.utcnow()
        if self.state.last_daily_reset is None or now.date() > self.state.last_daily_reset.date():
            logger.info("📅 New trading day - resetting daily counters")
            self.state.reset_daily()

    def record_signal(self, signal: str):
        self.state.signal_history.append(signal)
        if len(self.state.signal_history) > self.signal_history_size:
            self.state.signal_history = self.state.signal_history[-self.signal_history_size:]

    def record_trade(self, signal: str, pnl: float = 0.0):
        self.state.last_trade_time = datetime.utcnow()
        self.state.last_signal = signal
        self.state.daily_trades += 1
        self.state.daily_pnl += pnl
        self.state.consecutive_losses = self.state.consecutive_losses + 1 if pnl < 0 else 0

    def can_trade(self, signal: str, trend: TrendDirection, ml_healthy: bool = True) -> Tuple[bool, List[str]]:
        """
        Check if trade is allowed based on all guard conditions.

        Returns: (allowed, list of rejection reasons)
        """
        self.check_daily_reset()
        reasons = []

        # Guard 1: Cooldown
        if self.state.last_trade_time:
            elapsed = (datetime.utcnow() - self.state.last_trade_time).total_seconds()
            cooldown_seconds = self.cooldown_minutes * 60
            if elapsed < cooldown_seconds:
                remaining = int((cooldown_seconds - elapsed) / 60)
                reasons.append(f"Cooldown: {remaining}min remaining")

        # Guard 2: Daily limit
        if self.state.daily_trades >= self.max_daily_trades:
            reasons.append(f"Daily limit: {self.state.daily_trades}/{self.max_daily_trades}")

        # Guard 3: Consecutive losses
        if self.state.consecutive_losses >= self.max_consecutive_losses:
            reasons.append(f"Loss streak: {self.state.consecutive_losses} losses")

        # Guard 4: Anti-flip-flop
        if self.state.last_signal and signal != self.state.last_signal and signal != 'hold':
            if self.state.last_trade_time:
                elapsed = (datetime.utcnow() - self.state.last_trade_time).total_seconds()
                if elapsed < 3600:
                    reasons.append(f"Flip-flop: {self.state.last_signal}→{signal} within {int(elapsed/60)}min")

        # Guard 5: Trend alignment
        if signal == 'buy' and trend == TrendDirection.DOWN:
            reasons.append("Counter-trend: BUY in DOWNTREND")
        elif signal == 'sell' and trend == TrendDirection.UP:
            reasons.append("Counter-trend: SELL in UPTREND")

        # Guard 6: Signal consistency
        if len(self.state.signal_history) >= self.signal_consistency_required:
            recent = self.state.signal_history[-self.signal_consistency_required:]
            if sum(1 for s in recent if s == signal) < self.signal_consistency_required:
                reasons.append(f"Inconsistent signal")

        return len(reasons) == 0, reasons


class AdvancedRiskCalculator:
    """
    Advanced risk calculation with swing-based stops.

    CRITICAL: Enforces 1:3 Risk-Reward minimum
    """

    def __init__(
        self,
        min_rr_ratio: float = 3.0,
        risk_per_trade: float = 0.01,
        min_sl_atr: float = 1.5,
        max_sl_atr: float = 3.0
    ):
        self.min_rr_ratio = min_rr_ratio
        self.risk_per_trade = risk_per_trade
        self.min_sl_atr = min_sl_atr
        self.max_sl_atr = max_sl_atr

    def calculate_swing_stop(self, df: pd.DataFrame, signal: str, atr: float, lookback: int = 10) -> float:
        """
        Calculate stop loss based on recent swing points.

        BUY: SL below recent swing low - 0.5*ATR
        SELL: SL above recent swing high + 0.5*ATR
        """
        recent = df.iloc[-lookback:]
        current_price = df['close'].iloc[-1]

        if signal == 'buy':
            swing_low = recent['low'].min()
            sl_price = swing_low - (0.5 * atr)
            sl_distance = current_price - sl_price
        else:
            swing_high = recent['high'].max()
            sl_price = swing_high + (0.5 * atr)
            sl_distance = sl_price - current_price

        # Enforce min/max ATR
        sl_distance = max(self.min_sl_atr * atr, min(sl_distance, self.max_sl_atr * atr))

        return current_price - sl_distance if signal == 'buy' else current_price + sl_distance

    def calculate_take_profit(self, entry: float, stop_loss: float, signal: str) -> float:
        """Calculate TP for minimum 1:3 RR"""
        sl_distance = abs(entry - stop_loss)
        tp_distance = sl_distance * self.min_rr_ratio
        return entry + tp_distance if signal == 'buy' else entry - tp_distance

    def calculate_position_size(self, balance: float, entry: float, stop_loss: float) -> float:
        """Calculate position size for 1% risk"""
        risk_amount = balance * self.risk_per_trade
        sl_distance = abs(entry - stop_loss)
        sl_pips = sl_distance / 0.01  # Gold pip = 0.01
        position_size = risk_amount / sl_pips if sl_pips > 0 else 0.01
        return max(0.01, min(round(position_size, 2), 1.0))


class HybridSignalGenerator:
    """
    Main signal generator combining ML predictions with Technical Analysis.

    Pipeline:
    1. Get ML predictions (with health check)
    2. Get TA signals
    3. Detect market regime
    4. Apply trade guards
    5. Calculate risk parameters
    6. Generate final hybrid signal
    """

    def __init__(
        self,
        ml_weight: float = 0.6,
        ta_weight: float = 0.4,
        min_confidence: float = 0.55
    ):
        self.ml_weight = ml_weight
        self.ta_weight = ta_weight
        self.min_confidence = min_confidence

        self.health_monitor = ModelHealthMonitor()
        self.ta_engine = TechnicalAnalysisEngine()
        self.trade_guard = TradeGuard()
        self.risk_calc = AdvancedRiskCalculator()

    def generate_signal(
        self,
        df: pd.DataFrame,
        ml_predictions: Dict[str, Tuple[str, float]],
        balance: float,
        current_price: float
    ) -> HybridSignal:
        """Generate comprehensive trading signal."""
        warnings_list = []
        reasons = []

        # Step 1: Calculate indicators and trend
        df = self.ta_engine.calculate_all_indicators(df)
        trend_direction, trend_strength = self.ta_engine.get_trend_direction(df)

        # Determine regime
        adx = df['adx'].iloc[-1] if not np.isnan(df['adx'].iloc[-1]) else 20
        regime = 'trending' if adx > 25 else 'ranging' if adx < 15 else 'neutral'

        # Step 2: Check ML health
        healthy_models = []
        for model_name, (pred_signal, pred_conf) in ml_predictions.items():
            pred_value = pred_conf if pred_signal == 'buy' else -pred_conf if pred_signal == 'sell' else 0
            self.health_monitor.record_prediction(model_name, pred_value)

            health = self.health_monitor.check_model_health(model_name)
            if health['healthy']:
                healthy_models.append(model_name)
            else:
                warnings_list.append(f"{model_name}: {health.get('reason', 'Unhealthy')}")

        ml_healthy = len(healthy_models) > 0

        # Step 3: Get ML signal (healthy models only)
        if ml_healthy:
            buy_votes = sum(conf for name, (sig, conf) in ml_predictions.items()
                          if name in healthy_models and sig == 'buy')
            sell_votes = sum(conf for name, (sig, conf) in ml_predictions.items()
                           if name in healthy_models and sig == 'sell')
            total = buy_votes + sell_votes + 1e-10

            if buy_votes > sell_votes:
                ml_signal, ml_confidence = 'buy', buy_votes / total
            elif sell_votes > buy_votes:
                ml_signal, ml_confidence = 'sell', sell_votes / total
            else:
                ml_signal, ml_confidence = 'hold', 0.0
        else:
            ml_signal, ml_confidence = 'hold', 0.0
            warnings_list.append("All ML models unhealthy - TA-only mode")

        # Step 4: Get TA signal
        ta_signal, ta_confidence, ta_confirmations = self.ta_engine.generate_signal(df, trend_direction)

        # Step 5: Combine signals
        eff_ml_weight = 0.0 if not ml_healthy else self.ml_weight
        eff_ta_weight = 1.0 if not ml_healthy else self.ta_weight

        buy_score = (ml_confidence * eff_ml_weight if ml_signal == 'buy' else 0) + \
                    (ta_confidence * eff_ta_weight if ta_signal == 'buy' else 0)
        sell_score = (ml_confidence * eff_ml_weight if ml_signal == 'sell' else 0) + \
                     (ta_confidence * eff_ta_weight if ta_signal == 'sell' else 0)

        if buy_score > sell_score and buy_score > self.min_confidence:
            final_signal, final_confidence = 'buy', buy_score
        elif sell_score > buy_score and sell_score > self.min_confidence:
            final_signal, final_confidence = 'sell', sell_score
        else:
            final_signal, final_confidence = 'hold', max(buy_score, sell_score)

        # Step 6: Apply trade guards
        self.trade_guard.record_signal(final_signal)

        if final_signal != 'hold':
            can_trade, guard_reasons = self.trade_guard.can_trade(final_signal, trend_direction, ml_healthy)
            if not can_trade:
                reasons.extend(guard_reasons)
                final_signal, final_confidence = 'hold', 0.0

        # Step 7: Calculate risk parameters
        atr = df['atr'].iloc[-1] if not np.isnan(df['atr'].iloc[-1]) else current_price * 0.002

        if final_signal != 'hold':
            stop_loss = self.risk_calc.calculate_swing_stop(df, final_signal, atr)
            take_profit = self.risk_calc.calculate_take_profit(current_price, stop_loss, final_signal)
            position_size = self.risk_calc.calculate_position_size(balance, current_price, stop_loss)
            rr_ratio = abs(take_profit - current_price) / abs(current_price - stop_loss) if abs(current_price - stop_loss) > 0 else 0
        else:
            stop_loss = take_profit = current_price
            position_size = rr_ratio = 0.0

        # Determine strength
        if final_confidence >= 0.7 and len(warnings_list) == 0:
            strength = SignalStrength.STRONG
        elif final_confidence >= 0.55:
            strength = SignalStrength.MODERATE
        elif final_signal != 'hold':
            strength = SignalStrength.WEAK
        else:
            strength = SignalStrength.NONE

        return HybridSignal(
            signal=final_signal,
            confidence=final_confidence,
            strength=strength,
            ml_signal=ml_signal,
            ml_confidence=ml_confidence,
            ta_signal=ta_signal,
            ta_confidence=ta_confidence,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            regime=regime,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            risk_reward_ratio=rr_ratio,
            confirmations=ta_confirmations,
            reasons=reasons,
            warnings=warnings_list
        )

    def log_signal(self, signal: HybridSignal):
        """Log signal details"""
        logger.info("=" * 80)
        logger.info("🔮 HYBRID SIGNAL GENERATED")
        logger.info("=" * 80)
        logger.info(f"   Signal: {signal.signal.upper()} | Confidence: {signal.confidence:.2%}")
        logger.info(f"   Strength: {signal.strength.value}")
        logger.info(f"   ML: {signal.ml_signal} ({signal.ml_confidence:.2%})")
        logger.info(f"   TA: {signal.ta_signal} ({signal.ta_confidence:.2%})")
        logger.info(f"   Trend: {signal.trend_direction.value.upper()} (ADX: {signal.trend_strength:.1f})")
        if signal.signal != 'hold':
            logger.info(f"   Entry: {signal.entry_price:.2f} | SL: {signal.stop_loss:.2f} | TP: {signal.take_profit:.2f}")
            logger.info(f"   R:R: 1:{signal.risk_reward_ratio:.1f} | Size: {signal.position_size:.2f}")
        if signal.warnings:
            logger.warning(f"   ⚠️ Warnings: {', '.join(signal.warnings)}")
        if signal.reasons:
            logger.info(f"   📝 Blocked: {', '.join(signal.reasons)}")
        logger.info("=" * 80)
