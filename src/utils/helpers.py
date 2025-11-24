"""
Helper functions and utilities for AI Trading Bot.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Union
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


def calculate_returns(prices: np.ndarray, log_returns: bool = True) -> np.ndarray:
    """
    Calculate returns from price series.

    Args:
        prices: Array of prices
        log_returns: If True, calculate log returns; otherwise simple returns

    Returns:
        Array of returns
    """
    prices = np.asarray(prices, dtype=np.float64)

    if log_returns:
        returns = np.diff(np.log(prices))
    else:
        returns = np.diff(prices) / prices[:-1]

    return returns


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate the Sharpe ratio.

    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of trading periods per year

    Returns:
        Annualized Sharpe ratio
    """
    returns = np.asarray(returns, dtype=np.float64)

    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0

    excess_returns = returns - (risk_free_rate / periods_per_year)
    sharpe = np.mean(excess_returns) / np.std(excess_returns)

    return sharpe * np.sqrt(periods_per_year)


def calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate the Sortino ratio (only considers downside volatility).

    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of trading periods per year

    Returns:
        Annualized Sortino ratio
    """
    returns = np.asarray(returns, dtype=np.float64)

    if len(returns) == 0:
        return 0.0

    excess_returns = returns - (risk_free_rate / periods_per_year)
    downside_returns = returns[returns < 0]

    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return float('inf') if np.mean(excess_returns) > 0 else 0.0

    sortino = np.mean(excess_returns) / np.std(downside_returns)

    return sortino * np.sqrt(periods_per_year)


def calculate_max_drawdown(equity_curve: np.ndarray) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown and its duration.

    Args:
        equity_curve: Array of equity values over time

    Returns:
        Tuple of (max_drawdown, peak_idx, trough_idx)
    """
    equity_curve = np.asarray(equity_curve, dtype=np.float64)

    if len(equity_curve) == 0:
        return 0.0, 0, 0

    # Calculate running maximum
    running_max = np.maximum.accumulate(equity_curve)

    # Calculate drawdown at each point
    drawdowns = (equity_curve - running_max) / running_max

    # Find maximum drawdown
    max_dd_idx = np.argmin(drawdowns)
    max_dd = drawdowns[max_dd_idx]

    # Find peak before max drawdown
    peak_idx = np.argmax(equity_curve[:max_dd_idx + 1])

    return abs(max_dd), peak_idx, max_dd_idx


def calculate_calmar_ratio(
    returns: np.ndarray,
    periods_per_year: int = 252
) -> float:
    """
    Calculate the Calmar ratio (annual return / max drawdown).

    Args:
        returns: Array of returns
        periods_per_year: Number of trading periods per year

    Returns:
        Calmar ratio
    """
    returns = np.asarray(returns, dtype=np.float64)

    if len(returns) == 0:
        return 0.0

    # Calculate equity curve from returns
    equity_curve = np.cumprod(1 + returns)

    # Calculate annualized return
    total_return = equity_curve[-1] / equity_curve[0] - 1
    years = len(returns) / periods_per_year
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # Calculate max drawdown
    max_dd, _, _ = calculate_max_drawdown(equity_curve)

    if max_dd == 0:
        return float('inf') if annual_return > 0 else 0.0

    return annual_return / max_dd


def calculate_profit_factor(trades: List[dict]) -> float:
    """
    Calculate profit factor (gross profit / gross loss).

    Args:
        trades: List of trade dictionaries with 'pnl' key

    Returns:
        Profit factor
    """
    if not trades:
        return 0.0

    gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))

    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


def calculate_win_rate(trades: List[dict]) -> float:
    """
    Calculate win rate from list of trades.

    Args:
        trades: List of trade dictionaries with 'pnl' key

    Returns:
        Win rate as decimal
    """
    if not trades:
        return 0.0

    wins = sum(1 for t in trades if t['pnl'] > 0)
    return wins / len(trades)


def normalize_data(
    data: np.ndarray,
    method: str = "minmax",
    feature_range: Tuple[float, float] = (0, 1)
) -> Tuple[np.ndarray, object]:
    """
    Normalize data using specified method.

    Args:
        data: Data to normalize
        method: Normalization method ('minmax', 'standard', 'robust')
        feature_range: Range for minmax scaling

    Returns:
        Tuple of (normalized_data, scaler)
    """
    data = np.asarray(data, dtype=np.float64)

    # Reshape if 1D
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    if method == "minmax":
        scaler = MinMaxScaler(feature_range=feature_range)
    elif method == "standard":
        scaler = StandardScaler()
    elif method == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    normalized = scaler.fit_transform(data)

    return normalized, scaler


def create_sequences(
    data: np.ndarray,
    sequence_length: int,
    target_column: int = -1,
    prediction_horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time series prediction.

    Args:
        data: Input data array
        sequence_length: Length of each sequence
        target_column: Column index for target variable (-1 for last column)
        prediction_horizon: How many steps ahead to predict

    Returns:
        Tuple of (X sequences, y targets)
    """
    data = np.asarray(data, dtype=np.float64)

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    X, y = [], []
    num_samples = len(data) - sequence_length - prediction_horizon + 1

    for i in range(num_samples):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length + prediction_horizon - 1, target_column])

    return np.array(X), np.array(y)


def create_classification_labels(
    prices: np.ndarray,
    threshold: float = 0.0,
    num_classes: int = 3
) -> np.ndarray:
    """
    Create classification labels from price changes.

    Args:
        prices: Array of prices
        threshold: Threshold for neutral class
        num_classes: Number of classes (2 for up/down, 3 for up/neutral/down)

    Returns:
        Array of labels
    """
    returns = calculate_returns(prices, log_returns=False)

    if num_classes == 2:
        labels = (returns > 0).astype(int)
    else:
        labels = np.zeros(len(returns), dtype=int)
        labels[returns > threshold] = 2  # Up
        labels[returns < -threshold] = 0  # Down
        labels[(returns >= -threshold) & (returns <= threshold)] = 1  # Neutral

    return labels


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicator features to DataFrame.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with additional features
    """
    df = df.copy()

    # Price-based features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    # Moving averages
    for period in [5, 10, 20, 50, 200]:
        df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

    # Volatility
    df['volatility_20'] = df['returns'].rolling(window=20).std() * np.sqrt(252)

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']

    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * bb_std
    df['bb_lower'] = df['bb_middle'] - 2 * bb_std
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(window=14).mean()

    # Volume features
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']

    # Price position
    df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)

    return df


def calculate_position_size(
    account_balance: float,
    risk_per_trade: float,
    entry_price: float,
    stop_loss_price: float,
    leverage: float = 1.0
) -> float:
    """
    Calculate position size based on risk.

    Args:
        account_balance: Total account balance
        risk_per_trade: Fraction of account to risk (e.g., 0.02 for 2%)
        entry_price: Entry price for the trade
        stop_loss_price: Stop loss price
        leverage: Leverage multiplier

    Returns:
        Position size in base units
    """
    risk_amount = account_balance * risk_per_trade
    price_risk = abs(entry_price - stop_loss_price)

    if price_risk == 0:
        return 0.0

    position_size = (risk_amount / price_risk) * leverage

    return position_size


def calculate_kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Calculate optimal position size using Kelly Criterion.

    Args:
        win_rate: Probability of winning
        avg_win: Average winning trade amount
        avg_loss: Average losing trade amount (positive number)

    Returns:
        Optimal fraction of capital to risk
    """
    if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
        return 0.0

    win_loss_ratio = avg_win / avg_loss
    kelly = win_rate - ((1 - win_rate) / win_loss_ratio)

    # Often use fractional Kelly (e.g., half Kelly) for safety
    return max(0, kelly)


def resample_ohlcv(
    df: pd.DataFrame,
    timeframe: str
) -> pd.DataFrame:
    """
    Resample OHLCV data to different timeframe.

    Args:
        df: DataFrame with OHLCV data and datetime index
        timeframe: Target timeframe (e.g., '1H', '4H', '1D')

    Returns:
        Resampled DataFrame
    """
    ohlcv_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }

    return df.resample(timeframe).agg(ohlcv_dict).dropna()


def detect_trend(prices: np.ndarray, window: int = 20) -> str:
    """
    Detect current market trend.

    Args:
        prices: Array of prices
        window: Lookback window

    Returns:
        'uptrend', 'downtrend', or 'sideways'
    """
    if len(prices) < window:
        return 'sideways'

    prices = np.asarray(prices)
    recent_prices = prices[-window:]

    # Calculate linear regression slope
    x = np.arange(len(recent_prices))
    slope = np.polyfit(x, recent_prices, 1)[0]

    # Normalize slope by average price
    normalized_slope = slope / np.mean(recent_prices)

    if normalized_slope > 0.001:
        return 'uptrend'
    elif normalized_slope < -0.001:
        return 'downtrend'
    else:
        return 'sideways'


def identify_support_resistance(
    prices: np.ndarray,
    window: int = 20,
    num_levels: int = 3
) -> Tuple[List[float], List[float]]:
    """
    Identify support and resistance levels.

    Args:
        prices: Array of prices (highs and lows)
        window: Window for local extrema detection
        num_levels: Number of levels to return

    Returns:
        Tuple of (support_levels, resistance_levels)
    """
    prices = np.asarray(prices)

    # Find local minima (support)
    support_levels = []
    for i in range(window, len(prices) - window):
        if prices[i] == min(prices[i - window:i + window + 1]):
            support_levels.append(prices[i])

    # Find local maxima (resistance)
    resistance_levels = []
    for i in range(window, len(prices) - window):
        if prices[i] == max(prices[i - window:i + window + 1]):
            resistance_levels.append(prices[i])

    # Cluster and return most significant levels
    support_levels = sorted(set(support_levels))[-num_levels:]
    resistance_levels = sorted(set(resistance_levels))[:num_levels]

    return support_levels, resistance_levels
