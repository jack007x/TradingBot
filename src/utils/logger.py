"""
Logging configuration for AI Trading Bot.
"""

import sys
from pathlib import Path
from typing import Optional
from loguru import logger


def setup_logger(
    log_file: Optional[str] = None,
    level: str = "INFO",
    rotation: str = "100 MB",
    retention: str = "30 days",
    format_string: Optional[str] = None
) -> logger:
    """
    Configure and return the logger instance.

    Args:
        log_file: Path to log file (optional)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        rotation: When to rotate log files
        retention: How long to keep old log files
        format_string: Custom format string

    Returns:
        Configured logger instance
    """
    # Remove default handler
    logger.remove()

    # Default format
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

    # Add console handler with colors
    logger.add(
        sys.stdout,
        format=format_string,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_file,
            format=format_string.replace("<green>", "").replace("</green>", "")
                  .replace("<level>", "").replace("</level>", "")
                  .replace("<cyan>", "").replace("</cyan>", ""),
            level=level,
            rotation=rotation,
            retention=retention,
            compression="zip",
            backtrace=True,
            diagnose=True,
        )

    logger.info(f"Logger initialized with level: {level}")
    return logger


class TradingLogger:
    """
    Specialized logger for trading operations.
    Provides structured logging for trades, signals, and performance.
    """

    def __init__(self, name: str = "trading"):
        self.name = name
        self._logger = logger.bind(component=name)

    def trade_opened(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        size: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> None:
        """Log trade opening."""
        self._logger.info(
            f"TRADE OPENED | {symbol} | {side.upper()} | "
            f"Entry: {entry_price:.6f} | Size: {size:.4f} | "
            f"SL: {stop_loss or 'N/A'} | TP: {take_profit or 'N/A'}"
        )

    def trade_closed(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        reason: str = "manual"
    ) -> None:
        """Log trade closing."""
        emoji = "+" if pnl >= 0 else ""
        self._logger.info(
            f"TRADE CLOSED | {symbol} | {side.upper()} | "
            f"Entry: {entry_price:.6f} | Exit: {exit_price:.6f} | "
            f"PnL: {emoji}{pnl:.2f} ({emoji}{pnl_pct:.2f}%) | "
            f"Reason: {reason}"
        )

    def signal_generated(
        self,
        symbol: str,
        signal: str,
        confidence: float,
        source: str
    ) -> None:
        """Log trading signal generation."""
        self._logger.info(
            f"SIGNAL | {symbol} | {signal.upper()} | "
            f"Confidence: {confidence:.2%} | Source: {source}"
        )

    def model_prediction(
        self,
        model_name: str,
        symbol: str,
        prediction: str,
        probability: float
    ) -> None:
        """Log model prediction."""
        self._logger.debug(
            f"PREDICTION | {model_name} | {symbol} | "
            f"{prediction} | Prob: {probability:.4f}"
        )

    def risk_alert(
        self,
        alert_type: str,
        message: str,
        current_value: float,
        threshold: float
    ) -> None:
        """Log risk management alerts."""
        self._logger.warning(
            f"RISK ALERT | {alert_type} | {message} | "
            f"Current: {current_value:.4f} | Threshold: {threshold:.4f}"
        )

    def performance_update(
        self,
        total_trades: int,
        win_rate: float,
        profit_factor: float,
        total_pnl: float
    ) -> None:
        """Log performance metrics."""
        self._logger.info(
            f"PERFORMANCE | Trades: {total_trades} | "
            f"Win Rate: {win_rate:.2%} | PF: {profit_factor:.2f} | "
            f"Total PnL: {total_pnl:.2f}"
        )

    def model_training(
        self,
        model_name: str,
        epoch: int,
        loss: float,
        metrics: dict
    ) -> None:
        """Log model training progress."""
        metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        self._logger.debug(
            f"TRAINING | {model_name} | Epoch: {epoch} | "
            f"Loss: {loss:.6f} | {metrics_str}"
        )

    def self_learning_update(
        self,
        component: str,
        old_params: dict,
        new_params: dict,
        improvement: float
    ) -> None:
        """Log self-learning parameter updates."""
        self._logger.info(
            f"SELF-LEARNING | {component} | "
            f"Improvement: {improvement:.2%} | "
            f"Updated parameters: {list(new_params.keys())}"
        )

    def error(self, message: str, exc_info: bool = True) -> None:
        """Log error with optional exception info."""
        self._logger.error(message, exc_info=exc_info)

    def debug(self, message: str) -> None:
        """Log debug message."""
        self._logger.debug(message)

    def info(self, message: str) -> None:
        """Log info message."""
        self._logger.info(message)

    def warning(self, message: str) -> None:
        """Log warning message."""
        self._logger.warning(message)
