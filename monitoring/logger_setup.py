"""
Logger Setup Module
====================
Configures logging for the trading bot.

Usage:
    from monitoring import setup_logging, get_logger

    # Setup logging (call once at startup)
    setup_logging()

    # Get logger for a module
    logger = get_logger(__name__)
    logger.info("This is a log message")
"""

import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from config.config_loader import Config


def setup_logging(
        config: Optional[Config] = None,
        log_dir: Optional[Path] = None,
        level: Optional[str] = None,
        console: bool = True
) -> None:
    """
    Setup logging configuration.

    Args:
        config: Configuration object (optional)
        log_dir: Log directory (optional, uses config if not specified)
        level: Log level (optional, uses config if not specified)
        console: Enable console output
    """
    # Get parameters from config or defaults
    if config is not None:
        log_dir = log_dir or config.get_logs_path()
        level = level or config.logging.level
        max_bytes = config.logging.max_bytes
        backup_count = config.logging.backup_count
        main_log = config.logging.main_log
        trade_log = config.logging.trade_log
        decision_log = config.logging.decision_log
    else:
        log_dir = log_dir or Path("logs")
        level = level or "INFO"
        max_bytes = 10 * 1024 * 1024  # 10 MB
        backup_count = 5
        main_log = "trading_bot.log"
        trade_log = "trades.log"
        decision_log = "decisions.log"

    # Ensure log directory exists
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Convert level string to constant
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    simple_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )

    trade_formatter = logging.Formatter(
        fmt='%(asctime)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Main log file handler (rotating)
    main_handler = RotatingFileHandler(
        log_dir / main_log,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    main_handler.setLevel(log_level)
    main_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(main_handler)

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(simple_formatter)
        root_logger.addHandler(console_handler)

    # Trade-specific logger
    trade_logger = logging.getLogger('trades')
    trade_logger.setLevel(logging.INFO)
    trade_logger.propagate = False

    trade_handler = RotatingFileHandler(
        log_dir / trade_log,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    trade_handler.setFormatter(trade_formatter)
    trade_logger.addHandler(trade_handler)

    # Decision-specific logger
    decision_logger = logging.getLogger('decisions')
    decision_logger.setLevel(logging.DEBUG)
    decision_logger.propagate = False

    decision_handler = RotatingFileHandler(
        log_dir / decision_log,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    decision_handler.setFormatter(detailed_formatter)
    decision_logger.addHandler(decision_handler)

    # Log startup
    root_logger.info("=" * 60)
    root_logger.info("Logging initialized")
    root_logger.info(f"Log directory: {log_dir}")
    root_logger.info(f"Log level: {level}")
    root_logger.info("=" * 60)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)


def get_trade_logger() -> logging.Logger:
    """Get the trade-specific logger."""
    return logging.getLogger('trades')


def get_decision_logger() -> logging.Logger:
    """Get the decision-specific logger."""
    return logging.getLogger('decisions')


def log_trade(
        action: str,
        symbol: str,
        direction: str,
        lot_size: float,
        price: float,
        sl: float = 0,
        tp: float = 0,
        pnl: float = 0,
        **kwargs
) -> None:
    """
    Log a trade action.

    Args:
        action: OPEN, CLOSE, MODIFY
        symbol: Trading symbol
        direction: BUY or SELL
        lot_size: Position size
        price: Execution price
        sl: Stop loss
        tp: Take profit
        pnl: Realized PnL (for closes)
        **kwargs: Additional fields
    """
    logger = get_trade_logger()

    parts = [
        f"{action}",
        f"{symbol}",
        f"{direction}",
        f"{lot_size:.2f}",
        f"@{price:.2f}",
    ]

    if sl > 0:
        parts.append(f"SL:{sl:.2f}")
    if tp > 0:
        parts.append(f"TP:{tp:.2f}")
    if action == "CLOSE":
        parts.append(f"PnL:${pnl:.2f}")

    for key, value in kwargs.items():
        parts.append(f"{key}:{value}")

    logger.info(" | ".join(parts))


def log_decision(
        bar_time: datetime,
        prediction: int,
        probability: float,
        signal: int,
        reason: str,
        **kwargs
) -> None:
    """
    Log a trading decision.

    Args:
        bar_time: Bar timestamp
        prediction: Model prediction
        probability: Prediction probability
        signal: Final signal
        reason: Decision reason
        **kwargs: Additional fields
    """
    logger = get_decision_logger()

    pred_str = {1: 'UP', 0: 'NEUTRAL', -1: 'DOWN'}.get(prediction, str(prediction))
    sig_str = {1: 'BUY', 0: 'NO_TRADE', -1: 'SELL'}.get(signal, str(signal))

    msg = f"Bar:{bar_time} | Pred:{pred_str} ({probability:.2%}) | Signal:{sig_str} | {reason}"

    for key, value in kwargs.items():
        msg += f" | {key}:{value}"

    logger.debug(msg)


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    from config import get_config

    config = get_config()
    setup_logging(config)

    logger = get_logger(__name__)

    print("Testing logging...")

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    # Test trade logging
    log_trade(
        action="OPEN",
        symbol="XAUUSD",
        direction="BUY",
        lot_size=0.1,
        price=2000.50,
        sl=1995.0,
        tp=2010.0
    )

    log_trade(
        action="CLOSE",
        symbol="XAUUSD",
        direction="BUY",
        lot_size=0.1,
        price=2008.00,
        pnl=75.0,
        reason="take_profit"
    )

    # Test decision logging
    log_decision(
        bar_time=datetime.now(),
        prediction=1,
        probability=0.65,
        signal=1,
        reason="All filters passed",
        atr=5.2
    )

    print("\nCheck the logs/ directory for log files")
