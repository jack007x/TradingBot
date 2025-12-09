"""
Configuration Loader Module
============================
Loads and validates configuration from YAML file.
Supports environment variable substitution for sensitive values.

Usage:
    from config import get_config
    config = get_config()
    print(config.trading.symbol)
"""

import os
import re
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class BrokerConfig:
    terminal_path: str
    login: Optional[str]
    password: Optional[str]
    server: Optional[str]
    timeout_ms: int = 10000
    retry_attempts: int = 3
    retry_delay_seconds: int = 5


@dataclass
class TradingConfig:
    symbol: str
    timeframe: str
    allowed_sessions: Dict[str, List[int]]
    blocked_hours_utc: List[int]
    magic_number: int


@dataclass
class RiskConfig:
    risk_per_trade_percent: float
    max_open_trades: int
    max_lot_size: float
    min_lot_size: float
    max_daily_loss_percent: float
    max_weekly_loss_percent: float
    max_drawdown_percent: float
    max_consecutive_losses: int
    cooldown_after_losses_minutes: int
    sl_atr_multiplier: float
    tp_atr_multiplier: float
    max_sl_pips: float
    min_sl_pips: float


@dataclass
class ModelConfig:
    type: str
    prediction_horizon_bars: int
    return_threshold_percent: float
    min_probability_threshold: float
    lgbm_params: Dict[str, Any]


@dataclass
class FeatureConfig:
    lookback_window: int
    price_features: List[str]
    technical_indicators: Dict[str, Any]
    time_features: List[str]
    volatility_features: Dict[str, Any]


@dataclass
class TrainingConfig:
    training_window_days: int
    validation_split: float
    cv_splits: int
    gap_bars: int
    early_stopping_rounds: int
    retrain_frequency_days: int
    retrain_hour_utc: int


@dataclass
class RetrainingConfig:
    min_sharpe_ratio: float
    max_allowed_drawdown: float
    min_win_rate: float
    min_profit_factor: float
    sharpe_tolerance: float
    drawdown_tolerance: float
    win_rate_tolerance: float
    fallback_trigger_dd_percent: float
    fallback_lookback_days: int


@dataclass
class BacktestConfig:
    initial_balance: float
    spread_pips: float
    commission_per_lot: float
    slippage_pips: float
    save_trade_log: bool
    save_equity_curve: bool
    generate_report: bool


@dataclass
class StorageConfig:
    format: str
    data_cache_dir: str
    saved_models_dir: str
    logs_dir: str
    reports_dir: str
    historical_data_file: str
    use_sqlite: bool
    sqlite_db: str


@dataclass
class LoggingConfig:
    level: str
    main_log: str
    trade_log: str
    decision_log: str
    max_bytes: int
    backup_count: int
    console_output: bool


@dataclass
class MonitoringConfig:
    track_daily_pnl: bool
    track_trade_history: bool
    alert_on_daily_loss_percent: float
    alert_on_consecutive_losses: int
    daily_report: bool
    weekly_report: bool


@dataclass
class Config:
    """Main configuration container."""
    broker: BrokerConfig
    trading: TradingConfig
    risk: RiskConfig
    model: ModelConfig
    features: FeatureConfig
    training: TrainingConfig
    retraining: RetrainingConfig
    backtesting: BacktestConfig
    storage: StorageConfig
    logging: LoggingConfig
    monitoring: MonitoringConfig

    # Project root path
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)

    def get_data_cache_path(self) -> Path:
        """Get absolute path to data cache directory."""
        return self.project_root / self.storage.data_cache_dir

    def get_models_path(self) -> Path:
        """Get absolute path to saved models directory."""
        return self.project_root / self.storage.saved_models_dir

    def get_logs_path(self) -> Path:
        """Get absolute path to logs directory."""
        return self.project_root / self.storage.logs_dir

    def get_reports_path(self) -> Path:
        """Get absolute path to reports directory."""
        path = self.project_root / self.storage.reports_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_historical_data_path(self) -> Path:
        """Get absolute path to historical data file."""
        return self.get_data_cache_path() / self.storage.historical_data_file

    def get_sqlite_path(self) -> Path:
        """Get absolute path to SQLite database."""
        return self.project_root / self.storage.sqlite_db


# =============================================================================
# Configuration Loader Class
# =============================================================================

class ConfigLoader:
    """Loads and validates configuration from YAML file."""

    # Environment variable pattern: ${VAR_NAME}
    ENV_VAR_PATTERN = re.compile(r'\$\{([^}]+)\}')

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config loader.

        Args:
            config_path: Path to config.yaml. If None, uses default location.
        """
        if config_path is None:
            self.config_path = Path(__file__).parent / "config.yaml"
        else:
            self.config_path = Path(config_path)

        self._raw_config: Dict = {}
        self._config: Optional[Config] = None

    def _substitute_env_vars(self, value: Any) -> Any:
        """
        Recursively substitute environment variables in config values.

        Supports ${VAR_NAME} syntax. Returns None if env var not set.
        """
        if isinstance(value, str):
            match = self.ENV_VAR_PATTERN.match(value)
            if match:
                env_var = match.group(1)
                return os.environ.get(env_var)
            return value
        elif isinstance(value, dict):
            return {k: self._substitute_env_vars(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._substitute_env_vars(v) for v in value]
        return value

    def load(self) -> Config:
        """
        Load and parse configuration file.

        Returns:
            Config: Parsed configuration object

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If required config keys are missing
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._raw_config = yaml.safe_load(f)

        # Substitute environment variables
        self._raw_config = self._substitute_env_vars(self._raw_config)

        # Parse into dataclasses
        self._config = self._parse_config()

        # Validate configuration
        self._validate()

        # Ensure directories exist
        self._ensure_directories()

        return self._config

    def _parse_config(self) -> Config:
        """Parse raw YAML dict into Config dataclass."""
        return Config(
            broker=BrokerConfig(**self._raw_config['broker']),
            trading=TradingConfig(**self._raw_config['trading']),
            risk=RiskConfig(**self._raw_config['risk']),
            model=ModelConfig(**self._raw_config['model']),
            features=FeatureConfig(**self._raw_config['features']),
            training=TrainingConfig(**self._raw_config['training']),
            retraining=RetrainingConfig(**self._raw_config['retraining']),
            backtesting=BacktestConfig(**self._raw_config['backtesting']),
            storage=StorageConfig(**self._raw_config['storage']),
            logging=LoggingConfig(**self._raw_config['logging']),
            monitoring=MonitoringConfig(**self._raw_config['monitoring']),
        )

    def _validate(self) -> None:
        """
        Validate configuration values.

        Raises:
            ValueError: If validation fails
        """
        errors = []

        # Validate risk parameters
        if self._config.risk.risk_per_trade_percent <= 0:
            errors.append("risk_per_trade_percent must be > 0")
        if self._config.risk.risk_per_trade_percent > 10:
            errors.append("risk_per_trade_percent > 10% is extremely risky")

        if self._config.risk.max_open_trades < 1:
            errors.append("max_open_trades must be >= 1")

        if self._config.risk.sl_atr_multiplier <= 0:
            errors.append("sl_atr_multiplier must be > 0")

        # Validate model parameters
        if self._config.model.prediction_horizon_bars < 1:
            errors.append("prediction_horizon_bars must be >= 1")

        if not 0 < self._config.model.min_probability_threshold < 1:
            errors.append("min_probability_threshold must be between 0 and 1")

        # Validate training parameters
        if not 0 < self._config.training.validation_split < 1:
            errors.append("validation_split must be between 0 and 1")

        if self._config.training.cv_splits < 2:
            errors.append("cv_splits must be >= 2")

        # Validate timeframe
        valid_timeframes = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1']
        if self._config.trading.timeframe not in valid_timeframes:
            errors.append(f"timeframe must be one of {valid_timeframes}")

        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    def _ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        dirs_to_create = [
            self._config.get_data_cache_path(),
            self._config.get_models_path(),
            self._config.get_logs_path(),
        ]

        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)

    def get_raw(self) -> Dict:
        """Get raw configuration dictionary."""
        return self._raw_config


# =============================================================================
# Singleton Config Instance
# =============================================================================

_config_instance: Optional[Config] = None


def get_config(config_path: Optional[str] = None, reload: bool = False) -> Config:
    """
    Get configuration singleton.

    Args:
        config_path: Optional path to config file
        reload: Force reload configuration

    Returns:
        Config: Configuration object
    """
    global _config_instance

    if _config_instance is None or reload:
        loader = ConfigLoader(config_path)
        _config_instance = loader.load()

    return _config_instance


def get_timeframe_minutes(timeframe: str) -> int:
    """
    Convert timeframe string to minutes.

    Args:
        timeframe: Timeframe string (M1, M5, M15, M30, H1, H4, D1)

    Returns:
        int: Number of minutes
    """
    mapping = {
        'M1': 1,
        'M5': 5,
        'M15': 15,
        'M30': 30,
        'H1': 60,
        'H4': 240,
        'D1': 1440,
    }
    return mapping.get(timeframe, 15)


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    # Test configuration loading
    try:
        config = get_config()
        print("=" * 60)
        print("Configuration loaded successfully!")
        print("=" * 60)
        print(f"Symbol: {config.trading.symbol}")
        print(f"Timeframe: {config.trading.timeframe}")
        print(f"Risk per trade: {config.risk.risk_per_trade_percent}%")
        print(f"Max open trades: {config.risk.max_open_trades}")
        print(f"Prediction horizon: {config.model.prediction_horizon_bars} bars")
        print(f"Training window: {config.training.training_window_days} days")
        print(f"Data cache path: {config.get_data_cache_path()}")
        print(f"Models path: {config.get_models_path()}")
        print("=" * 60)
    except Exception as e:
        print(f"Error loading config: {e}")
