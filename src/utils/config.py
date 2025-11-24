"""
Configuration management for AI Trading Bot.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv


@dataclass
class TradingConfig:
    """Trading-specific configuration."""
    symbols: list = field(default_factory=lambda: ["BTC/USDT"])
    timeframes: list = field(default_factory=lambda: ["1h"])
    default_timeframe: str = "1h"
    max_open_positions: int = 5
    use_leverage: bool = False
    max_leverage: int = 3


@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_risk_per_trade: float = 0.02
    max_daily_drawdown: float = 0.05
    max_total_drawdown: float = 0.15
    risk_reward_ratio: float = 2.0
    trailing_stop: bool = True
    trailing_stop_pct: float = 0.01
    use_dynamic_position_sizing: bool = True


@dataclass
class NeuralNetworkConfig:
    """Neural network configuration."""
    lstm_sequence_length: int = 60
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 3
    lstm_dropout: float = 0.2
    lstm_learning_rate: float = 0.001
    gru_sequence_length: int = 60
    gru_hidden_size: int = 128
    gru_num_layers: int = 2
    cnn_image_size: tuple = (64, 64)
    cnn_num_filters: list = field(default_factory=lambda: [32, 64, 128])


@dataclass
class RLConfig:
    """Reinforcement learning configuration."""
    ppo_learning_rate: float = 0.0003
    ppo_n_steps: int = 2048
    ppo_batch_size: int = 64
    ppo_gamma: float = 0.99
    dql_learning_rate: float = 0.0001
    dql_buffer_size: int = 100000
    dql_epsilon_start: float = 1.0
    dql_epsilon_end: float = 0.01


@dataclass
class GeneticConfig:
    """Genetic algorithm configuration."""
    population_size: int = 100
    generations: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    tournament_size: int = 5
    elite_size: int = 10


class Config:
    """
    Central configuration manager for the AI Trading Bot.
    Loads settings from YAML files and environment variables.
    """

    _instance = None
    _initialized = False

    def __new__(cls, config_path: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_path: Optional[str] = None):
        if Config._initialized:
            return

        # Load environment variables
        load_dotenv()

        # Determine config path
        if config_path is None:
            config_path = os.environ.get(
                "CONFIG_PATH",
                str(Path(__file__).parent.parent.parent / "config" / "settings.yaml")
            )

        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}

        # Load configuration
        self._load_config()

        # Initialize sub-configs
        self.trading = self._create_trading_config()
        self.risk = self._create_risk_config()
        self.neural_network = self._create_nn_config()
        self.reinforcement_learning = self._create_rl_config()
        self.genetic = self._create_genetic_config()

        Config._initialized = True

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f) or {}
        else:
            self._config = {}

    def _create_trading_config(self) -> TradingConfig:
        """Create trading configuration."""
        trading = self._config.get("trading", {})
        return TradingConfig(
            symbols=trading.get("symbols", ["BTC/USDT"]),
            timeframes=trading.get("timeframes", ["1h"]),
            default_timeframe=trading.get("default_timeframe", "1h"),
            max_open_positions=trading.get("max_open_positions", 5),
            use_leverage=trading.get("use_leverage", False),
            max_leverage=trading.get("max_leverage", 3),
        )

    def _create_risk_config(self) -> RiskConfig:
        """Create risk management configuration."""
        risk = self._config.get("risk", {})
        return RiskConfig(
            max_risk_per_trade=risk.get("max_risk_per_trade", 0.02),
            max_daily_drawdown=risk.get("max_daily_drawdown", 0.05),
            max_total_drawdown=risk.get("max_total_drawdown", 0.15),
            risk_reward_ratio=risk.get("risk_reward_ratio", 2.0),
            trailing_stop=risk.get("trailing_stop", True),
            trailing_stop_pct=risk.get("trailing_stop_pct", 0.01),
            use_dynamic_position_sizing=risk.get("use_dynamic_position_sizing", True),
        )

    def _create_nn_config(self) -> NeuralNetworkConfig:
        """Create neural network configuration."""
        nn = self._config.get("neural_networks", {})
        lstm = nn.get("lstm", {})
        gru = nn.get("gru", {})
        cnn = nn.get("cnn", {})

        return NeuralNetworkConfig(
            lstm_sequence_length=lstm.get("sequence_length", 60),
            lstm_hidden_size=lstm.get("hidden_size", 128),
            lstm_num_layers=lstm.get("num_layers", 3),
            lstm_dropout=lstm.get("dropout", 0.2),
            lstm_learning_rate=lstm.get("learning_rate", 0.001),
            gru_sequence_length=gru.get("sequence_length", 60),
            gru_hidden_size=gru.get("hidden_size", 128),
            gru_num_layers=gru.get("num_layers", 2),
            cnn_image_size=tuple(cnn.get("image_size", [64, 64])),
            cnn_num_filters=cnn.get("num_filters", [32, 64, 128]),
        )

    def _create_rl_config(self) -> RLConfig:
        """Create reinforcement learning configuration."""
        rl = self._config.get("reinforcement_learning", {})
        ppo = rl.get("ppo", {})
        dql = rl.get("dql", {})

        return RLConfig(
            ppo_learning_rate=ppo.get("learning_rate", 0.0003),
            ppo_n_steps=ppo.get("n_steps", 2048),
            ppo_batch_size=ppo.get("batch_size", 64),
            ppo_gamma=ppo.get("gamma", 0.99),
            dql_learning_rate=dql.get("learning_rate", 0.0001),
            dql_buffer_size=dql.get("buffer_size", 100000),
            dql_epsilon_start=dql.get("epsilon_start", 1.0),
            dql_epsilon_end=dql.get("epsilon_end", 0.01),
        )

    def _create_genetic_config(self) -> GeneticConfig:
        """Create genetic algorithm configuration."""
        ga = self._config.get("genetic_algorithm", {})
        return GeneticConfig(
            population_size=ga.get("population_size", 100),
            generations=ga.get("generations", 50),
            mutation_rate=ga.get("mutation_rate", 0.1),
            crossover_rate=ga.get("crossover_rate", 0.8),
            tournament_size=ga.get("tournament_size", 5),
            elite_size=ga.get("elite_size", 10),
        )

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key path (e.g., 'trading.symbols')."""
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default

            if value is None:
                return default

        return value

    def get_env(self, key: str, default: Any = None) -> Any:
        """Get an environment variable."""
        return os.environ.get(key, default)

    @property
    def exchange_api_key(self) -> Optional[str]:
        """Get exchange API key from environment."""
        return os.environ.get("EXCHANGE_API_KEY")

    @property
    def exchange_secret(self) -> Optional[str]:
        """Get exchange secret from environment."""
        return os.environ.get("EXCHANGE_SECRET")

    @property
    def is_paper_trading(self) -> bool:
        """Check if running in paper trading mode."""
        return self.get("general.mode", "paper") == "paper"

    def reload(self) -> None:
        """Reload configuration from file."""
        self._load_config()
        self.trading = self._create_trading_config()
        self.risk = self._create_risk_config()
        self.neural_network = self._create_nn_config()
        self.reinforcement_learning = self._create_rl_config()
        self.genetic = self._create_genetic_config()

    def to_dict(self) -> Dict[str, Any]:
        """Return full configuration as dictionary."""
        return self._config.copy()

    @classmethod
    def reset(cls) -> None:
        """Reset singleton instance (useful for testing)."""
        cls._instance = None
        cls._initialized = False
