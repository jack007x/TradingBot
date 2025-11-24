"""
Main AI Trading Bot - Self-Learning Trading System
===================================================

A cutting-edge artificial intelligence trading system featuring:
- Deep Learning (LSTM, GRU, CNN)
- Reinforcement Learning (PPO, DQL)
- Natural Language Processing for sentiment analysis
- Genetic Algorithms for optimization
- Explainable AI for transparency
- Self-learning capabilities
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import numpy as np
from loguru import logger

from .utils.config import Config
from .utils.logger import setup_logger, TradingLogger
from .data.data_manager import DataManager
from .data.data_preprocessor import DataPreprocessor
from .models.neural_networks.lstm_model import LSTMPredictor
from .models.neural_networks.gru_model import GRUPredictor
from .models.neural_networks.cnn_model import CNNPatternRecognizer
from .models.reinforcement_learning.ppo_agent import PPOTradingAgent
from .models.reinforcement_learning.dql_agent import DQLTradingAgent
from .models.reinforcement_learning.trading_env import TradingEnvironment
from .models.nlp.sentiment_analyzer import SentimentAnalyzer
from .models.genetic.optimizer import GeneticOptimizer, ParameterRange
from .models.xai.explainer import ModelExplainer
from .risk_management.risk_manager import RiskManager
from .risk_management.position_sizer import PositionSizer
from .strategies.self_learning_engine import SelfLearningEngine
from .strategies.ensemble_strategy import EnsembleStrategy


class AITradingBot:
    """
    Main orchestrator for the AI Trading Bot system.
    Coordinates all components for autonomous trading.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        mode: str = 'paper'
    ):
        """
        Initialize AI Trading Bot.

        Args:
            config_path: Path to configuration file
            mode: Trading mode ('paper', 'live', 'backtest')
        """
        # Load configuration
        self.config = Config(config_path)
        self.mode = mode

        # Setup logging
        setup_logger(
            log_file=self.config.get('logging.file', 'logs/trading_bot.log'),
            level=self.config.get('general.log_level', 'INFO')
        )
        self.logger = TradingLogger('main')

        # Initialize components
        self.data_manager: Optional[DataManager] = None
        self.preprocessor = DataPreprocessor()

        # AI Models
        self.lstm_model: Optional[LSTMPredictor] = None
        self.gru_model: Optional[GRUPredictor] = None
        self.cnn_model: Optional[CNNPatternRecognizer] = None
        self.ppo_agent: Optional[PPOTradingAgent] = None
        self.dql_agent: Optional[DQLTradingAgent] = None
        self.sentiment_analyzer: Optional[SentimentAnalyzer] = None

        # Optimization and Explainability
        self.genetic_optimizer: Optional[GeneticOptimizer] = None
        self.model_explainer: Optional[ModelExplainer] = None

        # Risk Management
        self.risk_manager: Optional[RiskManager] = None
        self.position_sizer: Optional[PositionSizer] = None

        # Strategy Components
        self.self_learning_engine: Optional[SelfLearningEngine] = None
        self.ensemble_strategy: Optional[EnsembleStrategy] = None

        # State
        self.is_running = False
        self.models_trained = False
        self.feature_columns: List[str] = []

        logger.info(f"AITradingBot initialized in {mode} mode")

    async def initialize(self) -> None:
        """Initialize all components."""
        logger.info("Initializing trading bot components...")

        # Initialize data manager
        self.data_manager = DataManager(
            exchange_id=self.config.get('exchange.name', 'binance'),
            api_key=self.config.exchange_api_key,
            secret=self.config.exchange_secret,
            testnet=self.config.get('exchange.testnet', True)
        )

        # Initialize risk management
        initial_balance = self.config.get('trading.initial_balance', 10000)
        self.risk_manager = RiskManager(
            initial_balance=initial_balance,
            max_risk_per_trade=self.config.risk.max_risk_per_trade,
            max_daily_drawdown=self.config.risk.max_daily_drawdown,
            max_total_drawdown=self.config.risk.max_total_drawdown
        )

        self.position_sizer = PositionSizer(
            method='dynamic',
            base_risk=self.config.risk.max_risk_per_trade
        )

        # Initialize self-learning engine
        self.self_learning_engine = SelfLearningEngine(
            learning_rate=0.1,
            adaptation_threshold=self.config.get('self_learning.strategy_update_threshold', 0.1),
            min_trades_for_update=self.config.get('self_learning.min_trades_for_update', 10)
        )

        # Initialize sentiment analyzer
        self.sentiment_analyzer = SentimentAnalyzer(
            model_name=self.config.get('nlp.model_name', 'ProsusAI/finbert')
        )

        logger.info("Bot components initialized successfully")

    async def train_models(
        self,
        symbol: str = 'BTC/USDT',
        days: int = 365
    ) -> Dict[str, Any]:
        """
        Train all AI models on historical data.

        Args:
            symbol: Trading symbol
            days: Days of historical data

        Returns:
            Training results
        """
        logger.info(f"Starting model training for {symbol}...")
        results = {}

        # Prepare training data
        training_data = self.data_manager.prepare_training_data(
            symbol=symbol,
            sequence_length=self.config.neural_network.lstm_sequence_length,
            prediction_horizon=1
        )

        self.feature_columns = training_data['feature_names']
        input_size = len(self.feature_columns)

        # Train LSTM
        logger.info("Training LSTM model...")
        self.lstm_model = LSTMPredictor(
            input_size=input_size,
            hidden_size=self.config.neural_network.lstm_hidden_size,
            num_layers=self.config.neural_network.lstm_num_layers,
            dropout=self.config.neural_network.lstm_dropout,
            learning_rate=self.config.neural_network.lstm_learning_rate
        )

        lstm_history = self.lstm_model.train(
            training_data['X_train'],
            training_data['y_train'],
            training_data['X_val'],
            training_data['y_val'],
            epochs=50,
            early_stopping=15
        )
        results['lstm'] = self.lstm_model.evaluate(
            training_data['X_test'],
            training_data['y_test']
        )

        # Train GRU
        logger.info("Training GRU model...")
        self.gru_model = GRUPredictor(
            input_size=input_size,
            hidden_size=self.config.neural_network.gru_hidden_size,
            num_layers=self.config.neural_network.gru_num_layers
        )

        gru_history = self.gru_model.train(
            training_data['X_train'],
            training_data['y_train'],
            training_data['X_val'],
            training_data['y_val'],
            epochs=50,
            early_stopping=15
        )
        results['gru'] = self.gru_model.evaluate(
            training_data['X_test'],
            training_data['y_test']
        )

        # Train CNN for pattern recognition
        logger.info("Training CNN model...")
        cnn_data = self.data_manager.prepare_cnn_data(symbol)

        self.cnn_model = CNNPatternRecognizer(
            input_channels=1,
            image_size=self.config.neural_network.cnn_image_size
        )

        self.cnn_model.train(
            cnn_data['X_train'],
            cnn_data['y_train'],
            cnn_data['X_val'],
            cnn_data['y_val'],
            epochs=30
        )
        results['cnn'] = self.cnn_model.evaluate(
            cnn_data['X_test'],
            cnn_data['y_test']
        )

        # Train RL agents
        logger.info("Training DQL agent...")
        df = self.data_manager.get_historical_data(symbol, add_features=True)
        df_clean = df.select_dtypes(include=[np.number]).dropna()

        # State size includes market features + 3 account features (position, balance, unrealized_pnl)
        state_size = (df_clean.shape[1] + 3) * self.config.neural_network.lstm_sequence_length

        self.dql_agent = DQLTradingAgent(
            state_size=state_size,
            action_size=3,
            learning_rate=self.config.reinforcement_learning.dql_learning_rate
        )

        env = TradingEnvironment(
            df=df_clean.values,
            feature_columns=list(df_clean.columns),
            window_size=self.config.neural_network.lstm_sequence_length
        )

        dql_results = self.dql_agent.train(env, episodes=100, verbose=True)
        results['dql'] = self.dql_agent.evaluate(env)

        # Setup ensemble strategy
        self._setup_ensemble()

        # Setup model explainer
        self._setup_explainer(training_data)

        self.models_trained = True
        logger.info("All models trained successfully")

        return results

    def _setup_ensemble(self) -> None:
        """Setup ensemble strategy with trained models."""
        models = {}

        if self.lstm_model:
            models['lstm'] = self.lstm_model
        if self.gru_model:
            models['gru'] = self.gru_model
        if self.cnn_model:
            models['cnn'] = self.cnn_model
        if self.dql_agent:
            models['dql'] = self.dql_agent
        if self.sentiment_analyzer:
            models['sentiment'] = self.sentiment_analyzer

        self.ensemble_strategy = EnsembleStrategy(
            models=models,
            voting_method='weighted',
            confidence_threshold=0.6
        )

    def _setup_explainer(self, training_data: Dict) -> None:
        """Setup model explainer."""
        self.model_explainer = ModelExplainer(
            feature_names=self.feature_columns,
            model_type='neural_network'
        )

        # Setup SHAP with background data
        if self.lstm_model:
            self.model_explainer.setup_shap(
                model_predict=lambda x: self.lstm_model.predict(x),
                background_data=training_data['X_train'][:100]
            )

    async def run_optimization(
        self,
        symbol: str = 'BTC/USDT'
    ) -> Dict:
        """
        Run genetic algorithm optimization.

        Args:
            symbol: Trading symbol

        Returns:
            Optimization results
        """
        logger.info("Running genetic algorithm optimization...")

        parameter_ranges = [
            ParameterRange('take_profit', 0.01, 0.10),
            ParameterRange('stop_loss', 0.01, 0.05),
            ParameterRange('entry_threshold', 0.4, 0.8),
            ParameterRange('rsi_period', 7, 21, is_integer=True),
        ]

        self.genetic_optimizer = GeneticOptimizer(
            parameter_ranges=parameter_ranges,
            population_size=self.config.genetic.population_size,
            generations=self.config.genetic.generations,
            mutation_rate=self.config.genetic.mutation_rate
        )

        def backtest_fitness(params: Dict) -> Dict:
            # Simplified backtest for optimization
            return {
                'sharpe_ratio': np.random.uniform(-1, 2),  # Placeholder
                'profit_factor': np.random.uniform(0.5, 2),
                'win_rate': np.random.uniform(0.3, 0.7)
            }

        results = self.genetic_optimizer.optimize_strategy(
            backtest_function=backtest_fitness,
            metric='sharpe_ratio'
        )

        # Update self-learning engine with optimized params
        if results.get('best_parameters'):
            self.self_learning_engine.strategy_params.update(results['best_parameters'])

        return results

    async def get_trading_signal(
        self,
        symbol: str
    ) -> Dict:
        """
        Get trading signal for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Trading signal with analysis
        """
        if not self.models_trained:
            logger.warning("Models not trained yet")
            return {'signal': 'hold', 'reason': 'Models not trained'}

        # Get current market data
        features = self.data_manager.get_realtime_features(
            symbol,
            sequence_length=self.config.neural_network.lstm_sequence_length
        )

        if features is None:
            return {'signal': 'hold', 'reason': 'Insufficient data'}

        # Get sentiment data
        news = await self.sentiment_analyzer.fetch_news(symbol)
        sentiment_data = self.sentiment_analyzer.get_market_sentiment(news)

        # Get current price
        ticker = self.data_manager.market_data.fetch_ticker(symbol)
        current_price = ticker.get('last', 0)

        # Get ensemble signal
        signal = self.ensemble_strategy.get_trading_signal(
            features=features,
            sentiment_data=sentiment_data,
            current_price=current_price
        )

        # Get self-learning recommendation
        if self.self_learning_engine:
            predictions = signal.get('individual_predictions', {})
            sl_signal = self.self_learning_engine.suggest_action(
                predictions={
                    k: {'signal': v.get('signal', 'hold'), 'confidence': v.get('confidence', 0.5)}
                    for k, v in predictions.items()
                },
                current_price=current_price
            )
            signal['self_learning_recommendation'] = sl_signal

        # Add explainability
        if self.model_explainer and signal['signal'] != 'hold':
            explanation = self.model_explainer.explain_trading_decision(
                model_predict=lambda x: self.lstm_model.predict(x),
                instance=features[0, -1],  # Last timestep features
                decision=signal['signal'],
                confidence=signal['confidence']
            )
            signal['explanation'] = explanation

        return signal

    async def execute_trade(
        self,
        symbol: str,
        signal: Dict
    ) -> Optional[Dict]:
        """
        Execute a trade based on signal.

        Args:
            symbol: Trading symbol
            signal: Trading signal

        Returns:
            Trade result or None
        """
        if signal['signal'] == 'hold':
            return None

        ticker = self.data_manager.market_data.fetch_ticker(symbol)
        current_price = ticker.get('last', 0)

        if current_price == 0:
            logger.error(f"Could not get price for {symbol}")
            return None

        # Get sizing recommendation
        sl_params = self.self_learning_engine.get_current_params()
        take_profit = sl_params.get('take_profit', 0.03)
        stop_loss = sl_params.get('stop_loss', 0.02)

        if signal['signal'] == 'buy':
            tp_price = current_price * (1 + take_profit)
            sl_price = current_price * (1 - stop_loss)
            side = 'long'
        else:
            tp_price = current_price * (1 - take_profit)
            sl_price = current_price * (1 + stop_loss)
            side = 'short'

        # Calculate position size
        sizing = self.position_sizer.get_sizing_recommendation(
            balance=self.risk_manager.current_balance,
            entry_price=current_price,
            stop_loss=sl_price,
            confidence=signal['confidence']
        )

        # Check risk and open position
        position = self.risk_manager.open_position(
            symbol=symbol,
            side=side,
            entry_price=current_price,
            size=sizing['recommended_size'],
            stop_loss=sl_price,
            take_profit=tp_price
        )

        if position:
            self.logger.trade_opened(
                symbol=symbol,
                side=side,
                entry_price=current_price,
                size=sizing['recommended_size'],
                stop_loss=sl_price,
                take_profit=tp_price
            )

            return {
                'symbol': symbol,
                'side': side,
                'entry_price': current_price,
                'size': sizing['recommended_size'],
                'stop_loss': sl_price,
                'take_profit': tp_price,
                'confidence': signal['confidence']
            }

        return None

    async def monitor_positions(self) -> List[Dict]:
        """Monitor and update open positions."""
        closed_trades = []

        for symbol, position in list(self.risk_manager.positions.items()):
            ticker = self.data_manager.market_data.fetch_ticker(symbol)
            current_price = ticker.get('last', 0)

            if current_price == 0:
                continue

            exit_reason = self.risk_manager.update_position(symbol, current_price)

            if exit_reason:
                trade_result = self.risk_manager.close_position(
                    symbol=symbol,
                    exit_price=current_price,
                    reason=exit_reason
                )

                if trade_result:
                    closed_trades.append(trade_result)

                    # Update self-learning engine
                    self.self_learning_engine.record_trade(trade_result)

                    # Update position sizer stats
                    self.position_sizer.update_stats(trade_result)

                    self.logger.trade_closed(
                        symbol=symbol,
                        side=trade_result['side'],
                        entry_price=trade_result['entry_price'],
                        exit_price=current_price,
                        pnl=trade_result['pnl'],
                        pnl_pct=trade_result['pnl_pct'],
                        reason=exit_reason
                    )

        return closed_trades

    async def run(
        self,
        symbols: Optional[List[str]] = None,
        interval: int = 60
    ) -> None:
        """
        Main trading loop.

        Args:
            symbols: List of symbols to trade
            interval: Seconds between iterations
        """
        symbols = symbols or self.config.trading.symbols
        self.is_running = True

        logger.info(f"Starting trading bot for symbols: {symbols}")

        while self.is_running:
            try:
                for symbol in symbols:
                    # Get signal
                    signal = await self.get_trading_signal(symbol)

                    # Execute trade if signal is strong enough
                    if signal['signal'] != 'hold' and signal['confidence'] > 0.6:
                        await self.execute_trade(symbol, signal)

                # Monitor positions
                await self.monitor_positions()

                # Log performance
                if datetime.utcnow().minute == 0:
                    report = self.risk_manager.get_risk_report()
                    self.logger.performance_update(
                        total_trades=len(self.risk_manager.trade_history),
                        win_rate=report.get('total_return', 0),
                        profit_factor=1.0,
                        total_pnl=report.get('total_return', 0) * self.risk_manager.initial_balance
                    )

                await asyncio.sleep(interval)

            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(10)

    def stop(self) -> None:
        """Stop the trading bot."""
        self.is_running = False
        logger.info("Trading bot stopped")

    def get_status(self) -> Dict:
        """Get current bot status."""
        return {
            'mode': self.mode,
            'is_running': self.is_running,
            'models_trained': self.models_trained,
            'open_positions': len(self.risk_manager.positions) if self.risk_manager else 0,
            'risk_report': self.risk_manager.get_risk_report() if self.risk_manager else {},
            'learning_stats': self.self_learning_engine.get_learning_stats() if self.self_learning_engine else {}
        }

    async def save_state(self, directory: str = 'saved_models') -> None:
        """Save all model states."""
        save_dir = Path(directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.lstm_model:
            self.lstm_model.save(save_dir / 'lstm_model.pt')
        if self.gru_model:
            self.gru_model.save(save_dir / 'gru_model.pt')
        if self.cnn_model:
            self.cnn_model.save(save_dir / 'cnn_model.pt')
        if self.dql_agent:
            self.dql_agent.save(save_dir / 'dql_agent.pt')
        if self.self_learning_engine:
            self.self_learning_engine.save(save_dir / 'learning_state.json')
        if self.genetic_optimizer:
            self.genetic_optimizer.save(save_dir / 'optimizer_state.json')

        logger.info(f"All states saved to {save_dir}")

    async def load_state(self, directory: str = 'saved_models') -> None:
        """Load all model states."""
        load_dir = Path(directory)

        if (load_dir / 'lstm_model.pt').exists():
            self.lstm_model = LSTMPredictor.from_checkpoint(load_dir / 'lstm_model.pt')
        if (load_dir / 'learning_state.json').exists():
            self.self_learning_engine.load(load_dir / 'learning_state.json')

        self._setup_ensemble()
        self.models_trained = True

        logger.info(f"States loaded from {load_dir}")


async def main():
    """Main entry point."""
    bot = AITradingBot(mode='paper')
    await bot.initialize()

    # Train models
    results = await bot.train_models(symbol='BTC/USDT', days=180)
    print(f"Training results: {results}")

    # Run optimization
    opt_results = await bot.run_optimization()
    print(f"Optimization results: {opt_results}")

    # Start trading
    await bot.run(symbols=['BTC/USDT', 'ETH/USDT'], interval=60)


if __name__ == "__main__":
    asyncio.run(main())
