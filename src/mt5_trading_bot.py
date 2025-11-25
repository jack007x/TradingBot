"""
MetaTrader 5 AI Trading Bot - Self-Learning Trading System
==========================================================

Main orchestrator for MT5-based AI trading with self-learning capabilities.
"""

import asyncio
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import numpy as np
from loguru import logger

from .utils.config import Config
from .utils.logger import setup_logger, TradingLogger
from .mt5.mt5_connector import MT5Connector
from .mt5.mt5_data import MT5DataFetcher
from .mt5.mt5_trader import MT5Trader, TradeResult
from .data.data_preprocessor import DataPreprocessor
from .models.neural_networks.lstm_model import LSTMPredictor
from .models.neural_networks.gru_model import GRUPredictor
from .models.neural_networks.cnn_model import CNNPatternRecognizer
from .models.neural_networks.directional_predictor import DirectionalPredictor
from .models.reinforcement_learning.dql_agent import DQLTradingAgent
from .models.reinforcement_learning.trading_env import TradingEnvironment
from .models.nlp.sentiment_analyzer import SentimentAnalyzer
from .models.genetic.optimizer import GeneticOptimizer, ParameterRange
from .models.xai.explainer import ModelExplainer
from .risk_management.risk_manager import RiskManager
from .risk_management.position_sizer import PositionSizer
from .strategies.self_learning_engine import SelfLearningEngine
from .strategies.ensemble_strategy import EnsembleStrategy


class MT5TradingBot:
    """
    MetaTrader 5 AI Trading Bot with self-learning capabilities.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        mode: str = 'paper'
    ):
        """
        Initialize MT5 Trading Bot.

        Args:
            config_path: Path to configuration file
            mode: Trading mode ('paper', 'live')
        """
        # Load configuration
        self.config = Config(config_path)
        self.mode = mode

        # Setup logging
        setup_logger(
            log_file=self.config.get('logging.file', 'logs/mt5_trading_bot.log'),
            level=self.config.get('general.log_level', 'INFO')
        )
        self.logger = TradingLogger('mt5_bot')

        # MT5 Components
        self.mt5_connector: Optional[MT5Connector] = None
        self.mt5_data: Optional[MT5DataFetcher] = None
        self.mt5_trader: Optional[MT5Trader] = None

        # Data Processing
        self.preprocessor = DataPreprocessor()

        # AI Models
        self.lstm_model: Optional[LSTMPredictor] = None
        self.gru_model: Optional[GRUPredictor] = None
        self.cnn_model: Optional[CNNPatternRecognizer] = None
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

        logger.info(f"MT5TradingBot initialized in {mode} mode")

    def initialize(self) -> bool:
        """
        Initialize all components and connect to MT5.

        Returns:
            True if initialization successful
        """
        logger.info("Initializing MT5 Trading Bot...")

        # Get MT5 credentials from config or environment
        mt5_login = self.config.get('mt5.login') or os.environ.get('MT5_LOGIN')
        mt5_password = self.config.get('mt5.password') or os.environ.get('MT5_PASSWORD')
        mt5_server = self.config.get('mt5.server') or os.environ.get('MT5_SERVER')
        mt5_path = self.config.get('mt5.path') or os.environ.get('MT5_PATH')

        # Initialize MT5 connector
        self.mt5_connector = MT5Connector(
            path=mt5_path,
            login=int(mt5_login) if mt5_login else None,
            password=mt5_password,
            server=mt5_server,
            timeout=self.config.get('mt5.timeout', 60000),
            portable=self.config.get('mt5.portable', False)
        )

        # Connect to MT5
        if not self.mt5_connector.connect():
            logger.error("Failed to connect to MetaTrader 5")
            return False

        # Initialize data fetcher and trader
        self.mt5_data = MT5DataFetcher(self.mt5_connector)
        self.mt5_trader = MT5Trader(
            connector=self.mt5_connector,
            magic_number=self.config.get('mt5.magic_number', 123456),
            deviation=self.config.get('mt5.deviation', 20),
            fill_policy=self.config.get('mt5.fill_policy', 'ioc')
        )

        # Initialize risk management
        account_info = self.mt5_connector.get_account_info()
        initial_balance = account_info['balance'] if account_info else 10000

        self.risk_manager = RiskManager(
            initial_balance=initial_balance,
            max_risk_per_trade=self.config.risk.max_risk_per_trade,
            max_daily_drawdown=self.config.risk.max_daily_drawdown,
            max_total_drawdown=self.config.risk.max_total_drawdown,
            max_open_positions=self.config.get('trading.max_open_positions', 5)
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

        logger.info("MT5 Trading Bot initialized successfully")
        logger.info(f"Account: {account_info['login'] if account_info else 'N/A'}")
        logger.info(f"Balance: {account_info['balance'] if account_info else 'N/A'} {account_info['currency'] if account_info else ''}")

        return True

    def train_models(
        self,
        symbol: str = 'EURUSD',
        timeframe: str = '1h',
        days: int = 365
    ) -> Dict[str, Any]:
        """
        Train all AI models on historical MT5 data.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe for training
            days: Days of historical data

        Returns:
            Training results
        """
        logger.info(f"Starting model training for {symbol}...")
        results = {}

        # Fetch historical data from MT5
        df = self.mt5_data.fetch_historical_data(symbol, timeframe, days)

        if df.empty:
            logger.error("Failed to fetch historical data")
            return {'error': 'No data available'}

        # Add technical indicators
        df = self.preprocessor.add_technical_indicators(df)
        df = df.dropna()

        logger.info(f"Training data: {len(df)} bars with {len(df.columns)} features")

        # Prepare directional sequences for classification
        # Use adaptive threshold based on ATR to handle volatility
        logger.info("Using adaptive threshold based on ATR for direction labeling")
        X, y, feature_names = self.preprocessor.prepare_directional_sequences(
            df,
            sequence_length=self.config.neural_network.lstm_sequence_length,
            prediction_horizon=1,
            direction_threshold=None,  # Use adaptive
            adaptive_threshold=True,
            target_col='close'
        )

        self.feature_columns = feature_names
        input_size = len(feature_names)

        # Split data
        splits = self.preprocessor.split_data(X, y, train_ratio=0.7, val_ratio=0.15)

        # Train Directional LSTM with improved hyperparameters
        logger.info("Training Directional LSTM model...")
        self.lstm_model = DirectionalPredictor(
            input_size=input_size,
            model_type='lstm',
            hidden_size=96,  # Reduced to prevent overfitting
            num_layers=2,
            dropout=0.2,  # Reduced dropout
            num_classes=3,
            learning_rate=5e-4,  # Lower LR for stability
            weight_decay=1e-5
        )

        self.lstm_model.train(
            splits['X_train'], splits['y_train'],
            splits['X_val'], splits['y_val'],
            epochs=100,
            batch_size=64,  # Larger batch
            early_stopping_patience=20,  # More patience
            use_smote=True,  # Enable SMOTE for class balancing
            smote_k_neighbors=3  # Conservative neighbors
        )
        results['lstm'] = self.lstm_model.evaluate(splits['X_test'], splits['y_test'])
        logger.info(f"LSTM Results: Accuracy={results['lstm']['accuracy']:.4f}, "
                   f"Balanced={results['lstm']['balanced_accuracy']:.4f}")

        # Check if accuracy meets minimum threshold
        if results['lstm']['balanced_accuracy'] < 0.50:
            logger.warning(f"LSTM balanced accuracy ({results['lstm']['balanced_accuracy']:.4f}) "
                          "below 50%! Model needs improvement.")

        # Train Directional GRU with improved hyperparameters
        logger.info("Training Directional GRU model...")
        self.gru_model = DirectionalPredictor(
            input_size=input_size,
            model_type='gru',
            hidden_size=96,  # Reduced to prevent overfitting
            num_layers=2,
            dropout=0.2,  # Reduced dropout
            num_classes=3,
            learning_rate=5e-4,  # Lower LR for stability
            weight_decay=1e-5
        )

        self.gru_model.train(
            splits['X_train'], splits['y_train'],
            splits['X_val'], splits['y_val'],
            epochs=100,
            batch_size=64,  # Larger batch
            early_stopping_patience=20,  # More patience
            use_smote=True,  # Enable SMOTE for class balancing
            smote_k_neighbors=3  # Conservative neighbors
        )
        results['gru'] = self.gru_model.evaluate(splits['X_test'], splits['y_test'])
        logger.info(f"GRU Results: Accuracy={results['gru']['accuracy']:.4f}, "
                   f"Balanced={results['gru']['balanced_accuracy']:.4f}")

        # Check if accuracy meets minimum threshold
        if results['gru']['balanced_accuracy'] < 0.50:
            logger.warning(f"GRU balanced accuracy ({results['gru']['balanced_accuracy']:.4f}) "
                          "below 50%! Model needs improvement.")

        # Train DQL Agent with reduced features and window
        logger.info("Training DQL agent with compact state...")
        from src.utils.feature_selector import SimpleFeatureSelector

        # Select only essential features for RL (10-15 features)
        df_rl, rl_features = SimpleFeatureSelector.select_features(df)

        # Use small window for RL (10 bars instead of 60)
        rl_window = 10
        # State size: essential features + 3 account features
        state_size = (len(rl_features) + 3) * rl_window

        logger.info(f"DQL State: {len(rl_features)} features × {rl_window} window = {state_size} dims")

        self.dql_agent = DQLTradingAgent(
            state_size=state_size,
            action_size=3,
            learning_rate=1e-4,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay_steps=20000,
            max_buffer_memory_mb=128
        )

        env = TradingEnvironment(
            df=df_rl.values,
            feature_columns=rl_features,
            window_size=rl_window
        )

        self.dql_agent.train(env, episodes=100, verbose=True)
        results['dql'] = self.dql_agent.evaluate(env)
        logger.info(f"DQL Results: Mean Return={results['dql']['mean_return']:.4f}, "
                   f"Win Rate={results['dql']['mean_win_rate']:.4f}, "
                   f"Sharpe={results['dql']['mean_sharpe']:.4f}")

        # Check if DQL meets minimum thresholds
        if results['dql']['mean_return'] < 0:
            logger.warning(f"DQL mean return ({results['dql']['mean_return']:.4f}) is negative!")
        if results['dql']['mean_win_rate'] < 0.50:
            logger.warning(f"DQL win rate ({results['dql']['mean_win_rate']:.4f}) below 50%!")
        if results['dql']['mean_sharpe'] < 0.5:
            logger.warning(f"DQL Sharpe ratio ({results['dql']['mean_sharpe']:.4f}) below 0.5!")

        # Setup ensemble
        self._setup_ensemble()
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
        if self.dql_agent:
            models['dql'] = self.dql_agent
        if self.sentiment_analyzer:
            models['sentiment'] = self.sentiment_analyzer

        self.ensemble_strategy = EnsembleStrategy(
            models=models,
            voting_method='weighted',
            confidence_threshold=0.6
        )

    def get_realtime_features(
        self,
        symbol: str,
        sequence_length: int = 60
    ) -> Optional[np.ndarray]:
        """Get real-time feature sequence for prediction."""
        df = self.mt5_data.fetch_ohlcv(symbol, '1h', sequence_length + 50)

        if df.empty or len(df) < sequence_length:
            return None

        df = self.preprocessor.add_technical_indicators(df)
        df = df.dropna()

        if len(df) < sequence_length:
            return None

        # Scale data
        try:
            df_scaled = self.preprocessor.scale_data(df, fit=False, scaler_name=symbol)
        except ValueError:
            df_scaled = self.preprocessor.scale_data(df, fit=True, scaler_name=symbol)

        numeric_df = df_scaled.select_dtypes(include=[np.number])
        return numeric_df.values[-sequence_length:].reshape(1, sequence_length, -1)

    async def get_trading_signal(self, symbol: str) -> Dict:
        """
        Get trading signal for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Trading signal with analysis
        """
        if not self.models_trained:
            return {'signal': 'hold', 'reason': 'Models not trained'}

        # Get features
        features = self.get_realtime_features(
            symbol,
            sequence_length=self.config.neural_network.lstm_sequence_length
        )

        if features is None:
            return {'signal': 'hold', 'reason': 'Insufficient data'}

        # Get current price
        ticker = self.mt5_data.get_ticker(symbol)
        if not ticker:
            return {'signal': 'hold', 'reason': 'Cannot get price'}

        current_price = ticker['last'] or ticker['bid']

        # Get sentiment (if available)
        try:
            news = await self.sentiment_analyzer.fetch_news(symbol)
            sentiment_data = self.sentiment_analyzer.get_market_sentiment(news)
        except Exception:
            sentiment_data = None

        # Get ensemble signal
        signal = self.ensemble_strategy.get_trading_signal(
            features=features,
            sentiment_data=sentiment_data,
            current_price=current_price
        )

        # Get self-learning recommendation
        predictions = signal.get('individual_predictions', {})
        sl_signal = self.self_learning_engine.suggest_action(
            predictions={
                k: {'signal': v.get('signal', 'hold'), 'confidence': v.get('confidence', 0.5)}
                for k, v in predictions.items()
            },
            current_price=current_price
        )
        signal['self_learning_recommendation'] = sl_signal
        signal['current_price'] = current_price

        return signal

    def execute_trade(
        self,
        symbol: str,
        signal: Dict
    ) -> Optional[TradeResult]:
        """
        Execute a trade based on signal.

        Args:
            symbol: Trading symbol
            signal: Trading signal

        Returns:
            TradeResult or None
        """
        if signal['signal'] == 'hold':
            return None

        current_price = signal.get('current_price', 0)
        if current_price == 0:
            ticker = self.mt5_data.get_ticker(symbol)
            current_price = ticker['bid'] if ticker else 0

        if current_price == 0:
            logger.error(f"Could not get price for {symbol}")
            return None

        # Get symbol info for lot size calculation
        symbol_info = self.mt5_connector.get_symbol_info(symbol)
        if not symbol_info:
            return None

        # Calculate position parameters
        sl_params = self.self_learning_engine.get_current_params()
        take_profit_pct = sl_params.get('take_profit', 0.03)
        stop_loss_pct = sl_params.get('stop_loss', 0.02)

        # Get ATR for dynamic SL/TP
        atr = self.mt5_data.calculate_atr(symbol, '1h', 14)

        if signal['signal'] == 'buy':
            order_type = 'buy'
            if atr:
                stop_loss = current_price - (atr * 2)
                take_profit = current_price + (atr * 3)
            else:
                stop_loss = current_price * (1 - stop_loss_pct)
                take_profit = current_price * (1 + take_profit_pct)
        else:
            order_type = 'sell'
            if atr:
                stop_loss = current_price + (atr * 2)
                take_profit = current_price - (atr * 3)
            else:
                stop_loss = current_price * (1 + stop_loss_pct)
                take_profit = current_price * (1 - take_profit_pct)

        # Calculate position size
        account_info = self.mt5_connector.get_account_info()
        balance = account_info['balance'] if account_info else 10000

        sizing = self.position_sizer.get_sizing_recommendation(
            balance=balance,
            entry_price=current_price,
            stop_loss=stop_loss,
            confidence=signal['confidence']
        )

        # Convert to lots
        lot_size = self._calculate_lot_size(
            symbol_info,
            sizing['risk_amount'],
            current_price,
            stop_loss
        )

        # Check with risk manager
        can_trade, reason, adjusted_size = self.risk_manager.can_open_position(
            symbol=symbol,
            side='long' if order_type == 'buy' else 'short',
            entry_price=current_price,
            stop_loss=stop_loss,
            proposed_size=lot_size
        )

        if not can_trade:
            logger.warning(f"Trade rejected: {reason}")
            return None

        # Execute order
        result = self.mt5_trader.place_market_order(
            symbol=symbol,
            order_type=order_type,
            volume=adjusted_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            comment=f"AI Bot - Conf: {signal['confidence']:.2f}"
        )

        if result.success:
            self.logger.trade_opened(
                symbol=symbol,
                side=order_type,
                entry_price=result.price,
                size=result.volume,
                stop_loss=stop_loss,
                take_profit=take_profit
            )

        return result

    def _calculate_lot_size(
        self,
        symbol_info: Dict,
        risk_amount: float,
        entry_price: float,
        stop_loss: float
    ) -> float:
        """Calculate lot size based on risk."""
        point = symbol_info['point']
        tick_value = symbol_info['tick_value']
        contract_size = symbol_info['contract_size']

        # Calculate pips risk
        pip_risk = abs(entry_price - stop_loss) / point

        if pip_risk == 0:
            return symbol_info['volume_min']

        # Calculate lot size
        lot_size = risk_amount / (pip_risk * tick_value)

        # Normalize to symbol constraints
        lot_size = max(symbol_info['volume_min'], min(symbol_info['volume_max'], lot_size))
        lot_size = round(lot_size / symbol_info['volume_step']) * symbol_info['volume_step']

        return round(lot_size, 2)

    def monitor_positions(self) -> List[Dict]:
        """Monitor and update open positions."""
        closed_trades = []
        positions = self.mt5_trader.get_positions(magic=self.config.get('mt5.magic_number'))

        for pos in positions:
            symbol = pos['symbol']
            ticker = self.mt5_data.get_ticker(symbol)

            if not ticker:
                continue

            current_price = ticker['bid'] if pos['type'] == 'buy' else ticker['ask']

            # Check if position was closed by SL/TP
            pnl = pos['profit']
            pnl_pct = pnl / self.risk_manager.current_balance if self.risk_manager.current_balance > 0 else 0

            # Record for self-learning
            if pos['ticket'] not in [t.get('ticket') for t in self.self_learning_engine.trade_history]:
                self.logger.info(f"Position {pos['ticket']}: {symbol} {pos['type']} PnL: {pnl:.2f}")

        # Get trade history for learning
        recent_deals = self.mt5_trader.get_trade_history(
            from_date=datetime.now() - timedelta(hours=24)
        )

        for deal in recent_deals:
            if deal['type'] in ['buy', 'sell'] and deal.get('profit', 0) != 0:
                trade_record = {
                    'symbol': deal['symbol'],
                    'side': deal['type'],
                    'pnl': deal['profit'],
                    'pnl_pct': deal['profit'] / self.risk_manager.initial_balance,
                    'reason': 'mt5_closed'
                }
                self.self_learning_engine.record_trade(trade_record)
                self.position_sizer.update_stats(trade_record)

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

        logger.info(f"Starting MT5 trading loop for: {symbols}")

        while self.is_running:
            try:
                # Check MT5 connection
                if not self.mt5_connector.is_connected():
                    logger.warning("MT5 disconnected, attempting reconnect...")
                    if not self.mt5_connector.reconnect():
                        await asyncio.sleep(30)
                        continue

                for symbol in symbols:
                    # Get signal
                    signal = await self.get_trading_signal(symbol)

                    # Execute trade if signal is strong
                    if signal['signal'] != 'hold' and signal['confidence'] > 0.6:
                        self.execute_trade(symbol, signal)

                # Monitor positions
                self.monitor_positions()

                # Log performance periodically
                if datetime.utcnow().minute == 0:
                    self._log_performance()

                await asyncio.sleep(interval)

            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(10)

    def _log_performance(self) -> None:
        """Log current performance metrics."""
        account = self.mt5_connector.get_account_info()
        if account:
            self.logger.info(
                f"Balance: {account['balance']:.2f} | "
                f"Equity: {account['equity']:.2f} | "
                f"Profit: {account['profit']:.2f}"
            )

        stats = self.self_learning_engine.get_learning_stats()
        self.logger.performance_update(
            total_trades=stats['total_trades'],
            win_rate=stats['win_rate'],
            profit_factor=1.0,
            total_pnl=stats['total_pnl']
        )

    def stop(self) -> None:
        """Stop the trading bot."""
        self.is_running = False
        if self.mt5_connector:
            self.mt5_connector.disconnect()
        logger.info("MT5 Trading Bot stopped")

    def get_status(self) -> Dict:
        """Get current bot status."""
        account = self.mt5_connector.get_account_info() if self.mt5_connector else None
        positions = self.mt5_trader.get_positions() if self.mt5_trader else []

        return {
            'mode': self.mode,
            'is_running': self.is_running,
            'connected': self.mt5_connector.is_connected() if self.mt5_connector else False,
            'models_trained': self.models_trained,
            'account': account,
            'open_positions': len(positions),
            'positions': positions,
            'learning_stats': self.self_learning_engine.get_learning_stats() if self.self_learning_engine else {}
        }

    def save_state(self, directory: str = 'saved_models') -> None:
        """Save all model states."""
        save_dir = Path(directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.lstm_model:
            self.lstm_model.save(save_dir / 'lstm_model.pt')
        if self.gru_model:
            self.gru_model.save(save_dir / 'gru_model.pt')
        if self.dql_agent:
            self.dql_agent.save(save_dir / 'dql_agent.pt')
        if self.self_learning_engine:
            self.self_learning_engine.save(save_dir / 'learning_state.json')

        logger.info(f"States saved to {save_dir}")

    def load_state(self, directory: str = 'saved_models') -> None:
        """Load all model states."""
        load_dir = Path(directory)

        if (load_dir / 'lstm_model.pt').exists():
            self.lstm_model = LSTMPredictor.from_checkpoint(load_dir / 'lstm_model.pt')
        if (load_dir / 'learning_state.json').exists():
            self.self_learning_engine.load(load_dir / 'learning_state.json')

        self._setup_ensemble()
        self.models_trained = True
        logger.info(f"States loaded from {load_dir}")
