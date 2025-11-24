"""
Central data management for AI Trading Bot.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import sqlite3
import json
from loguru import logger

from .market_data import MarketDataFetcher
from .data_preprocessor import DataPreprocessor


class DataManager:
    """
    Central manager for all data operations.
    Handles caching, preprocessing, and data distribution to models.
    """

    def __init__(
        self,
        exchange_id: str = 'binance',
        api_key: Optional[str] = None,
        secret: Optional[str] = None,
        testnet: bool = True,
        cache_dir: str = 'data_cache',
        db_path: Optional[str] = None
    ):
        """
        Initialize the data manager.

        Args:
            exchange_id: Exchange identifier
            api_key: Exchange API key
            secret: Exchange API secret
            testnet: Use testnet/sandbox mode
            cache_dir: Directory for caching data
            db_path: Path to SQLite database
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.market_data = MarketDataFetcher(
            exchange_id=exchange_id,
            api_key=api_key,
            secret=secret,
            testnet=testnet
        )
        self.preprocessor = DataPreprocessor()

        # Database for storing processed data
        self.db_path = db_path or str(self.cache_dir / 'trading.db')
        self._init_database()

        # In-memory cache
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._last_update: Dict[str, datetime] = {}

        logger.info("DataManager initialized")

    def _init_database(self) -> None:
        """Initialize SQLite database for data storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # OHLCV data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ohlcv (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                UNIQUE(symbol, timeframe, timestamp)
            )
        ''')

        # Trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT UNIQUE,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                size REAL NOT NULL,
                pnl REAL,
                pnl_pct REAL,
                entry_time INTEGER NOT NULL,
                exit_time INTEGER,
                status TEXT DEFAULT 'open',
                strategy TEXT,
                metadata TEXT
            )
        ''')

        # Model predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL,
                actual_outcome TEXT,
                metadata TEXT
            )
        ''')

        # Performance metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                total_trades INTEGER,
                win_rate REAL,
                profit_factor REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                total_pnl REAL,
                metadata TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def get_historical_data(
        self,
        symbol: str,
        timeframe: str = '1h',
        days: int = 365,
        add_features: bool = True,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Get historical data with optional features.

        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for candles
            days: Number of days of history
            add_features: Whether to add technical indicators
            use_cache: Whether to use cached data

        Returns:
            DataFrame with OHLCV data and optional features
        """
        cache_key = f"{symbol}_{timeframe}_{days}"

        # Check memory cache
        if use_cache and cache_key in self._data_cache:
            last_update = self._last_update.get(cache_key)
            if last_update and (datetime.utcnow() - last_update).seconds < 3600:
                logger.debug(f"Using cached data for {cache_key}")
                return self._data_cache[cache_key].copy()

        # Fetch from exchange
        df = self.market_data.fetch_historical_data(symbol, timeframe, days)

        if df.empty:
            logger.warning(f"No data fetched for {symbol}")
            return pd.DataFrame()

        # Add technical indicators
        if add_features:
            df = self.preprocessor.add_technical_indicators(df)

        # Cache the data
        self._data_cache[cache_key] = df
        self._last_update[cache_key] = datetime.utcnow()

        # Store in database
        self._store_ohlcv(df, symbol, timeframe)

        return df.copy()

    def _store_ohlcv(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str
    ) -> None:
        """Store OHLCV data in database."""
        conn = sqlite3.connect(self.db_path)

        for idx, row in df.iterrows():
            try:
                conn.execute('''
                    INSERT OR REPLACE INTO ohlcv
                    (symbol, timeframe, timestamp, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    timeframe,
                    int(idx.timestamp()),
                    row['open'],
                    row['high'],
                    row['low'],
                    row['close'],
                    row['volume']
                ))
            except Exception as e:
                logger.debug(f"Error storing OHLCV: {e}")

        conn.commit()
        conn.close()

    def prepare_training_data(
        self,
        symbol: str,
        timeframe: str = '1h',
        sequence_length: int = 60,
        prediction_horizon: int = 1,
        target_type: str = 'regression',
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Dict[str, np.ndarray]:
        """
        Prepare data for training neural networks.

        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for data
            sequence_length: Sequence length for LSTM/GRU
            prediction_horizon: Steps ahead to predict
            target_type: 'regression' or 'classification'
            train_ratio: Training data ratio
            val_ratio: Validation data ratio

        Returns:
            Dictionary with train/val/test splits
        """
        # Get data with features
        df = self.get_historical_data(symbol, timeframe, add_features=True)

        if df.empty:
            raise ValueError(f"No data available for {symbol}")

        # Scale data
        df_scaled = self.preprocessor.scale_data(df, fit=True, scaler_name=symbol)

        # Create sequences
        X, y, feature_names = self.preprocessor.prepare_sequences(
            df_scaled,
            sequence_length=sequence_length,
            target_column='close',
            prediction_horizon=prediction_horizon,
            target_type=target_type
        )

        # Split data
        splits = self.preprocessor.split_data(
            X, y,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            shuffle=False  # Maintain temporal order
        )

        splits['feature_names'] = feature_names
        splits['scaler_name'] = symbol

        logger.info(f"Prepared training data for {symbol}: {X.shape}")
        return splits

    def prepare_cnn_data(
        self,
        symbol: str,
        timeframe: str = '1h',
        window_size: int = 60,
        image_size: Tuple[int, int] = (64, 64)
    ) -> Dict[str, np.ndarray]:
        """
        Prepare chart images for CNN.

        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for data
            window_size: Number of candles per image
            image_size: Output image size

        Returns:
            Dictionary with image data and labels
        """
        df = self.get_historical_data(symbol, timeframe, add_features=False)

        if df.empty:
            raise ValueError(f"No data available for {symbol}")

        # Create images
        images = self.preprocessor.create_chart_images(
            df, window_size=window_size, image_size=image_size
        )

        # Create labels (price direction)
        labels = []
        for i in range(window_size, len(df)):
            current = df['close'].iloc[i - 1]
            future = df['close'].iloc[i] if i < len(df) else current
            labels.append(1 if future > current else 0)

        labels = np.array(labels[:len(images)])

        # Split data
        train_size = int(len(images) * 0.7)
        val_size = int(len(images) * 0.15)

        return {
            'X_train': images[:train_size],
            'y_train': labels[:train_size],
            'X_val': images[train_size:train_size + val_size],
            'y_val': labels[train_size:train_size + val_size],
            'X_test': images[train_size + val_size:],
            'y_test': labels[train_size + val_size:]
        }

    def get_latest_data(
        self,
        symbol: str,
        timeframe: str = '1h',
        bars: int = 100,
        add_features: bool = True
    ) -> pd.DataFrame:
        """
        Get latest market data for real-time trading.

        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for candles
            bars: Number of recent bars
            add_features: Whether to add technical indicators

        Returns:
            DataFrame with latest data
        """
        df = self.market_data.fetch_ohlcv(symbol, timeframe, limit=bars)

        if df.empty:
            return pd.DataFrame()

        if add_features:
            df = self.preprocessor.add_technical_indicators(df)

        return df

    def get_realtime_features(
        self,
        symbol: str,
        sequence_length: int = 60
    ) -> Optional[np.ndarray]:
        """
        Get real-time feature sequence for prediction.

        Args:
            symbol: Trading pair symbol
            sequence_length: Required sequence length

        Returns:
            Feature array ready for model input
        """
        df = self.get_latest_data(symbol, bars=sequence_length + 50)

        if len(df) < sequence_length:
            logger.warning(f"Insufficient data for {symbol}")
            return None

        # Add features and scale
        df = self.preprocessor.add_technical_indicators(df)
        df = df.dropna()

        if len(df) < sequence_length:
            return None

        # Scale using existing scaler
        try:
            df_scaled = self.preprocessor.scale_data(df, fit=False, scaler_name=symbol)
        except ValueError:
            # Fit new scaler if not exists
            df_scaled = self.preprocessor.scale_data(df, fit=True, scaler_name=symbol)

        # Get numeric features only
        numeric_df = df_scaled.select_dtypes(include=[np.number])

        # Return last sequence
        return numeric_df.values[-sequence_length:].reshape(1, sequence_length, -1)

    def store_trade(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        entry_price: float,
        size: float,
        strategy: str = 'unknown',
        metadata: Optional[Dict] = None
    ) -> None:
        """Store a new trade in database."""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            INSERT INTO trades
            (trade_id, symbol, side, entry_price, size, entry_time, strategy, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade_id,
            symbol,
            side,
            entry_price,
            size,
            int(datetime.utcnow().timestamp()),
            strategy,
            json.dumps(metadata or {})
        ))
        conn.commit()
        conn.close()

    def close_trade(
        self,
        trade_id: str,
        exit_price: float,
        pnl: float,
        pnl_pct: float
    ) -> None:
        """Close a trade in database."""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            UPDATE trades
            SET exit_price = ?, pnl = ?, pnl_pct = ?,
                exit_time = ?, status = 'closed'
            WHERE trade_id = ?
        ''', (
            exit_price,
            pnl,
            pnl_pct,
            int(datetime.utcnow().timestamp()),
            trade_id
        ))
        conn.commit()
        conn.close()

    def get_trade_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
        status: str = 'closed'
    ) -> List[Dict]:
        """Get trade history from database."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        query = "SELECT * FROM trades WHERE status = ?"
        params = [status]

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        query += " ORDER BY entry_time DESC LIMIT ?"
        params.append(limit)

        cursor = conn.execute(query, params)
        trades = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return trades

    def store_prediction(
        self,
        model_name: str,
        symbol: str,
        prediction: str,
        confidence: float,
        metadata: Optional[Dict] = None
    ) -> None:
        """Store model prediction in database."""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            INSERT INTO predictions
            (model_name, symbol, timestamp, prediction, confidence, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            model_name,
            symbol,
            int(datetime.utcnow().timestamp()),
            prediction,
            confidence,
            json.dumps(metadata or {})
        ))
        conn.commit()
        conn.close()

    def update_performance_metrics(
        self,
        metrics: Dict
    ) -> None:
        """Store performance metrics in database."""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            INSERT INTO performance
            (timestamp, total_trades, win_rate, profit_factor,
             sharpe_ratio, max_drawdown, total_pnl, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            int(datetime.utcnow().timestamp()),
            metrics.get('total_trades', 0),
            metrics.get('win_rate', 0),
            metrics.get('profit_factor', 0),
            metrics.get('sharpe_ratio', 0),
            metrics.get('max_drawdown', 0),
            metrics.get('total_pnl', 0),
            json.dumps(metrics)
        ))
        conn.commit()
        conn.close()

    def clear_cache(self) -> None:
        """Clear in-memory data cache."""
        self._data_cache.clear()
        self._last_update.clear()
        logger.info("Data cache cleared")

    async def close(self) -> None:
        """Clean up resources."""
        await self.market_data.close()
        self.clear_cache()
