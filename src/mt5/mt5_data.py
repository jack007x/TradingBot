"""
MetaTrader 5 Data Fetcher.
Handles fetching historical and real-time market data.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pytz
from loguru import logger

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

from .mt5_connector import MT5Connector


class MT5DataFetcher:
    """
    Fetches market data from MetaTrader 5.
    Provides OHLCV data, ticks, and market depth.
    """

    def __init__(self, connector: MT5Connector):
        """
        Initialize data fetcher.

        Args:
            connector: MT5Connector instance
        """
        self.connector = connector
        self.timezone = pytz.timezone('UTC')

        logger.info("MT5DataFetcher initialized")

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = '1h',
        count: int = 1000,
        start_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            count: Number of bars to fetch
            start_time: Start time for data (if None, fetches most recent)

        Returns:
            DataFrame with OHLCV data
        """
        if not self.connector.is_connected():
            logger.error("Not connected to MT5")
            return pd.DataFrame()

        # Select symbol
        if not self.connector.select_symbol(symbol):
            return pd.DataFrame()

        tf = self.connector.get_timeframe(timeframe)

        try:
            if start_time:
                rates = mt5.copy_rates_from(symbol, tf, start_time, count)
            else:
                rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)

            if rates is None or len(rates) == 0:
                error = mt5.last_error()
                logger.warning(f"No data received for {symbol}. Error: {error}")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)

            # Rename columns to standard format
            df = df.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'tick_volume': 'volume',
                'spread': 'spread',
                'real_volume': 'real_volume'
            })

            df['symbol'] = symbol
            df['timeframe'] = timeframe

            logger.debug(f"Fetched {len(df)} bars for {symbol} ({timeframe})")
            return df

        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_historical_data(
        self,
        symbol: str,
        timeframe: str = '1h',
        days: int = 365
    ) -> pd.DataFrame:
        """
        Fetch historical data for specified number of days.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            days: Number of days of history

        Returns:
            DataFrame with historical OHLCV data
        """
        # Calculate approximate number of bars needed
        tf_minutes = self._timeframe_to_minutes(timeframe)
        bars_per_day = (24 * 60) // tf_minutes if tf_minutes > 0 else 24
        total_bars = days * bars_per_day

        # MT5 has limits on bars per request
        max_bars_per_request = 10000
        all_data = []

        start_time = datetime.utcnow() - timedelta(days=days)
        remaining_bars = total_bars

        while remaining_bars > 0:
            batch_size = min(remaining_bars, max_bars_per_request)
            df = self.fetch_ohlcv(symbol, timeframe, batch_size, start_time)

            if df.empty:
                break

            all_data.append(df)
            remaining_bars -= len(df)

            if len(df) < batch_size:
                break

            start_time = df.index[-1] + timedelta(minutes=tf_minutes)

        if all_data:
            result = pd.concat(all_data)
            result = result[~result.index.duplicated(keep='first')]
            result.sort_index(inplace=True)
            logger.info(f"Fetched {len(result)} historical bars for {symbol}")
            return result

        return pd.DataFrame()

    def fetch_ticks(
        self,
        symbol: str,
        count: int = 1000,
        start_time: Optional[datetime] = None,
        flags: int = 0
    ) -> pd.DataFrame:
        """
        Fetch tick data.

        Args:
            symbol: Trading symbol
            count: Number of ticks to fetch
            start_time: Start time for ticks
            flags: Tick flags (mt5.COPY_TICKS_ALL, COPY_TICKS_INFO, COPY_TICKS_TRADE)

        Returns:
            DataFrame with tick data
        """
        if not self.connector.is_connected():
            return pd.DataFrame()

        if not self.connector.select_symbol(symbol):
            return pd.DataFrame()

        try:
            if flags == 0:
                flags = mt5.COPY_TICKS_ALL if MT5_AVAILABLE else 0

            if start_time:
                ticks = mt5.copy_ticks_from(symbol, start_time, count, flags)
            else:
                ticks = mt5.copy_ticks_from(symbol, datetime.utcnow() - timedelta(hours=1), count, flags)

            if ticks is None or len(ticks) == 0:
                return pd.DataFrame()

            df = pd.DataFrame(ticks)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)

            return df

        except Exception as e:
            logger.error(f"Error fetching ticks for {symbol}: {e}")
            return pd.DataFrame()

    def get_current_price(self, symbol: str) -> Optional[Dict]:
        """
        Get current bid/ask price.

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary with bid, ask, and spread
        """
        if not self.connector.is_connected():
            return None

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None

        return {
            'symbol': symbol,
            'bid': tick.bid,
            'ask': tick.ask,
            'last': tick.last,
            'volume': tick.volume,
            'time': datetime.fromtimestamp(tick.time),
            'spread': (tick.ask - tick.bid) / self._get_point(symbol)
        }

    def get_ticker(self, symbol: str) -> Optional[Dict]:
        """
        Get ticker information (similar to exchange API format).

        Args:
            symbol: Trading symbol

        Returns:
            Ticker information dictionary
        """
        price = self.get_current_price(symbol)
        if not price:
            return None

        # Get daily data for high/low
        daily_data = self.fetch_ohlcv(symbol, '1d', 2)

        if not daily_data.empty:
            today = daily_data.iloc[-1]
            return {
                'symbol': symbol,
                'last': price['last'] or price['bid'],
                'bid': price['bid'],
                'ask': price['ask'],
                'high': today['high'],
                'low': today['low'],
                'open': today['open'],
                'close': today['close'],
                'volume': today['volume'],
                'spread': price['spread'],
                'timestamp': price['time']
            }

        return price

    def get_market_depth(
        self,
        symbol: str
    ) -> Optional[Dict]:
        """
        Get market depth (DOM - Depth of Market).

        Args:
            symbol: Trading symbol

        Returns:
            Market depth dictionary
        """
        if not self.connector.is_connected():
            return None

        # Subscribe to market depth
        if not mt5.market_book_add(symbol):
            logger.warning(f"Could not subscribe to market depth for {symbol}")
            return None

        try:
            book = mt5.market_book_get(symbol)
            if book is None:
                return None

            bids = []
            asks = []

            for item in book:
                entry = {
                    'price': item.price,
                    'volume': item.volume,
                    'volume_real': item.volume_real
                }
                if item.type == mt5.BOOK_TYPE_BUY:
                    bids.append(entry)
                elif item.type == mt5.BOOK_TYPE_SELL:
                    asks.append(entry)

            return {
                'symbol': symbol,
                'bids': bids,
                'asks': asks,
                'timestamp': datetime.utcnow()
            }

        finally:
            mt5.market_book_release(symbol)

    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        timeframe: str = '1h',
        count: int = 1000
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple symbols.

        Args:
            symbols: List of trading symbols
            timeframe: Timeframe string
            count: Number of bars per symbol

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        data = {}
        for symbol in symbols:
            df = self.fetch_ohlcv(symbol, timeframe, count)
            data[symbol] = df

        return data

    def calculate_atr(
        self,
        symbol: str,
        timeframe: str = '1h',
        period: int = 14
    ) -> Optional[float]:
        """
        Calculate current ATR for a symbol.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            period: ATR period

        Returns:
            Current ATR value
        """
        df = self.fetch_ohlcv(symbol, timeframe, period + 10)
        if df.empty or len(df) < period:
            return None

        # Calculate True Range
        df['prev_close'] = df['close'].shift(1)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['prev_close'])
        df['tr3'] = abs(df['low'] - df['prev_close'])
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

        # Calculate ATR
        atr = df['tr'].rolling(window=period).mean().iloc[-1]

        return float(atr)

    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes."""
        tf_map = {
            '1m': 1, '2m': 2, '3m': 3, '4m': 4, '5m': 5,
            '6m': 6, '10m': 10, '12m': 12, '15m': 15, '20m': 20, '30m': 30,
            '1h': 60, '2h': 120, '3h': 180, '4h': 240, '6h': 360, '8h': 480, '12h': 720,
            '1d': 1440, '1w': 10080, '1M': 43200
        }
        return tf_map.get(timeframe, 60)

    def _get_point(self, symbol: str) -> float:
        """Get point value for a symbol."""
        info = self.connector.get_symbol_info(symbol)
        if info:
            return info['point']
        return 0.00001  # Default for forex pairs

    def get_trading_session(self, symbol: str) -> Optional[Dict]:
        """
        Get current trading session information.

        Args:
            symbol: Trading symbol

        Returns:
            Session information
        """
        if not self.connector.is_connected():
            return None

        info = self.connector.get_symbol_info(symbol)
        if not info:
            return None

        return {
            'symbol': symbol,
            'session_open': info['session_open'],
            'session_close': info['session_close'],
            'trade_mode': info['trade_mode'],
        }
