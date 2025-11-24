"""
Market data fetching and management.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
import ccxt
import ccxt.async_support as ccxt_async
from loguru import logger


class MarketDataFetcher:
    """
    Fetches market data from various exchanges using CCXT.
    Supports both synchronous and asynchronous operations.
    """

    TIMEFRAME_MAP = {
        '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
        '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '12h': '12h',
        '1d': '1d', '1w': '1w', '1M': '1M'
    }

    TIMEFRAME_SECONDS = {
        '1m': 60, '5m': 300, '15m': 900, '30m': 1800,
        '1h': 3600, '2h': 7200, '4h': 14400, '6h': 21600, '12h': 43200,
        '1d': 86400, '1w': 604800, '1M': 2592000
    }

    def __init__(
        self,
        exchange_id: str = 'binance',
        api_key: Optional[str] = None,
        secret: Optional[str] = None,
        testnet: bool = True
    ):
        """
        Initialize the market data fetcher.

        Args:
            exchange_id: Exchange identifier (e.g., 'binance', 'coinbase')
            api_key: API key for the exchange
            secret: API secret for the exchange
            testnet: Whether to use testnet/sandbox mode
        """
        self.exchange_id = exchange_id
        self.api_key = api_key
        self.secret = secret
        self.testnet = testnet

        # Initialize synchronous exchange
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class({
            'apiKey': api_key,
            'secret': secret,
            'sandbox': testnet,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future' if 'future' in exchange_id else 'spot'
            }
        })

        # Async exchange (initialized when needed)
        self._async_exchange = None

        logger.info(f"Initialized MarketDataFetcher for {exchange_id} (testnet={testnet})")

    async def _get_async_exchange(self):
        """Get or create async exchange instance."""
        if self._async_exchange is None:
            exchange_class = getattr(ccxt_async, self.exchange_id)
            self._async_exchange = exchange_class({
                'apiKey': self.api_key,
                'secret': self.secret,
                'sandbox': self.testnet,
                'enableRateLimit': True,
            })
        return self._async_exchange

    async def close(self):
        """Close async exchange connection."""
        if self._async_exchange:
            await self._async_exchange.close()

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = '1h',
        since: Optional[int] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data synchronously.

        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe for candles
            since: Start timestamp in milliseconds
            limit: Maximum number of candles

        Returns:
            DataFrame with OHLCV data
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=since,
                limit=limit
            )

            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df['symbol'] = symbol
            df['timeframe'] = timeframe

            logger.debug(f"Fetched {len(df)} candles for {symbol} ({timeframe})")
            return df

        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            return pd.DataFrame()

    async def fetch_ohlcv_async(
        self,
        symbol: str,
        timeframe: str = '1h',
        since: Optional[int] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data asynchronously.

        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for candles
            since: Start timestamp in milliseconds
            limit: Maximum number of candles

        Returns:
            DataFrame with OHLCV data
        """
        try:
            exchange = await self._get_async_exchange()
            ohlcv = await exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=since,
                limit=limit
            )

            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df['symbol'] = symbol
            df['timeframe'] = timeframe

            return df

        except Exception as e:
            logger.error(f"Error fetching OHLCV async for {symbol}: {e}")
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
            symbol: Trading pair symbol
            timeframe: Timeframe for candles
            days: Number of days of historical data

        Returns:
            DataFrame with historical OHLCV data
        """
        all_data = []
        tf_seconds = self.TIMEFRAME_SECONDS.get(timeframe, 3600)
        candles_per_request = 1000
        candles_needed = (days * 86400) // tf_seconds

        since = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)

        while candles_needed > 0:
            limit = min(candles_per_request, candles_needed)
            df = self.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)

            if df.empty:
                break

            all_data.append(df)
            candles_needed -= len(df)

            if len(df) < limit:
                break

            # Update since to last timestamp
            since = int(df.index[-1].timestamp() * 1000) + (tf_seconds * 1000)
            time.sleep(self.exchange.rateLimit / 1000)

        if all_data:
            result = pd.concat(all_data)
            result = result[~result.index.duplicated(keep='first')]
            result.sort_index(inplace=True)
            logger.info(f"Fetched {len(result)} historical candles for {symbol}")
            return result

        return pd.DataFrame()

    async def fetch_multiple_symbols(
        self,
        symbols: List[str],
        timeframe: str = '1h',
        limit: int = 1000
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple symbols concurrently.

        Args:
            symbols: List of trading pair symbols
            timeframe: Timeframe for candles
            limit: Maximum number of candles per symbol

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        tasks = [
            self.fetch_ohlcv_async(symbol, timeframe, limit=limit)
            for symbol in symbols
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        data = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Error fetching {symbol}: {result}")
                data[symbol] = pd.DataFrame()
            else:
                data[symbol] = result

        return data

    def fetch_ticker(self, symbol: str) -> Dict:
        """
        Fetch current ticker data.

        Args:
            symbol: Trading pair symbol

        Returns:
            Ticker data dictionary
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return {
                'symbol': symbol,
                'last': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'high': ticker['high'],
                'low': ticker['low'],
                'volume': ticker['baseVolume'],
                'change': ticker['percentage'],
                'timestamp': datetime.utcnow()
            }
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            return {}

    def fetch_order_book(
        self,
        symbol: str,
        limit: int = 20
    ) -> Dict:
        """
        Fetch order book data.

        Args:
            symbol: Trading pair symbol
            limit: Depth of order book

        Returns:
            Order book dictionary with bids and asks
        """
        try:
            order_book = self.exchange.fetch_order_book(symbol, limit=limit)
            return {
                'symbol': symbol,
                'bids': order_book['bids'],
                'asks': order_book['asks'],
                'timestamp': datetime.utcnow()
            }
        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {e}")
            return {}

    def fetch_trades(
        self,
        symbol: str,
        since: Optional[int] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch recent trades.

        Args:
            symbol: Trading pair symbol
            since: Start timestamp in milliseconds
            limit: Maximum number of trades

        Returns:
            DataFrame with trade data
        """
        try:
            trades = self.exchange.fetch_trades(symbol, since=since, limit=limit)

            df = pd.DataFrame([{
                'timestamp': t['timestamp'],
                'price': t['price'],
                'amount': t['amount'],
                'side': t['side'],
                'cost': t['cost']
            } for t in trades])

            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)

            return df

        except Exception as e:
            logger.error(f"Error fetching trades for {symbol}: {e}")
            return pd.DataFrame()

    def get_markets(self) -> List[Dict]:
        """
        Get list of available markets.

        Returns:
            List of market information dictionaries
        """
        try:
            markets = self.exchange.load_markets()
            return [
                {
                    'symbol': symbol,
                    'base': market['base'],
                    'quote': market['quote'],
                    'active': market['active'],
                    'type': market['type']
                }
                for symbol, market in markets.items()
                if market['active']
            ]
        except Exception as e:
            logger.error(f"Error loading markets: {e}")
            return []

    def calculate_imbalance(
        self,
        symbol: str,
        depth: int = 10
    ) -> float:
        """
        Calculate order book imbalance.

        Args:
            symbol: Trading pair symbol
            depth: Number of levels to consider

        Returns:
            Imbalance ratio (-1 to 1, positive = more bids)
        """
        order_book = self.fetch_order_book(symbol, limit=depth)

        if not order_book:
            return 0.0

        bid_volume = sum(b[1] for b in order_book['bids'][:depth])
        ask_volume = sum(a[1] for a in order_book['asks'][:depth])
        total_volume = bid_volume + ask_volume

        if total_volume == 0:
            return 0.0

        return (bid_volume - ask_volume) / total_volume

    def get_funding_rate(self, symbol: str) -> Optional[float]:
        """
        Get current funding rate for perpetual futures.

        Args:
            symbol: Trading pair symbol

        Returns:
            Current funding rate or None
        """
        try:
            if hasattr(self.exchange, 'fetch_funding_rate'):
                funding = self.exchange.fetch_funding_rate(symbol)
                return funding.get('fundingRate')
        except Exception as e:
            logger.debug(f"Funding rate not available for {symbol}: {e}")
        return None
