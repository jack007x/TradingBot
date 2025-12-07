"""
Data Fetcher Module
====================
Fetches historical and live OHLCV data from MetaTrader 5.

Usage:
    from data import DataFetcher
    from config import get_config

    config = get_config()
    fetcher = DataFetcher(config)

    # Fetch historical data
    df = fetcher.fetch_historical(
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2024, 12, 1)
    )

    # Fetch latest bars for live trading
    df_latest = fetcher.fetch_latest_bars(n_bars=100)
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple
import pandas as pd
import numpy as np

# MT5 import with fallback for development without MT5
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None

from config.config_loader import Config, get_timeframe_minutes

logger = logging.getLogger(__name__)


# =============================================================================
# MT5 Timeframe Mapping
# =============================================================================

MT5_TIMEFRAME_MAP = {
    'M1': mt5.TIMEFRAME_M1 if MT5_AVAILABLE else 1,
    'M5': mt5.TIMEFRAME_M5 if MT5_AVAILABLE else 5,
    'M15': mt5.TIMEFRAME_M15 if MT5_AVAILABLE else 15,
    'M30': mt5.TIMEFRAME_M30 if MT5_AVAILABLE else 30,
    'H1': mt5.TIMEFRAME_H1 if MT5_AVAILABLE else 60,
    'H4': mt5.TIMEFRAME_H4 if MT5_AVAILABLE else 240,
    'D1': mt5.TIMEFRAME_D1 if MT5_AVAILABLE else 1440,
}


# =============================================================================
# Data Fetcher Class
# =============================================================================

class DataFetcher:
    """
    Fetches OHLCV data from MetaTrader 5.

    Handles:
    - Historical data download
    - Live bar fetching
    - Timezone conversion (MT5 uses broker time, we convert to UTC)
    - Error handling and retries
    """

    def __init__(self, config: Config):
        """
        Initialize data fetcher.

        Args:
            config: Configuration object
        """
        self.config = config
        self.symbol = config.trading.symbol
        self.timeframe = config.trading.timeframe
        self.mt5_timeframe = MT5_TIMEFRAME_MAP.get(self.timeframe)

        self._connected = False

    def connect(self) -> bool:
        """
        Connect to MetaTrader 5 terminal.

        Returns:
            bool: True if connected successfully
        """
        if not MT5_AVAILABLE:
            logger.warning("MetaTrader5 library not available. Using demo mode.")
            return False

        if self._connected:
            return True

        # Initialize MT5
        if not mt5.initialize(path=self.config.broker.terminal_path):
            error = mt5.last_error()
            logger.error(f"MT5 initialization failed: {error}")
            return False

        # Login if credentials provided
        if self.config.broker.login and self.config.broker.password:
            try:
                login = int(self.config.broker.login)
                authorized = mt5.login(
                    login=login,
                    password=self.config.broker.password,
                    server=self.config.broker.server,
                    timeout=self.config.broker.timeout_ms
                )
                if not authorized:
                    error = mt5.last_error()
                    logger.error(f"MT5 login failed: {error}")
                    mt5.shutdown()
                    return False
            except (ValueError, TypeError) as e:
                logger.error(f"Invalid login credentials: {e}")
                return False

        self._connected = True
        logger.info(f"Connected to MT5 terminal")

        # Verify symbol is available
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            logger.error(f"Symbol {self.symbol} not found in MT5")
            self.disconnect()
            return False

        if not symbol_info.visible:
            if not mt5.symbol_select(self.symbol, True):
                logger.error(f"Failed to select symbol {self.symbol}")
                self.disconnect()
                return False

        logger.info(f"Symbol {self.symbol} selected successfully")
        return True

    def disconnect(self) -> None:
        """Disconnect from MetaTrader 5."""
        if MT5_AVAILABLE and self._connected:
            mt5.shutdown()
            self._connected = False
            logger.info("Disconnected from MT5")

    def fetch_historical(
            self,
            start_date: datetime,
            end_date: Optional[datetime] = None,
            include_current_bar: bool = False
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data.

        Args:
            start_date: Start date for data fetch
            end_date: End date (default: now)
            include_current_bar: Include incomplete current bar

        Returns:
            pd.DataFrame: OHLCV data with columns:
                - time (datetime, UTC)
                - open, high, low, close (float)
                - volume (int)
                - spread (int)
        """
        if end_date is None:
            end_date = datetime.now()

        # Clean datetime objects for MT5 (remove microseconds, ensure naive datetime)
        start_date_clean = datetime(
            start_date.year, start_date.month, start_date.day,
            start_date.hour, start_date.minute, 0
        )
        end_date_clean = datetime(
            end_date.year, end_date.month, end_date.day,
            end_date.hour, end_date.minute, 0
        )

        logger.info(f"Fetching {self.symbol} {self.timeframe} data: {start_date_clean} to {end_date_clean}")

        if not MT5_AVAILABLE:
            logger.warning("MT5 not available, generating demo data")
            return self._generate_demo_data(start_date_clean, end_date_clean)

        if not self._connected:
            if not self.connect():
                raise ConnectionError("Failed to connect to MT5")

        # Fetch data from MT5 using cleaned datetime
        rates = mt5.copy_rates_range(
            self.symbol,
            self.mt5_timeframe,
            start_date_clean,
            end_date_clean
        )

        if rates is None or len(rates) == 0:
            error = mt5.last_error()
            logger.warning(f"No data returned from MT5: {error}")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(rates)

        # Convert time to datetime and set to UTC
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)

        # Rename columns to standard names
        df = df.rename(columns={
            'tick_volume': 'volume',
            'real_volume': 'real_volume'
        })

        # Select and order columns
        columns = ['time', 'open', 'high', 'low', 'close', 'volume', 'spread']
        df = df[[c for c in columns if c in df.columns]]

        # Remove current incomplete bar if requested
        if not include_current_bar and len(df) > 0:
            tf_minutes = get_timeframe_minutes(self.timeframe)
            now = datetime.now(df['time'].dt.tz)
            current_bar_start = now.replace(
                minute=(now.minute // tf_minutes) * tf_minutes,
                second=0, microsecond=0
            )
            df = df[df['time'] < current_bar_start]

        # Sort by time
        df = df.sort_values('time').reset_index(drop=True)

        logger.info(f"Fetched {len(df)} bars")
        return df

    def fetch_latest_bars(self, n_bars: int = 100) -> pd.DataFrame:
        """
        Fetch the latest N bars.

        Args:
            n_bars: Number of bars to fetch

        Returns:
            pd.DataFrame: Latest OHLCV data
        """
        if not MT5_AVAILABLE:
            end_date = datetime.now()
            tf_minutes = get_timeframe_minutes(self.timeframe)
            start_date = end_date - timedelta(minutes=tf_minutes * n_bars * 2)
            df = self._generate_demo_data(start_date, end_date)
            return df.tail(n_bars)

        if not self._connected:
            if not self.connect():
                raise ConnectionError("Failed to connect to MT5")

        rates = mt5.copy_rates_from_pos(
            self.symbol,
            self.mt5_timeframe,
            0,  # Start from current bar
            n_bars
        )

        if rates is None or len(rates) == 0:
            error = mt5.last_error()
            logger.warning(f"No data returned from MT5: {error}")
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df = df.rename(columns={'tick_volume': 'volume'})

        columns = ['time', 'open', 'high', 'low', 'close', 'volume', 'spread']
        df = df[[c for c in columns if c in df.columns]]

        return df.sort_values('time').reset_index(drop=True)

    def fetch_current_price(self) -> Tuple[float, float, float]:
        """
        Fetch current bid/ask prices.

        Returns:
            Tuple[float, float, float]: (bid, ask, spread_points)
        """
        if not MT5_AVAILABLE:
            # Demo prices for XAUUSD around 2000
            bid = 2000.0 + np.random.randn() * 5
            ask = bid + 0.30  # ~30 points spread
            return bid, ask, 30.0

        if not self._connected:
            if not self.connect():
                raise ConnectionError("Failed to connect to MT5")

        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            raise ValueError(f"Failed to get tick for {self.symbol}")

        spread_points = (tick.ask - tick.bid) * 100  # For gold, 1 pip = 0.01
        return tick.bid, tick.ask, spread_points

    def find_gold_symbol(self) -> Optional[str]:
        """
        Try to find the gold symbol in the broker.

        Different brokers use different names: XAUUSD, GOLD, XAUUSDm, etc.

        Returns:
            str: Found symbol name or None
        """
        if not MT5_AVAILABLE:
            return "XAUUSD"

        if not self._connected:
            if not self.connect():
                return None

        # Common gold symbol names
        possible_names = [
            "XAUUSD", "GOLD", "XAUUSDm", "XAUUSD.raw", "XAUUSD.",
            "GOLDm", "GOLD.raw", "XAU/USD", "XAUUSDpro"
        ]

        # Get all symbols
        symbols = mt5.symbols_get()
        if symbols is None:
            return None

        symbol_names = [s.name for s in symbols]

        # Try exact match first
        for name in possible_names:
            if name in symbol_names:
                logger.info(f"Found gold symbol: {name}")
                return name

        # Try partial match
        for sym in symbol_names:
            if "XAU" in sym.upper() or "GOLD" in sym.upper():
                logger.info(f"Found gold symbol (partial match): {sym}")
                return sym

        # Log available symbols for debugging
        logger.warning("Could not find gold symbol. Available symbols with 'XAU' or 'GOLD':")
        for sym in symbol_names:
            if "XAU" in sym.upper() or "GOLD" in sym.upper():
                logger.warning(f"  - {sym}")

        return None

    def list_available_symbols(self, filter_text: str = "") -> list:
        """
        List available trading symbols.

        Args:
            filter_text: Filter symbols containing this text

        Returns:
            list: Available symbol names
        """
        if not MT5_AVAILABLE:
            return ["XAUUSD"]

        if not self._connected:
            if not self.connect():
                return []

        symbols = mt5.symbols_get()
        if symbols is None:
            return []

        names = [s.name for s in symbols if s.visible]

        if filter_text:
            filter_upper = filter_text.upper()
            names = [n for n in names if filter_upper in n.upper()]

        return sorted(names)

    def get_symbol_info(self) -> dict:
        """
        Get symbol information (contract specs).

        Returns:
            dict: Symbol info including pip value, lot size, etc.
        """
        if not MT5_AVAILABLE:
            # Demo info for XAUUSD
            return {
                'symbol': self.symbol,
                'point': 0.01,
                'digits': 2,
                'trade_contract_size': 100.0,  # 1 lot = 100 oz
                'volume_min': 0.01,
                'volume_max': 100.0,
                'volume_step': 0.01,
                'trade_tick_value': 1.0,  # $1 per pip per lot
                'trade_tick_size': 0.01,
            }

        if not self._connected:
            if not self.connect():
                raise ConnectionError("Failed to connect to MT5")

        info = mt5.symbol_info(self.symbol)
        if info is None:
            raise ValueError(f"Failed to get info for {self.symbol}")

        return {
            'symbol': info.name,
            'point': info.point,
            'digits': info.digits,
            'trade_contract_size': info.trade_contract_size,
            'volume_min': info.volume_min,
            'volume_max': info.volume_max,
            'volume_step': info.volume_step,
            'trade_tick_value': info.trade_tick_value,
            'trade_tick_size': info.trade_tick_size,
        }

    def _generate_demo_data(
            self,
            start_date: datetime,
            end_date: datetime
    ) -> pd.DataFrame:
        """
        Generate synthetic OHLCV data for testing when MT5 is not available.

        Uses random walk with realistic XAUUSD characteristics.
        """
        tf_minutes = get_timeframe_minutes(self.timeframe)

        # Generate time index
        times = pd.date_range(start=start_date, end=end_date, freq=f'{tf_minutes}min', tz='UTC')

        if len(times) == 0:
            return pd.DataFrame()

        # Random walk for price (starting around 2000 for gold)
        np.random.seed(42)  # Reproducible for testing
        n = len(times)

        # Generate returns with slight drift and volatility clustering
        returns = np.random.randn(n) * 0.001  # ~0.1% per bar
        returns = np.cumsum(returns)

        # Base price around 2000 (typical gold price)
        base_price = 2000.0
        close_prices = base_price * (1 + returns)

        # Generate OHLC from close
        volatility = close_prices * 0.001  # 0.1% volatility per bar
        high_prices = close_prices + np.abs(np.random.randn(n)) * volatility
        low_prices = close_prices - np.abs(np.random.randn(n)) * volatility
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = close_prices[0]

        # Ensure OHLC consistency
        high_prices = np.maximum.reduce([open_prices, high_prices, close_prices])
        low_prices = np.minimum.reduce([open_prices, low_prices, close_prices])

        # Volume (random, higher during London/NY sessions)
        hours = times.hour
        volume = np.random.randint(100, 1000, n)
        volume = np.where((hours >= 8) & (hours <= 16), volume * 2, volume)

        df = pd.DataFrame({
            'time': times,
            'open': np.round(open_prices, 2),
            'high': np.round(high_prices, 2),
            'low': np.round(low_prices, 2),
            'close': np.round(close_prices, 2),
            'volume': volume,
            'spread': np.random.randint(20, 40, n),
        })

        return df


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from config import get_config

    config = get_config()
    fetcher = DataFetcher(config)

    # Test historical fetch (demo mode)
    start = datetime(2024, 1, 1)
    end = datetime(2024, 6, 1)

    print("Fetching historical data...")
    df = fetcher.fetch_historical(start, end)
    print(f"Fetched {len(df)} bars")
    print(df.head())
    print(df.tail())

    # Test latest bars
    print("\nFetching latest bars...")
    df_latest = fetcher.fetch_latest_bars(10)
    print(df_latest)

    # Test symbol info
    print("\nSymbol info:")
    info = fetcher.get_symbol_info()
    for k, v in info.items():
        print(f"  {k}: {v}")
