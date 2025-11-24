"""
MetaTrader 5 Connection Manager.
"""

import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pytz
from loguru import logger

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    logger.warning("MetaTrader5 package not installed. Install with: pip install MetaTrader5")


class MT5Connector:
    """
    Manages connection to MetaTrader 5 terminal.
    Handles initialization, authentication, and connection state.
    """

    # Timeframe mapping
    TIMEFRAMES = {
        '1m': mt5.TIMEFRAME_M1 if MT5_AVAILABLE else 1,
        '2m': mt5.TIMEFRAME_M2 if MT5_AVAILABLE else 2,
        '3m': mt5.TIMEFRAME_M3 if MT5_AVAILABLE else 3,
        '4m': mt5.TIMEFRAME_M4 if MT5_AVAILABLE else 4,
        '5m': mt5.TIMEFRAME_M5 if MT5_AVAILABLE else 5,
        '6m': mt5.TIMEFRAME_M6 if MT5_AVAILABLE else 6,
        '10m': mt5.TIMEFRAME_M10 if MT5_AVAILABLE else 10,
        '12m': mt5.TIMEFRAME_M12 if MT5_AVAILABLE else 12,
        '15m': mt5.TIMEFRAME_M15 if MT5_AVAILABLE else 15,
        '20m': mt5.TIMEFRAME_M20 if MT5_AVAILABLE else 20,
        '30m': mt5.TIMEFRAME_M30 if MT5_AVAILABLE else 30,
        '1h': mt5.TIMEFRAME_H1 if MT5_AVAILABLE else 16385,
        '2h': mt5.TIMEFRAME_H2 if MT5_AVAILABLE else 16386,
        '3h': mt5.TIMEFRAME_H3 if MT5_AVAILABLE else 16387,
        '4h': mt5.TIMEFRAME_H4 if MT5_AVAILABLE else 16388,
        '6h': mt5.TIMEFRAME_H6 if MT5_AVAILABLE else 16390,
        '8h': mt5.TIMEFRAME_H8 if MT5_AVAILABLE else 16392,
        '12h': mt5.TIMEFRAME_H12 if MT5_AVAILABLE else 16396,
        '1d': mt5.TIMEFRAME_D1 if MT5_AVAILABLE else 16408,
        '1w': mt5.TIMEFRAME_W1 if MT5_AVAILABLE else 32769,
        '1M': mt5.TIMEFRAME_MN1 if MT5_AVAILABLE else 49153,
    }

    # Order type mapping
    ORDER_TYPES = {
        'buy': mt5.ORDER_TYPE_BUY if MT5_AVAILABLE else 0,
        'sell': mt5.ORDER_TYPE_SELL if MT5_AVAILABLE else 1,
        'buy_limit': mt5.ORDER_TYPE_BUY_LIMIT if MT5_AVAILABLE else 2,
        'sell_limit': mt5.ORDER_TYPE_SELL_LIMIT if MT5_AVAILABLE else 3,
        'buy_stop': mt5.ORDER_TYPE_BUY_STOP if MT5_AVAILABLE else 4,
        'sell_stop': mt5.ORDER_TYPE_SELL_STOP if MT5_AVAILABLE else 5,
        'buy_stop_limit': mt5.ORDER_TYPE_BUY_STOP_LIMIT if MT5_AVAILABLE else 6,
        'sell_stop_limit': mt5.ORDER_TYPE_SELL_STOP_LIMIT if MT5_AVAILABLE else 7,
    }

    def __init__(
        self,
        path: Optional[str] = None,
        login: Optional[int] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
        timeout: int = 60000,
        portable: bool = False
    ):
        """
        Initialize MT5 connector.

        Args:
            path: Path to MetaTrader 5 terminal executable
            login: Trading account number
            password: Trading account password
            server: Broker server name
            timeout: Connection timeout in milliseconds
            portable: Use portable mode
        """
        if not MT5_AVAILABLE:
            raise ImportError("MetaTrader5 package is required. Install with: pip install MetaTrader5")

        self.path = path
        self.login = login
        self.password = password
        self.server = server
        self.timeout = timeout
        self.portable = portable

        self._connected = False
        self._account_info = None
        self._terminal_info = None

        logger.info("MT5Connector initialized")

    def connect(self) -> bool:
        """
        Connect to MetaTrader 5 terminal.

        Returns:
            True if connected successfully
        """
        if self._connected:
            logger.info("Already connected to MT5")
            return True

        # Initialize MT5
        init_params = {}
        if self.path:
            init_params['path'] = self.path
        if self.login:
            init_params['login'] = self.login
        if self.password:
            init_params['password'] = self.password
        if self.server:
            init_params['server'] = self.server
        if self.timeout:
            init_params['timeout'] = self.timeout
        if self.portable:
            init_params['portable'] = self.portable

        if init_params:
            initialized = mt5.initialize(**init_params)
        else:
            initialized = mt5.initialize()

        if not initialized:
            error_code = mt5.last_error()
            logger.error(f"MT5 initialization failed. Error code: {error_code}")
            return False

        self._connected = True
        self._terminal_info = mt5.terminal_info()
        self._account_info = mt5.account_info()

        logger.info(f"Connected to MT5: {self._terminal_info.name if self._terminal_info else 'Unknown'}")
        logger.info(f"Account: {self._account_info.login if self._account_info else 'Unknown'}")
        logger.info(f"Balance: {self._account_info.balance if self._account_info else 0}")

        return True

    def disconnect(self) -> None:
        """Disconnect from MetaTrader 5."""
        if self._connected:
            mt5.shutdown()
            self._connected = False
            logger.info("Disconnected from MT5")

    def is_connected(self) -> bool:
        """Check if connected to MT5."""
        if not self._connected:
            return False

        # Verify connection is still active
        terminal_info = mt5.terminal_info()
        if terminal_info is None:
            self._connected = False
            return False

        return terminal_info.connected

    def reconnect(self, max_retries: int = 3, delay: int = 5) -> bool:
        """
        Reconnect to MT5 with retries.

        Args:
            max_retries: Maximum reconnection attempts
            delay: Delay between attempts in seconds

        Returns:
            True if reconnected successfully
        """
        self.disconnect()

        for attempt in range(max_retries):
            logger.info(f"Reconnection attempt {attempt + 1}/{max_retries}")
            if self.connect():
                return True
            time.sleep(delay)

        logger.error("Failed to reconnect to MT5")
        return False

    def get_account_info(self) -> Optional[Dict]:
        """
        Get trading account information.

        Returns:
            Account information dictionary
        """
        if not self.is_connected():
            logger.error("Not connected to MT5")
            return None

        account = mt5.account_info()
        if account is None:
            return None

        return {
            'login': account.login,
            'trade_mode': account.trade_mode,
            'leverage': account.leverage,
            'limit_orders': account.limit_orders,
            'margin_so_mode': account.margin_so_mode,
            'trade_allowed': account.trade_allowed,
            'trade_expert': account.trade_expert,
            'balance': account.balance,
            'credit': account.credit,
            'profit': account.profit,
            'equity': account.equity,
            'margin': account.margin,
            'margin_free': account.margin_free,
            'margin_level': account.margin_level,
            'margin_so_call': account.margin_so_call,
            'margin_so_so': account.margin_so_so,
            'currency': account.currency,
            'server': account.server,
            'name': account.name,
            'company': account.company,
        }

    def get_terminal_info(self) -> Optional[Dict]:
        """
        Get terminal information.

        Returns:
            Terminal information dictionary
        """
        if not self.is_connected():
            return None

        terminal = mt5.terminal_info()
        if terminal is None:
            return None

        return {
            'community_account': terminal.community_account,
            'community_connection': terminal.community_connection,
            'connected': terminal.connected,
            'dlls_allowed': terminal.dlls_allowed,
            'trade_allowed': terminal.trade_allowed,
            'tradeapi_disabled': terminal.tradeapi_disabled,
            'email_enabled': terminal.email_enabled,
            'ftp_enabled': terminal.ftp_enabled,
            'notifications_enabled': terminal.notifications_enabled,
            'mqid': terminal.mqid,
            'build': terminal.build,
            'maxbars': terminal.maxbars,
            'codepage': terminal.codepage,
            'ping_last': terminal.ping_last,
            'community_balance': terminal.community_balance,
            'retransmission': terminal.retransmission,
            'company': terminal.company,
            'name': terminal.name,
            'language': terminal.language,
            'path': terminal.path,
            'data_path': terminal.data_path,
            'commondata_path': terminal.commondata_path,
        }

    def get_symbols(self, group: Optional[str] = None) -> List[str]:
        """
        Get list of available symbols.

        Args:
            group: Filter by symbol group (e.g., '*USD*', 'Forex*')

        Returns:
            List of symbol names
        """
        if not self.is_connected():
            return []

        if group:
            symbols = mt5.symbols_get(group=group)
        else:
            symbols = mt5.symbols_get()

        if symbols is None:
            return []

        return [s.name for s in symbols]

    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """
        Get detailed symbol information.

        Args:
            symbol: Symbol name

        Returns:
            Symbol information dictionary
        """
        if not self.is_connected():
            return None

        info = mt5.symbol_info(symbol)
        if info is None:
            logger.warning(f"Symbol {symbol} not found")
            return None

        return {
            'name': info.name,
            'description': info.description,
            'path': info.path,
            'point': info.point,
            'digits': info.digits,
            'spread': info.spread,
            'spread_float': info.spread_float,
            'trade_mode': info.trade_mode,
            'trade_calc_mode': info.trade_calc_mode,
            'volume_min': info.volume_min,
            'volume_max': info.volume_max,
            'volume_step': info.volume_step,
            'volume_limit': info.volume_limit,
            'swap_long': info.swap_long,
            'swap_short': info.swap_short,
            'margin_initial': info.margin_initial,
            'margin_maintenance': info.margin_maintenance,
            'contract_size': info.contract_size,
            'tick_value': info.tick_value,
            'tick_size': info.tick_size,
            'trade_stops_level': info.trade_stops_level,
            'trade_freeze_level': info.trade_freeze_level,
            'bid': info.bid,
            'ask': info.ask,
            'last': info.last,
            'session_open': info.session_open,
            'session_close': info.session_close,
            'currency_base': info.currency_base,
            'currency_profit': info.currency_profit,
            'currency_margin': info.currency_margin,
        }

    def select_symbol(self, symbol: str) -> bool:
        """
        Select symbol in Market Watch.

        Args:
            symbol: Symbol name

        Returns:
            True if successful
        """
        if not self.is_connected():
            return False

        # Enable symbol in Market Watch
        selected = mt5.symbol_select(symbol, True)
        if not selected:
            logger.error(f"Failed to select symbol {symbol}")
            return False

        return True

    def get_timeframe(self, timeframe: str) -> int:
        """Convert string timeframe to MT5 timeframe constant."""
        return self.TIMEFRAMES.get(timeframe, mt5.TIMEFRAME_H1 if MT5_AVAILABLE else 16385)

    def get_last_error(self) -> Tuple[int, str]:
        """
        Get last error from MT5.

        Returns:
            Tuple of (error_code, error_description)
        """
        error = mt5.last_error()
        return error

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False
