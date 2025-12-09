"""
MT5 Broker Module
==================
Handles trade execution via MetaTrader 5 API.

Usage:
    from execution import MT5Broker
    from config import get_config

    config = get_config()
    broker = MT5Broker(config)

    # Connect
    broker.connect()

    # Place order
    result = broker.place_market_order(
        direction=1,  # BUY
        lot_size=0.1,
        sl=1990.0,
        tp=2010.0
    )

    # Close position
    broker.close_position(ticket=12345)
"""

import logging
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any

# MT5 import with fallback
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None

from config.config_loader import Config

logger = logging.getLogger(__name__)


@dataclass
class OrderResult:
    """Result of an order operation."""
    success: bool
    ticket: Optional[int]
    message: str
    order_type: str
    price: float
    volume: float
    sl: float
    tp: float
    timestamp: datetime


@dataclass
class Position:
    """Open position information."""
    ticket: int
    symbol: str
    type: int  # 0 = BUY, 1 = SELL
    volume: float
    price_open: float
    price_current: float
    sl: float
    tp: float
    profit: float
    swap: float
    magic: int
    time_open: datetime
    comment: str


@dataclass
class AccountInfo:
    """Account information."""
    login: int
    balance: float
    equity: float
    margin: float
    free_margin: float
    margin_level: float
    profit: float
    server: str
    currency: str
    leverage: int


class MT5Broker:
    """
    MetaTrader 5 broker interface.

    Handles:
    - Connection management
    - Order placement (market orders)
    - Position management (close, modify)
    - Account information
    - Error handling and retries
    """

    def __init__(self, config: Config):
        """
        Initialize MT5 broker.

        Args:
            config: Configuration object
        """
        self.config = config
        self.symbol = config.trading.symbol
        self.magic_number = config.trading.magic_number

        # Connection settings
        self.terminal_path = config.broker.terminal_path
        self.timeout = config.broker.timeout_ms
        self.retry_attempts = config.broker.retry_attempts
        self.retry_delay = config.broker.retry_delay_seconds

        self._connected = False
        self._demo_mode = not MT5_AVAILABLE

        # Symbol info cache
        self._symbol_info: Optional[Dict] = None

    def connect(self) -> bool:
        """
        Connect to MT5 terminal.

        Returns:
            bool: True if connected successfully
        """
        if self._demo_mode:
            logger.info("Running in DEMO mode (MT5 not available)")
            self._connected = True
            return True

        if self._connected:
            return True

        # Initialize MT5
        for attempt in range(self.retry_attempts):
            try:
                if not mt5.initialize(path=self.terminal_path, timeout=self.timeout):
                    error = mt5.last_error()
                    logger.warning(f"MT5 init attempt {attempt + 1} failed: {error}")
                    time.sleep(self.retry_delay)
                    continue

                # Login if credentials provided
                login = self.config.broker.login
                password = self.config.broker.password
                server = self.config.broker.server

                if login and password and server:
                    try:
                        login_int = int(login)
                        authorized = mt5.login(
                            login=login_int,
                            password=password,
                            server=server,
                            timeout=self.timeout
                        )
                        if not authorized:
                            error = mt5.last_error()
                            logger.error(f"MT5 login failed: {error}")
                            mt5.shutdown()
                            continue
                    except ValueError as e:
                        logger.error(f"Invalid login value: {e}")
                        mt5.shutdown()
                        return False

                # Verify symbol
                if not self._select_symbol():
                    mt5.shutdown()
                    continue

                self._connected = True
                logger.info("Connected to MT5 successfully")
                return True

            except Exception as e:
                logger.error(f"MT5 connection error: {e}")
                time.sleep(self.retry_delay)

        return False

    def disconnect(self) -> None:
        """Disconnect from MT5."""
        if MT5_AVAILABLE and self._connected:
            mt5.shutdown()
        self._connected = False
        logger.info("Disconnected from MT5")

    def _select_symbol(self) -> bool:
        """Ensure symbol is selected and visible."""
        if self._demo_mode:
            return True

        info = mt5.symbol_info(self.symbol)
        if info is None:
            logger.error(f"Symbol {self.symbol} not found")
            return False

        if not info.visible:
            if not mt5.symbol_select(self.symbol, True):
                logger.error(f"Failed to select {self.symbol}")
                return False

        # Cache symbol info
        self._symbol_info = {
            'point': info.point,
            'digits': info.digits,
            'spread': info.spread,
            'volume_min': info.volume_min,
            'volume_max': info.volume_max,
            'volume_step': info.volume_step,
            'trade_contract_size': info.trade_contract_size,
        }

        return True

    def get_account_info(self) -> AccountInfo:
        """
        Get current account information.

        Returns:
            AccountInfo: Account details
        """
        if self._demo_mode:
            return AccountInfo(
                login=12345,
                balance=10000.0,
                equity=10000.0,
                margin=0.0,
                free_margin=10000.0,
                margin_level=0.0,
                profit=0.0,
                server="Demo",
                currency="USD",
                leverage=100
            )

        if not self._connected:
            raise ConnectionError("Not connected to MT5")

        info = mt5.account_info()
        if info is None:
            raise RuntimeError(f"Failed to get account info: {mt5.last_error()}")

        return AccountInfo(
            login=info.login,
            balance=info.balance,
            equity=info.equity,
            margin=info.margin,
            free_margin=info.margin_free,
            margin_level=info.margin_level,
            profit=info.profit,
            server=info.server,
            currency=info.currency,
            leverage=info.leverage
        )

    def get_current_price(self) -> Tuple[float, float]:
        """
        Get current bid/ask prices.

        Returns:
            Tuple[bid, ask]
        """
        if self._demo_mode:
            # Simulate price around 2000
            import random
            bid = 2000.0 + random.uniform(-5, 5)
            ask = bid + 0.30
            return bid, ask

        if not self._connected:
            raise ConnectionError("Not connected to MT5")

        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            raise RuntimeError(f"Failed to get tick: {mt5.last_error()}")

        return tick.bid, tick.ask

    def place_market_order(
            self,
            direction: int,
            lot_size: float,
            sl: float,
            tp: float,
            comment: str = ""
    ) -> OrderResult:
        """
        Place a market order.

        Args:
            direction: 1 for BUY, -1 for SELL
            lot_size: Volume in lots
            sl: Stop loss price
            tp: Take profit price
            comment: Order comment

        Returns:
            OrderResult: Order result
        """
        timestamp = datetime.now()

        # Validate inputs
        lot_size = round(lot_size, 2)
        if lot_size < self.config.risk.min_lot_size:
            return OrderResult(
                success=False,
                ticket=None,
                message=f"Lot size too small: {lot_size}",
                order_type="MARKET",
                price=0,
                volume=lot_size,
                sl=sl,
                tp=tp,
                timestamp=timestamp
            )

        if self._demo_mode:
            # Simulate order in demo mode
            import random
            ticket = random.randint(100000, 999999)
            bid, ask = self.get_current_price()
            price = ask if direction == 1 else bid

            logger.info(f"DEMO: {'BUY' if direction == 1 else 'SELL'} {lot_size} @ {price}")

            return OrderResult(
                success=True,
                ticket=ticket,
                message="Demo order placed",
                order_type="BUY" if direction == 1 else "SELL",
                price=price,
                volume=lot_size,
                sl=sl,
                tp=tp,
                timestamp=timestamp
            )

        if not self._connected:
            return OrderResult(
                success=False,
                ticket=None,
                message="Not connected to MT5",
                order_type="MARKET",
                price=0,
                volume=lot_size,
                sl=sl,
                tp=tp,
                timestamp=timestamp
            )

        # Get current prices
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return OrderResult(
                success=False,
                ticket=None,
                message=f"Failed to get price: {mt5.last_error()}",
                order_type="MARKET",
                price=0,
                volume=lot_size,
                sl=sl,
                tp=tp,
                timestamp=timestamp
            )

        # Prepare order request
        order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
        price = tick.ask if direction == 1 else tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,  # Max slippage in points
            "magic": self.magic_number,
            "comment": comment or "AI_BOT",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # Send order
        result = mt5.order_send(request)

        if result is None:
            error = mt5.last_error()
            return OrderResult(
                success=False,
                ticket=None,
                message=f"Order failed: {error}",
                order_type="BUY" if direction == 1 else "SELL",
                price=price,
                volume=lot_size,
                sl=sl,
                tp=tp,
                timestamp=timestamp
            )

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return OrderResult(
                success=False,
                ticket=None,
                message=f"Order rejected: {result.comment} (code: {result.retcode})",
                order_type="BUY" if direction == 1 else "SELL",
                price=price,
                volume=lot_size,
                sl=sl,
                tp=tp,
                timestamp=timestamp
            )

        logger.info(f"Order placed: ticket={result.order}, price={result.price}")

        return OrderResult(
            success=True,
            ticket=result.order,
            message="Order executed",
            order_type="BUY" if direction == 1 else "SELL",
            price=result.price,
            volume=result.volume,
            sl=sl,
            tp=tp,
            timestamp=timestamp
        )

    def close_position(
            self,
            ticket: int,
            comment: str = ""
    ) -> OrderResult:
        """
        Close an open position.

        Args:
            ticket: Position ticket
            comment: Close comment

        Returns:
            OrderResult: Close result
        """
        timestamp = datetime.now()

        if self._demo_mode:
            logger.info(f"DEMO: Closing position {ticket}")
            return OrderResult(
                success=True,
                ticket=ticket,
                message="Demo position closed",
                order_type="CLOSE",
                price=2000.0,
                volume=0,
                sl=0,
                tp=0,
                timestamp=timestamp
            )

        if not self._connected:
            return OrderResult(
                success=False,
                ticket=ticket,
                message="Not connected to MT5",
                order_type="CLOSE",
                price=0,
                volume=0,
                sl=0,
                tp=0,
                timestamp=timestamp
            )

        # Get position info
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return OrderResult(
                success=False,
                ticket=ticket,
                message="Position not found",
                order_type="CLOSE",
                price=0,
                volume=0,
                sl=0,
                tp=0,
                timestamp=timestamp
            )

        pos = position[0]

        # Get current price
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return OrderResult(
                success=False,
                ticket=ticket,
                message=f"Failed to get price: {mt5.last_error()}",
                order_type="CLOSE",
                price=0,
                volume=pos.volume,
                sl=0,
                tp=0,
                timestamp=timestamp
            )

        # Opposite order to close
        close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
        price = tick.bid if pos.type == 0 else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": pos.volume,
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": self.magic_number,
            "comment": comment or "CLOSE",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            return OrderResult(
                success=False,
                ticket=ticket,
                message=f"Close failed: {result.comment if result else mt5.last_error()}",
                order_type="CLOSE",
                price=price,
                volume=pos.volume,
                sl=0,
                tp=0,
                timestamp=timestamp
            )

        logger.info(f"Position closed: ticket={ticket}, price={result.price}")

        return OrderResult(
            success=True,
            ticket=ticket,
            message="Position closed",
            order_type="CLOSE",
            price=result.price,
            volume=pos.volume,
            sl=0,
            tp=0,
            timestamp=timestamp
        )

    def modify_position(
            self,
            ticket: int,
            sl: Optional[float] = None,
            tp: Optional[float] = None
    ) -> bool:
        """
        Modify position SL/TP.

        Args:
            ticket: Position ticket
            sl: New stop loss (None to keep current)
            tp: New take profit (None to keep current)

        Returns:
            bool: True if modified successfully
        """
        if self._demo_mode:
            logger.info(f"DEMO: Modifying position {ticket}: SL={sl}, TP={tp}")
            return True

        if not self._connected:
            raise ConnectionError("Not connected to MT5")

        position = mt5.positions_get(ticket=ticket)
        if not position:
            logger.error(f"Position {ticket} not found")
            return False

        pos = position[0]

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": self.symbol,
            "position": ticket,
            "sl": sl if sl is not None else pos.sl,
            "tp": tp if tp is not None else pos.tp,
        }

        result = mt5.order_send(request)

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Modify failed: {result.comment if result else mt5.last_error()}")
            return False

        logger.info(f"Position {ticket} modified: SL={sl}, TP={tp}")
        return True

    def get_open_positions(self) -> List[Position]:
        """
        Get all open positions for our magic number.

        Returns:
            List[Position]: Open positions
        """
        if self._demo_mode:
            return []

        if not self._connected:
            raise ConnectionError("Not connected to MT5")

        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            return []

        result = []
        for pos in positions:
            if pos.magic == self.magic_number:
                result.append(Position(
                    ticket=pos.ticket,
                    symbol=pos.symbol,
                    type=pos.type,
                    volume=pos.volume,
                    price_open=pos.price_open,
                    price_current=pos.price_current,
                    sl=pos.sl,
                    tp=pos.tp,
                    profit=pos.profit,
                    swap=pos.swap,
                    magic=pos.magic,
                    time_open=datetime.fromtimestamp(pos.time),
                    comment=pos.comment
                ))

        return result

    def close_all_positions(self, comment: str = "CLOSE_ALL") -> int:
        """
        Close all open positions.

        Returns:
            int: Number of positions closed
        """
        positions = self.get_open_positions()
        closed = 0

        for pos in positions:
            result = self.close_position(pos.ticket, comment)
            if result.success:
                closed += 1

        logger.info(f"Closed {closed}/{len(positions)} positions")
        return closed


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from config import get_config

    config = get_config()
    broker = MT5Broker(config)

    print("Testing MT5 Broker...")
    print("=" * 60)

    # Connect
    print("\n1. Connecting...")
    connected = broker.connect()
    print(f"   Connected: {connected}")

    if connected:
        # Get account info
        print("\n2. Account Info:")
        account = broker.get_account_info()
        print(f"   Balance: ${account.balance:,.2f}")
        print(f"   Equity: ${account.equity:,.2f}")
        print(f"   Server: {account.server}")

        # Get current price
        print("\n3. Current Price:")
        bid, ask = broker.get_current_price()
        print(f"   Bid: {bid:.2f}, Ask: {ask:.2f}")

        # Test order (demo mode)
        print("\n4. Test Order (Demo):")
        result = broker.place_market_order(
            direction=1,
            lot_size=0.01,
            sl=bid - 10,
            tp=ask + 15
        )
        print(f"   Success: {result.success}")
        print(f"   Ticket: {result.ticket}")
        print(f"   Message: {result.message}")

        # Disconnect
        print("\n5. Disconnecting...")
        broker.disconnect()
        print("   Done")

    print("\n" + "=" * 60)
