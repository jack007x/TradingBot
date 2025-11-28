"""
MetaTrader 5 Order Execution and Trade Management.
"""

import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from loguru import logger

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

from .mt5_connector import MT5Connector


class OrderType(Enum):
    """Order types."""
    BUY = "buy"
    SELL = "sell"
    BUY_LIMIT = "buy_limit"
    SELL_LIMIT = "sell_limit"
    BUY_STOP = "buy_stop"
    SELL_STOP = "sell_stop"


@dataclass
class TradeResult:
    """Trade execution result."""
    success: bool
    order_id: Optional[int] = None
    ticket: Optional[int] = None
    volume: float = 0.0
    price: float = 0.0
    symbol: str = ""
    order_type: str = ""
    comment: str = ""
    error_code: int = 0
    error_message: str = ""


class MT5Trader:
    """
    Handles order execution and trade management in MetaTrader 5.
    Supports market orders, pending orders, and position management.
    """

    # Fill policy mapping
    FILL_POLICIES = {
        'fok': mt5.ORDER_FILLING_FOK if MT5_AVAILABLE else 0,  # Fill or Kill
        'ioc': mt5.ORDER_FILLING_IOC if MT5_AVAILABLE else 1,  # Immediate or Cancel
        'return': mt5.ORDER_FILLING_RETURN if MT5_AVAILABLE else 2,  # Return (partial fill)
    }

    def __init__(
        self,
        connector: MT5Connector,
        magic_number: int = 123456,
        deviation: int = 20,
        fill_policy: str = 'ioc'
    ):
        """
        Initialize MT5 Trader.

        Args:
            connector: MT5Connector instance
            magic_number: Expert Advisor magic number for identifying trades
            deviation: Maximum price deviation in points
            fill_policy: Order fill policy ('fok', 'ioc', 'return')
        """
        self.connector = connector
        self.magic_number = magic_number
        self.deviation = deviation
        self.fill_policy = self.FILL_POLICIES.get(fill_policy, mt5.ORDER_FILLING_IOC if MT5_AVAILABLE else 1)

        logger.info(f"MT5Trader initialized with magic number: {magic_number}")

    def place_market_order(
        self,
        symbol: str,
        order_type: Union[str, OrderType],
        volume: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        comment: str = "AI Trading Bot"
    ) -> TradeResult:
        """
        Place a market order.

        Args:
            symbol: Trading symbol
            order_type: 'buy' or 'sell'
            volume: Trade volume in lots
            stop_loss: Stop loss price
            take_profit: Take profit price
            comment: Order comment

        Returns:
            TradeResult with execution details
        """
        if not self.connector.is_connected():
            return TradeResult(
                success=False,
                error_message="Not connected to MT5"
            )

        # Normalize order type
        if isinstance(order_type, OrderType):
            order_type = order_type.value

        # Select symbol
        if not self.connector.select_symbol(symbol):
            return TradeResult(
                success=False,
                symbol=symbol,
                error_message=f"Failed to select symbol {symbol}"
            )

        # Get symbol info
        symbol_info = self.connector.get_symbol_info(symbol)
        if not symbol_info:
            return TradeResult(
                success=False,
                symbol=symbol,
                error_message=f"Symbol info not available for {symbol}"
            )

        # Validate volume
        volume = self._normalize_volume(volume, symbol_info)

        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return TradeResult(
                success=False,
                symbol=symbol,
                error_message="Failed to get current price"
            )

        # Determine price and order type constant
        if order_type.lower() == 'buy':
            price = tick.ask
            mt5_order_type = mt5.ORDER_TYPE_BUY
        else:
            price = tick.bid
            mt5_order_type = mt5.ORDER_TYPE_SELL

        # Prepare request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5_order_type,
            "price": price,
            "deviation": self.deviation,
            "magic": self.magic_number,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self.fill_policy,
        }

        # Add SL/TP if provided
        if stop_loss:
            request["sl"] = stop_loss
        if take_profit:
            request["tp"] = take_profit

        # Execute order
        result = mt5.order_send(request)

        if result is None:
            error = mt5.last_error()
            return TradeResult(
                success=False,
                symbol=symbol,
                error_code=error[0],
                error_message=f"Order failed: {error}"
            )

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return TradeResult(
                success=False,
                symbol=symbol,
                error_code=result.retcode,
                error_message=f"Order failed: {result.comment}"
            )

        logger.info(f"Order executed: {order_type.upper()} {volume} {symbol} @ {result.price}")

        return TradeResult(
            success=True,
            order_id=result.order,
            ticket=result.deal,
            volume=result.volume,
            price=result.price,
            symbol=symbol,
            order_type=order_type,
            comment=comment
        )

    def place_pending_order(
        self,
        symbol: str,
        order_type: Union[str, OrderType],
        volume: float,
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        expiration: Optional[datetime] = None,
        comment: str = "AI Trading Bot"
    ) -> TradeResult:
        """
        Place a pending order.

        Args:
            symbol: Trading symbol
            order_type: 'buy_limit', 'sell_limit', 'buy_stop', 'sell_stop'
            volume: Trade volume in lots
            price: Order price
            stop_loss: Stop loss price
            take_profit: Take profit price
            expiration: Order expiration time
            comment: Order comment

        Returns:
            TradeResult with order details
        """
        if not self.connector.is_connected():
            return TradeResult(success=False, error_message="Not connected to MT5")

        if isinstance(order_type, OrderType):
            order_type = order_type.value

        if not self.connector.select_symbol(symbol):
            return TradeResult(success=False, error_message=f"Failed to select {symbol}")

        symbol_info = self.connector.get_symbol_info(symbol)
        if not symbol_info:
            return TradeResult(success=False, error_message="Symbol info not available")

        volume = self._normalize_volume(volume, symbol_info)
        price = self._normalize_price(price, symbol_info)

        # Map order type
        order_type_map = {
            'buy_limit': mt5.ORDER_TYPE_BUY_LIMIT,
            'sell_limit': mt5.ORDER_TYPE_SELL_LIMIT,
            'buy_stop': mt5.ORDER_TYPE_BUY_STOP,
            'sell_stop': mt5.ORDER_TYPE_SELL_STOP,
        }

        mt5_order_type = order_type_map.get(order_type.lower())
        if mt5_order_type is None:
            return TradeResult(success=False, error_message=f"Invalid order type: {order_type}")

        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": volume,
            "type": mt5_order_type,
            "price": price,
            "deviation": self.deviation,
            "magic": self.magic_number,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self.fill_policy,
        }

        if stop_loss:
            request["sl"] = self._normalize_price(stop_loss, symbol_info)
        if take_profit:
            request["tp"] = self._normalize_price(take_profit, symbol_info)
        if expiration:
            request["type_time"] = mt5.ORDER_TIME_SPECIFIED
            request["expiration"] = int(expiration.timestamp())

        result = mt5.order_send(request)

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            error = mt5.last_error() if result is None else (result.retcode, result.comment)
            return TradeResult(
                success=False,
                symbol=symbol,
                error_code=error[0] if isinstance(error, tuple) else 0,
                error_message=str(error)
            )

        logger.info(f"Pending order placed: {order_type.upper()} {volume} {symbol} @ {price}")

        return TradeResult(
            success=True,
            order_id=result.order,
            volume=result.volume,
            price=price,
            symbol=symbol,
            order_type=order_type,
            comment=comment
        )

    def modify_position(
        self,
        ticket: int,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> TradeResult:
        """
        Modify an existing position's SL/TP.

        Args:
            ticket: Position ticket
            stop_loss: New stop loss price
            take_profit: New take profit price

        Returns:
            TradeResult
        """
        if not self.connector.is_connected():
            return TradeResult(success=False, error_message="Not connected to MT5")

        position = mt5.positions_get(ticket=ticket)
        if not position:
            return TradeResult(success=False, error_message=f"Position {ticket} not found")

        position = position[0]
        symbol_info = self.connector.get_symbol_info(position.symbol)

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": position.symbol,
            "position": ticket,
            "sl": self._normalize_price(stop_loss, symbol_info) if stop_loss else position.sl,
            "tp": self._normalize_price(take_profit, symbol_info) if take_profit else position.tp,
        }

        result = mt5.order_send(request)

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            return TradeResult(success=False, error_message="Failed to modify position")

        logger.info(f"Position {ticket} modified: SL={stop_loss}, TP={take_profit}")
        return TradeResult(success=True, ticket=ticket)

    def close_position(
        self,
        ticket: int,
        volume: Optional[float] = None,
        comment: str = "AI Trading Bot - Close"
    ) -> TradeResult:
        """
        Close a position.

        Args:
            ticket: Position ticket
            volume: Volume to close (None for full close)
            comment: Close comment

        Returns:
            TradeResult
        """
        if not self.connector.is_connected():
            return TradeResult(success=False, error_message="Not connected to MT5")

        position = mt5.positions_get(ticket=ticket)
        if not position:
            return TradeResult(success=False, error_message=f"Position {ticket} not found")

        position = position[0]
        symbol = position.symbol
        close_volume = volume if volume else position.volume

        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return TradeResult(success=False, error_message="Failed to get price")

        # Determine close order type and price
        if position.type == mt5.ORDER_TYPE_BUY:
            close_type = mt5.ORDER_TYPE_SELL
            close_price = tick.bid
        else:
            close_type = mt5.ORDER_TYPE_BUY
            close_price = tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": close_volume,
            "type": close_type,
            "position": ticket,
            "price": close_price,
            "deviation": self.deviation,
            "magic": self.magic_number,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self.fill_policy,
        }

        result = mt5.order_send(request)

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            return TradeResult(success=False, error_message="Failed to close position")

        # Calculate PnL
        if position.type == mt5.ORDER_TYPE_BUY:
            pnl = (close_price - position.price_open) * close_volume * self._get_contract_size(symbol)
        else:
            pnl = (position.price_open - close_price) * close_volume * self._get_contract_size(symbol)

        logger.info(f"Position {ticket} closed @ {close_price}, PnL: {pnl:.2f}")

        return TradeResult(
            success=True,
            ticket=result.deal,
            volume=close_volume,
            price=close_price,
            symbol=symbol,
            comment=f"PnL: {pnl:.2f}"
        )

    def close_all_positions(
        self,
        symbol: Optional[str] = None,
        magic: Optional[int] = None
    ) -> List[TradeResult]:
        """
        Close all positions.

        Args:
            symbol: Only close positions for this symbol
            magic: Only close positions with this magic number

        Returns:
            List of TradeResults
        """
        results = []
        positions = self.get_positions(symbol, magic)

        for pos in positions:
            result = self.close_position(pos['ticket'])
            results.append(result)

        return results

    def cancel_order(self, ticket: int) -> TradeResult:
        """
        Cancel a pending order.

        Args:
            ticket: Order ticket

        Returns:
            TradeResult
        """
        if not self.connector.is_connected():
            return TradeResult(success=False, error_message="Not connected to MT5")

        request = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": ticket,
        }

        result = mt5.order_send(request)

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            return TradeResult(success=False, error_message="Failed to cancel order")

        logger.info(f"Order {ticket} cancelled")
        return TradeResult(success=True, order_id=ticket)

    def get_positions(
        self,
        symbol: Optional[str] = None,
        magic: Optional[int] = None
    ) -> List[Dict]:
        """
        Get open positions.

        Args:
            symbol: Filter by symbol
            magic: Filter by magic number

        Returns:
            List of position dictionaries
        """
        if not self.connector.is_connected():
            return []

        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()

        if positions is None:
            return []

        result = []
        for pos in positions:
            if magic and pos.magic != magic:
                continue

            result.append({
                'ticket': pos.ticket,
                'symbol': pos.symbol,
                'type': 'buy' if pos.type == mt5.ORDER_TYPE_BUY else 'sell',
                'volume': pos.volume,
                'price_open': pos.price_open,
                'price_current': pos.price_current,
                'sl': pos.sl,
                'tp': pos.tp,
                'profit': pos.profit,
                'swap': pos.swap,
                'commission': getattr(pos, 'commission', 0.0),  # FIXED: Safe access with default
                'magic': pos.magic,
                'comment': pos.comment,
                'time': datetime.fromtimestamp(pos.time),
            })

        return result

    def get_pending_orders(
        self,
        symbol: Optional[str] = None
    ) -> List[Dict]:
        """
        Get pending orders.

        Args:
            symbol: Filter by symbol

        Returns:
            List of order dictionaries
        """
        if not self.connector.is_connected():
            return []

        if symbol:
            orders = mt5.orders_get(symbol=symbol)
        else:
            orders = mt5.orders_get()

        if orders is None:
            return []

        result = []
        for order in orders:
            result.append({
                'ticket': order.ticket,
                'symbol': order.symbol,
                'type': self._order_type_to_string(order.type),
                'volume': order.volume_current,
                'price_open': order.price_open,
                'sl': order.sl,
                'tp': order.tp,
                'magic': order.magic,
                'comment': order.comment,
                'time_setup': datetime.fromtimestamp(order.time_setup),
            })

        return result

    def get_trade_history(
        self,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        symbol: Optional[str] = None
    ) -> List[Dict]:
        """
        Get trade history.

        Args:
            from_date: Start date
            to_date: End date
            symbol: Filter by symbol

        Returns:
            List of deal dictionaries
        """
        if not self.connector.is_connected():
            return []

        from_date = from_date or datetime(2020, 1, 1)
        to_date = to_date or datetime.now()

        deals = mt5.history_deals_get(from_date, to_date)

        if deals is None:
            return []

        result = []
        for deal in deals:
            if symbol and deal.symbol != symbol:
                continue

            result.append({
                'ticket': deal.ticket,
                'order': deal.order,
                'symbol': deal.symbol,
                'type': self._deal_type_to_string(deal.type),
                'volume': deal.volume,
                'price': deal.price,
                'profit': deal.profit,
                'swap': deal.swap,
                'commission': getattr(deal, 'commission', 0.0),  # FIXED: Safe access with default
                'magic': deal.magic,
                'comment': deal.comment,
                'time': datetime.fromtimestamp(deal.time),
            })

        return result

    def _normalize_volume(self, volume: float, symbol_info: Dict) -> float:
        """Normalize volume to symbol's step."""
        min_vol = symbol_info['volume_min']
        max_vol = symbol_info['volume_max']
        step = symbol_info['volume_step']

        volume = max(min_vol, min(max_vol, volume))
        volume = round(volume / step) * step
        return round(volume, 2)

    def _normalize_price(self, price: float, symbol_info: Dict) -> float:
        """Normalize price to symbol's digits."""
        digits = symbol_info['digits']
        return round(price, digits)

    def _get_contract_size(self, symbol: str) -> float:
        """Get contract size for a symbol."""
        info = self.connector.get_symbol_info(symbol)
        if info:
            return info['contract_size']
        return 100000  # Default for forex

    def _order_type_to_string(self, order_type: int) -> str:
        """Convert MT5 order type to string."""
        types = {
            mt5.ORDER_TYPE_BUY: 'buy',
            mt5.ORDER_TYPE_SELL: 'sell',
            mt5.ORDER_TYPE_BUY_LIMIT: 'buy_limit',
            mt5.ORDER_TYPE_SELL_LIMIT: 'sell_limit',
            mt5.ORDER_TYPE_BUY_STOP: 'buy_stop',
            mt5.ORDER_TYPE_SELL_STOP: 'sell_stop',
        } if MT5_AVAILABLE else {}
        return types.get(order_type, 'unknown')

    def _deal_type_to_string(self, deal_type: int) -> str:
        """Convert MT5 deal type to string."""
        if not MT5_AVAILABLE:
            return 'unknown'
        types = {
            mt5.DEAL_TYPE_BUY: 'buy',
            mt5.DEAL_TYPE_SELL: 'sell',
            mt5.DEAL_TYPE_BALANCE: 'balance',
            mt5.DEAL_TYPE_CREDIT: 'credit',
            mt5.DEAL_TYPE_CHARGE: 'charge',
            mt5.DEAL_TYPE_CORRECTION: 'correction',
        }
        return types.get(deal_type, 'unknown')
