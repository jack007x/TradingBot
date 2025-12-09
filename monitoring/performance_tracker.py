"""
Performance Tracker Module
===========================
Tracks trading performance and generates reports.

Usage:
    from monitoring import PerformanceTracker
    from config import get_config

    config = get_config()
    tracker = PerformanceTracker(config)

    # Record a trade
    tracker.record_trade(trade_data)

    # Get daily summary
    summary = tracker.get_daily_summary()

    # Generate report
    report = tracker.generate_report()
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import pandas as pd
import numpy as np

from config.config_loader import Config

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """
    Tracks and analyzes trading performance.

    Features:
    - Trade logging to SQLite
    - Daily/weekly/monthly summaries
    - Equity curve tracking
    - Performance reports
    """

    def __init__(self, config: Config):
        """
        Initialize performance tracker.

        Args:
            config: Configuration object
        """
        self.config = config
        self.db_path = config.get_sqlite_path() if config.storage.use_sqlite else None

        # In-memory tracking
        self._trades: List[Dict] = []
        self._equity_history: List[Dict] = []
        self._daily_pnl: Dict[str, float] = {}

        # Initialize database
        if self.db_path:
            self._init_database()

    def _init_database(self) -> None:
        """Initialize SQLite database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket INTEGER,
                symbol TEXT,
                direction TEXT,
                lot_size REAL,
                entry_price REAL,
                exit_price REAL,
                sl REAL,
                tp REAL,
                pnl REAL,
                pnl_pips REAL,
                entry_time TEXT,
                exit_time TEXT,
                exit_reason TEXT,
                model_version TEXT,
                created_at TEXT
            )
        ''')

        # Equity history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS equity_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                balance REAL,
                equity REAL,
                open_pnl REAL,
                open_positions INTEGER
            )
        ''')

        # Daily summaries table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_summary (
                date TEXT PRIMARY KEY,
                starting_balance REAL,
                ending_balance REAL,
                pnl REAL,
                trades INTEGER,
                wins INTEGER,
                losses INTEGER,
                max_drawdown REAL
            )
        ''')

        conn.commit()
        conn.close()

        logger.info(f"Database initialized: {self.db_path}")

    def record_trade(self, trade: Dict[str, Any]) -> None:
        """
        Record a completed trade.

        Args:
            trade: Trade data dict with keys:
                - ticket, symbol, direction, lot_size
                - entry_price, exit_price, sl, tp
                - pnl, pnl_pips, entry_time, exit_time
                - exit_reason, model_version
        """
        trade['created_at'] = datetime.now().isoformat()

        # Add to memory
        self._trades.append(trade)

        # Update daily PnL
        exit_date = trade.get('exit_time', datetime.now())
        if isinstance(exit_date, str):
            exit_date = datetime.fromisoformat(exit_date.replace('Z', '+00:00'))
        date_key = exit_date.strftime('%Y-%m-%d')

        if date_key not in self._daily_pnl:
            self._daily_pnl[date_key] = 0
        self._daily_pnl[date_key] += trade.get('pnl', 0)

        # Save to database
        if self.db_path:
            self._save_trade_to_db(trade)

        logger.info(f"Trade recorded: {trade.get('direction')} {trade.get('lot_size')} @ "
                    f"{trade.get('exit_price')}, PnL: ${trade.get('pnl', 0):.2f}")

    def _save_trade_to_db(self, trade: Dict) -> None:
        """Save trade to SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO trades (
                ticket, symbol, direction, lot_size,
                entry_price, exit_price, sl, tp,
                pnl, pnl_pips, entry_time, exit_time,
                exit_reason, model_version, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade.get('ticket'),
            trade.get('symbol', 'XAUUSD'),
            trade.get('direction'),
            trade.get('lot_size'),
            trade.get('entry_price'),
            trade.get('exit_price'),
            trade.get('sl'),
            trade.get('tp'),
            trade.get('pnl'),
            trade.get('pnl_pips'),
            str(trade.get('entry_time', '')),
            str(trade.get('exit_time', '')),
            trade.get('exit_reason'),
            trade.get('model_version'),
            trade.get('created_at'),
        ))

        conn.commit()
        conn.close()

    def record_equity(
            self,
            balance: float,
            equity: float,
            open_pnl: float = 0,
            open_positions: int = 0
    ) -> None:
        """
        Record equity snapshot.

        Args:
            balance: Account balance
            equity: Account equity
            open_pnl: Unrealized PnL
            open_positions: Number of open positions
        """
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'balance': balance,
            'equity': equity,
            'open_pnl': open_pnl,
            'open_positions': open_positions,
        }

        self._equity_history.append(snapshot)

        # Save to database periodically
        if self.db_path and len(self._equity_history) % 10 == 0:
            self._save_equity_to_db(snapshot)

    def _save_equity_to_db(self, snapshot: Dict) -> None:
        """Save equity snapshot to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO equity_history (
                timestamp, balance, equity, open_pnl, open_positions
            ) VALUES (?, ?, ?, ?, ?)
        ''', (
            snapshot['timestamp'],
            snapshot['balance'],
            snapshot['equity'],
            snapshot['open_pnl'],
            snapshot['open_positions'],
        ))

        conn.commit()
        conn.close()

    def get_daily_summary(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get summary for a specific day.

        Args:
            date: Date to summarize (default: today)

        Returns:
            Dict with daily statistics
        """
        if date is None:
            date = datetime.now()

        date_key = date.strftime('%Y-%m-%d')

        # Get trades for the day
        day_trades = [
            t for t in self._trades
            if str(t.get('exit_time', ''))[:10] == date_key
        ]

        if not day_trades:
            return {
                'date': date_key,
                'trades': 0,
                'pnl': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0,
            }

        pnls = [t.get('pnl', 0) for t in day_trades]

        return {
            'date': date_key,
            'trades': len(day_trades),
            'pnl': sum(pnls),
            'wins': sum(1 for p in pnls if p > 0),
            'losses': sum(1 for p in pnls if p < 0),
            'win_rate': sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0,
            'avg_win': np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0,
            'avg_loss': np.mean([p for p in pnls if p < 0]) if any(p < 0 for p in pnls) else 0,
            'largest_win': max(pnls) if pnls else 0,
            'largest_loss': min(pnls) if pnls else 0,
        }

    def get_weekly_summary(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """Get summary for the current week."""
        if date is None:
            date = datetime.now()

        # Get start of week (Monday)
        start_of_week = date - timedelta(days=date.weekday())

        summaries = []
        for i in range(7):
            day = start_of_week + timedelta(days=i)
            if day <= date:
                summaries.append(self.get_daily_summary(day))

        total_trades = sum(s['trades'] for s in summaries)
        total_pnl = sum(s['pnl'] for s in summaries)
        total_wins = sum(s['wins'] for s in summaries)
        total_losses = sum(s['losses'] for s in summaries)

        return {
            'week_start': start_of_week.strftime('%Y-%m-%d'),
            'week_end': date.strftime('%Y-%m-%d'),
            'trades': total_trades,
            'pnl': total_pnl,
            'wins': total_wins,
            'losses': total_losses,
            'win_rate': total_wins / total_trades if total_trades > 0 else 0,
            'daily_summaries': summaries,
        }

    def get_all_time_stats(self) -> Dict[str, Any]:
        """Get all-time trading statistics."""
        if not self._trades:
            return {'trades': 0, 'pnl': 0}

        pnls = [t.get('pnl', 0) for t in self._trades]

        # Calculate equity curve and drawdown
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative + 10000)  # Assuming starting balance
        drawdown = (running_max - (cumulative + 10000)) / running_max

        return {
            'trades': len(self._trades),
            'total_pnl': sum(pnls),
            'wins': sum(1 for p in pnls if p > 0),
            'losses': sum(1 for p in pnls if p < 0),
            'win_rate': sum(1 for p in pnls if p > 0) / len(pnls),
            'avg_pnl': np.mean(pnls),
            'avg_win': np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0,
            'avg_loss': np.mean([p for p in pnls if p < 0]) if any(p < 0 for p in pnls) else 0,
            'profit_factor': abs(sum(p for p in pnls if p > 0) / sum(p for p in pnls if p < 0)) if any(
                p < 0 for p in pnls) else float('inf'),
            'max_drawdown': drawdown.max() if len(drawdown) > 0 else 0,
            'largest_win': max(pnls),
            'largest_loss': min(pnls),
            'first_trade': str(self._trades[0].get('entry_time', '')),
            'last_trade': str(self._trades[-1].get('exit_time', '')),
        }

    def generate_report(self, period: str = 'all') -> str:
        """
        Generate a text performance report.

        Args:
            period: 'daily', 'weekly', or 'all'

        Returns:
            str: Formatted report
        """
        lines = [
            "=" * 60,
            f"PERFORMANCE REPORT - {period.upper()}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            "",
        ]

        if period == 'daily':
            stats = self.get_daily_summary()
        elif period == 'weekly':
            stats = self.get_weekly_summary()
        else:
            stats = self.get_all_time_stats()

        lines.append("SUMMARY")
        lines.append("-" * 30)

        for key, value in stats.items():
            if key in ['daily_summaries', 'first_trade', 'last_trade']:
                continue

            if isinstance(value, float):
                if 'rate' in key or 'drawdown' in key:
                    lines.append(f"  {key}: {value:.2%}")
                elif 'pnl' in key.lower() or 'win' in key.lower() or 'loss' in key.lower():
                    lines.append(f"  {key}: ${value:.2f}")
                else:
                    lines.append(f"  {key}: {value:.4f}")
            else:
                lines.append(f"  {key}: {value}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def export_trades_csv(self, filepath: Optional[Path] = None) -> Path:
        """
        Export trades to CSV file.

        Returns:
            Path: Path to exported file
        """
        if filepath is None:
            filepath = self.config.get_reports_path() / f"trades_{datetime.now().strftime('%Y%m%d')}.csv"

        df = pd.DataFrame(self._trades)
        df.to_csv(filepath, index=False)

        logger.info(f"Trades exported to {filepath}")
        return filepath

    def load_trades_from_db(self, days: int = 30) -> List[Dict]:
        """Load recent trades from database."""
        if not self.db_path:
            return []

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        cursor = conn.cursor()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        cursor.execute('''
            SELECT * FROM trades
            WHERE created_at >= ?
            ORDER BY created_at DESC
        ''', (cutoff,))

        rows = cursor.fetchall()
        conn.close()

        trades = [dict(row) for row in rows]
        self._trades.extend(trades)

        return trades


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from config import get_config

    config = get_config()
    tracker = PerformanceTracker(config)

    # Record some test trades
    test_trades = [
        {'direction': 'BUY', 'lot_size': 0.1, 'entry_price': 2000, 'exit_price': 2010,
         'pnl': 100, 'pnl_pips': 1000, 'exit_time': datetime.now(), 'exit_reason': 'take_profit'},
        {'direction': 'SELL', 'lot_size': 0.1, 'entry_price': 2010, 'exit_price': 2015,
         'pnl': -50, 'pnl_pips': -500, 'exit_time': datetime.now(), 'exit_reason': 'stop_loss'},
        {'direction': 'BUY', 'lot_size': 0.2, 'entry_price': 2015, 'exit_price': 2025,
         'pnl': 200, 'pnl_pips': 1000, 'exit_time': datetime.now(), 'exit_reason': 'take_profit'},
    ]

    print("Recording test trades...")
    for trade in test_trades:
        tracker.record_trade(trade)

    # Get summaries
    print("\nDaily Summary:")
    print(json.dumps(tracker.get_daily_summary(), indent=2, default=str))

    print("\nAll-Time Stats:")
    print(json.dumps(tracker.get_all_time_stats(), indent=2, default=str))

    # Generate report
    print("\n" + tracker.generate_report('all'))
