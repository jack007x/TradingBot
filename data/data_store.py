"""
Data Store Module
==================
Handles saving and loading OHLCV data to/from disk.
Supports Parquet (recommended) and CSV formats.

Usage:
    from data import DataStore
    from config import get_config

    config = get_config()
    store = DataStore(config)

    # Save data
    store.save_historical(df)

    # Load data
    df = store.load_historical()

    # Append new data
    store.append_data(df_new)
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import pandas as pd

from config.config_loader import Config

logger = logging.getLogger(__name__)


class DataStore:
    """
    Handles persistence of OHLCV data.

    Features:
    - Save/load in Parquet or CSV format
    - Append new data without duplicates
    - Automatic backup before overwrite
    - Data integrity checks
    """

    def __init__(self, config: Config):
        """
        Initialize data store.

        Args:
            config: Configuration object
        """
        self.config = config
        self.data_dir = config.get_data_cache_path()
        self.format = config.storage.format.lower()

        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # File path
        base_name = config.storage.historical_data_file
        if self.format == 'parquet' and not base_name.endswith('.parquet'):
            base_name = base_name.rsplit('.', 1)[0] + '.parquet'
        elif self.format == 'csv' and not base_name.endswith('.csv'):
            base_name = base_name.rsplit('.', 1)[0] + '.csv'

        self.data_path = self.data_dir / base_name

    def save_historical(
            self,
            df: pd.DataFrame,
            backup: bool = True
    ) -> Path:
        """
        Save historical data to disk.

        Args:
            df: OHLCV DataFrame
            backup: Create backup of existing file

        Returns:
            Path: Path to saved file
        """
        if df.empty:
            logger.warning("Empty DataFrame, nothing to save")
            return self.data_path

        # Validate columns
        required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Create backup if file exists
        if backup and self.data_path.exists():
            backup_path = self._create_backup()
            logger.info(f"Created backup: {backup_path}")

        # Ensure time is datetime with timezone
        if not pd.api.types.is_datetime64_any_dtype(df['time']):
            df = df.copy()
            df['time'] = pd.to_datetime(df['time'], utc=True)

        # Sort by time
        df = df.sort_values('time').reset_index(drop=True)

        # Save based on format
        if self.format == 'parquet':
            df.to_parquet(self.data_path, index=False, engine='pyarrow')
        else:
            df.to_csv(self.data_path, index=False)

        logger.info(f"Saved {len(df)} bars to {self.data_path}")
        return self.data_path

    def load_historical(
            self,
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Load historical data from disk.

        Args:
            start_date: Filter data from this date
            end_date: Filter data until this date

        Returns:
            pd.DataFrame: OHLCV data
        """
        if not self.data_path.exists():
            logger.warning(f"Data file not found: {self.data_path}")
            return pd.DataFrame()

        # Load based on format
        if self.format == 'parquet':
            df = pd.read_parquet(self.data_path)
        else:
            df = pd.read_csv(self.data_path, parse_dates=['time'])

        # Ensure time is datetime with UTC
        if 'time' in df.columns:
            if df['time'].dt.tz is None:
                df['time'] = df['time'].dt.tz_localize('UTC')
            else:
                df['time'] = df['time'].dt.tz_convert('UTC')

        # Filter by date range
        if start_date is not None:
            if start_date.tzinfo is None:
                start_date = start_date.replace(tzinfo=df['time'].dt.tz)
            df = df[df['time'] >= start_date]

        if end_date is not None:
            if end_date.tzinfo is None:
                end_date = end_date.replace(tzinfo=df['time'].dt.tz)
            df = df[df['time'] <= end_date]

        logger.info(f"Loaded {len(df)} bars from {self.data_path}")
        return df.reset_index(drop=True)

    def append_data(
            self,
            df_new: pd.DataFrame,
            deduplicate: bool = True
    ) -> pd.DataFrame:
        """
        Append new data to existing historical data.

        Args:
            df_new: New OHLCV data to append
            deduplicate: Remove duplicate timestamps

        Returns:
            pd.DataFrame: Combined data
        """
        if df_new.empty:
            logger.warning("Empty DataFrame, nothing to append")
            return self.load_historical()

        # Load existing data
        df_existing = self.load_historical()

        if df_existing.empty:
            # No existing data, just save new
            self.save_historical(df_new)
            return df_new

        # Ensure time columns are compatible
        if df_new['time'].dt.tz is None:
            df_new = df_new.copy()
            df_new['time'] = df_new['time'].dt.tz_localize('UTC')

        # Combine
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)

        # Remove duplicates based on time
        if deduplicate:
            initial_len = len(df_combined)
            df_combined = df_combined.drop_duplicates(subset=['time'], keep='last')
            duplicates_removed = initial_len - len(df_combined)
            if duplicates_removed > 0:
                logger.info(f"Removed {duplicates_removed} duplicate bars")

        # Sort by time
        df_combined = df_combined.sort_values('time').reset_index(drop=True)

        # Save
        self.save_historical(df_combined, backup=True)

        logger.info(f"Appended {len(df_new)} bars, total: {len(df_combined)}")
        return df_combined

    def get_last_timestamp(self) -> Optional[datetime]:
        """
        Get the timestamp of the last bar in storage.

        Returns:
            datetime: Last timestamp or None if no data
        """
        df = self.load_historical()
        if df.empty:
            return None
        return df['time'].max()

    def get_first_timestamp(self) -> Optional[datetime]:
        """
        Get the timestamp of the first bar in storage.

        Returns:
            datetime: First timestamp or None if no data
        """
        df = self.load_historical()
        if df.empty:
            return None
        return df['time'].min()

    def get_data_info(self) -> dict:
        """
        Get information about stored data.

        Returns:
            dict: Data statistics
        """
        df = self.load_historical()

        if df.empty:
            return {
                'exists': False,
                'rows': 0,
                'file_path': str(self.data_path),
            }

        return {
            'exists': True,
            'rows': len(df),
            'file_path': str(self.data_path),
            'file_size_mb': self.data_path.stat().st_size / (1024 * 1024),
            'first_date': df['time'].min().isoformat(),
            'last_date': df['time'].max().isoformat(),
            'columns': list(df.columns),
        }

    def _create_backup(self) -> Path:
        """
        Create a backup of the current data file.

        Returns:
            Path: Path to backup file
        """
        if not self.data_path.exists():
            return None

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"{self.data_path.stem}_backup_{timestamp}{self.data_path.suffix}"
        backup_path = self.data_dir / 'backups' / backup_name

        # Create backup directory
        backup_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy file
        import shutil
        shutil.copy2(self.data_path, backup_path)

        return backup_path

    def list_backups(self) -> List[Path]:
        """
        List all backup files.

        Returns:
            List[Path]: List of backup file paths
        """
        backup_dir = self.data_dir / 'backups'
        if not backup_dir.exists():
            return []

        pattern = f"*_backup_*{self.data_path.suffix}"
        return sorted(backup_dir.glob(pattern), reverse=True)

    def delete_old_backups(self, keep: int = 5) -> int:
        """
        Delete old backup files, keeping the most recent ones.

        Args:
            keep: Number of backups to keep

        Returns:
            int: Number of backups deleted
        """
        backups = self.list_backups()
        to_delete = backups[keep:]

        for path in to_delete:
            path.unlink()
            logger.info(f"Deleted old backup: {path}")

        return len(to_delete)


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from config import get_config
    from data.data_fetcher import DataFetcher

    config = get_config()
    store = DataStore(config)
    fetcher = DataFetcher(config)

    # Fetch some demo data
    start = datetime(2024, 1, 1)
    end = datetime(2024, 6, 1)
    df = fetcher.fetch_historical(start, end)

    # Save data
    print("Saving data...")
    store.save_historical(df)

    # Get info
    print("\nData info:")
    info = store.get_data_info()
    for k, v in info.items():
        print(f"  {k}: {v}")

    # Load data
    print("\nLoading data...")
    df_loaded = store.load_historical()
    print(f"Loaded {len(df_loaded)} rows")

    # Append more data
    print("\nAppending data...")
    end2 = datetime(2024, 8, 1)
    df_new = fetcher.fetch_historical(end, end2)
    df_combined = store.append_data(df_new)
    print(f"Total rows after append: {len(df_combined)}")
