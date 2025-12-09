"""
Data Validator Module
======================
Validates OHLCV data quality and detects issues.

Usage:
    from data import DataValidator
    from config import get_config

    config = get_config()
    validator = DataValidator(config)

    # Validate data
    is_valid, issues = validator.validate(df)
    if not is_valid:
        print("Data issues found:", issues)

    # Auto-fix minor issues
    df_clean = validator.clean(df)
"""

import logging
from datetime import timedelta
from typing import Tuple, List, Dict, Any
import pandas as pd
import numpy as np

from config.config_loader import Config, get_timeframe_minutes

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validates and cleans OHLCV data.

    Checks for:
    - Missing required columns
    - Invalid OHLC relationships (high < low, etc.)
    - Missing timestamps (gaps)
    - Duplicate timestamps
    - Extreme outliers
    - Zero/negative prices
    """

    def __init__(self, config: Config):
        """
        Initialize validator.

        Args:
            config: Configuration object
        """
        self.config = config
        self.timeframe = config.trading.timeframe
        self.tf_minutes = get_timeframe_minutes(self.timeframe)

        # Validation thresholds
        self.max_price_change_pct = 5.0  # Max 5% change per bar (extreme for gold)
        self.max_spread_pips = 100  # Max spread in pips
        self.min_price = 100.0  # Min realistic gold price
        self.max_price = 10000.0  # Max realistic gold price

    def validate(
            self,
            df: pd.DataFrame,
            strict: bool = False
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Validate OHLCV data.

        Args:
            df: OHLCV DataFrame
            strict: If True, treat warnings as errors

        Returns:
            Tuple[bool, List[Dict]]: (is_valid, list of issues)
        """
        issues = []

        if df.empty:
            issues.append({
                'type': 'error',
                'message': 'DataFrame is empty',
                'count': 0
            })
            return False, issues

        # Check required columns
        required_cols = ['time', 'open', 'high', 'low', 'close']
        missing = set(required_cols) - set(df.columns)
        if missing:
            issues.append({
                'type': 'error',
                'message': f'Missing required columns: {missing}',
                'columns': list(missing)
            })
            return False, issues

        # Check for duplicates
        duplicates = df['time'].duplicated().sum()
        if duplicates > 0:
            issues.append({
                'type': 'warning',
                'message': f'Found {duplicates} duplicate timestamps',
                'count': duplicates
            })

        # Check OHLC relationships
        invalid_ohlc = self._check_ohlc_validity(df)
        if invalid_ohlc > 0:
            issues.append({
                'type': 'error',
                'message': f'Found {invalid_ohlc} bars with invalid OHLC relationships',
                'count': invalid_ohlc
            })

        # Check for gaps
        gaps = self._find_gaps(df)
        if gaps:
            # Filter out expected gaps (weekends, etc.)
            unexpected_gaps = [g for g in gaps if not g['expected']]
            if unexpected_gaps:
                issues.append({
                    'type': 'warning',
                    'message': f'Found {len(unexpected_gaps)} unexpected gaps in data',
                    'gaps': unexpected_gaps[:10]  # First 10 gaps
                })

        # Check for outliers
        outliers = self._detect_outliers(df)
        if outliers > 0:
            issues.append({
                'type': 'warning',
                'message': f'Found {outliers} potential outlier bars',
                'count': outliers
            })

        # Check for zero/negative prices
        invalid_prices = self._check_price_validity(df)
        if invalid_prices > 0:
            issues.append({
                'type': 'error',
                'message': f'Found {invalid_prices} bars with invalid prices',
                'count': invalid_prices
            })

        # Check for NaN values
        nan_count = df[['open', 'high', 'low', 'close']].isna().sum().sum()
        if nan_count > 0:
            issues.append({
                'type': 'error',
                'message': f'Found {nan_count} NaN values in price data',
                'count': nan_count
            })

        # Determine if valid
        has_errors = any(i['type'] == 'error' for i in issues)
        has_warnings = any(i['type'] == 'warning' for i in issues)

        if has_errors:
            is_valid = False
        elif has_warnings and strict:
            is_valid = False
        else:
            is_valid = True

        return is_valid, issues

    def clean(
            self,
            df: pd.DataFrame,
            fix_ohlc: bool = True,
            remove_duplicates: bool = True,
            remove_outliers: bool = False,
            fill_gaps: bool = False
    ) -> pd.DataFrame:
        """
        Clean OHLCV data by fixing common issues.

        Args:
            df: OHLCV DataFrame
            fix_ohlc: Fix invalid OHLC relationships
            remove_duplicates: Remove duplicate timestamps
            remove_outliers: Remove extreme outlier bars
            fill_gaps: Forward-fill gaps (use with caution)

        Returns:
            pd.DataFrame: Cleaned data
        """
        if df.empty:
            return df

        df = df.copy()
        initial_len = len(df)

        # Remove duplicates (keep last)
        if remove_duplicates:
            before = len(df)
            df = df.drop_duplicates(subset=['time'], keep='last')
            removed = before - len(df)
            if removed > 0:
                logger.info(f"Removed {removed} duplicate rows")

        # Sort by time
        df = df.sort_values('time').reset_index(drop=True)

        # Fix OHLC relationships
        if fix_ohlc:
            df = self._fix_ohlc(df)

        # Remove outliers
        if remove_outliers:
            before = len(df)
            df = self._remove_outliers(df)
            removed = before - len(df)
            if removed > 0:
                logger.info(f"Removed {removed} outlier rows")

        # Fill gaps (forward fill)
        if fill_gaps:
            df = self._fill_gaps(df)

        # Remove rows with NaN in essential columns
        essential_cols = ['time', 'open', 'high', 'low', 'close']
        before = len(df)
        df = df.dropna(subset=essential_cols)
        removed = before - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} rows with NaN values")

        logger.info(f"Cleaning complete: {initial_len} -> {len(df)} rows")
        return df.reset_index(drop=True)

    def _check_ohlc_validity(self, df: pd.DataFrame) -> int:
        """
        Check for invalid OHLC relationships.

        Valid: Low <= Open, Close <= High
               Low <= min(Open, Close)
               High >= max(Open, Close)
        """
        invalid = (
                (df['high'] < df['low']) |
                (df['high'] < df['open']) |
                (df['high'] < df['close']) |
                (df['low'] > df['open']) |
                (df['low'] > df['close'])
        )
        return invalid.sum()

    def _fix_ohlc(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix invalid OHLC relationships."""
        df = df.copy()

        # Ensure high is highest
        df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)

        # Ensure low is lowest
        df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)

        return df

    def _find_gaps(self, df: pd.DataFrame) -> List[Dict]:
        """Find gaps in the time series."""
        if len(df) < 2:
            return []

        expected_delta = timedelta(minutes=self.tf_minutes)
        time_diffs = df['time'].diff()

        gaps = []
        for idx in range(1, len(df)):
            actual_delta = time_diffs.iloc[idx]

            if actual_delta > expected_delta:
                gap_start = df['time'].iloc[idx - 1]
                gap_end = df['time'].iloc[idx]

                # Check if gap is expected (weekend)
                is_weekend = (gap_start.weekday() == 4 and gap_end.weekday() == 6) or \
                             (gap_start.weekday() == 4 and gap_end.weekday() == 0)

                gaps.append({
                    'start': gap_start.isoformat(),
                    'end': gap_end.isoformat(),
                    'missing_bars': int(actual_delta.total_seconds() / 60 / self.tf_minutes) - 1,
                    'expected': is_weekend
                })

        return gaps

    def _fill_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill small gaps (use with caution)."""
        if len(df) < 2:
            return df

        # Create complete time index
        full_range = pd.date_range(
            start=df['time'].min(),
            end=df['time'].max(),
            freq=f'{self.tf_minutes}min',
            tz=df['time'].dt.tz
        )

        # Reindex and forward fill
        df_reindexed = df.set_index('time').reindex(full_range)
        df_reindexed = df_reindexed.ffill(limit=3)  # Only fill up to 3 bars

        df_filled = df_reindexed.reset_index().rename(columns={'index': 'time'})

        return df_filled.dropna()

    def _detect_outliers(self, df: pd.DataFrame) -> int:
        """Detect extreme price movements that might be outliers."""
        if len(df) < 2:
            return 0

        # Calculate returns
        returns = df['close'].pct_change().abs()

        # Count extreme movements
        outliers = (returns > self.max_price_change_pct / 100).sum()

        return outliers

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with extreme price movements."""
        if len(df) < 2:
            return df

        returns = df['close'].pct_change().abs()
        mask = returns <= self.max_price_change_pct / 100
        mask.iloc[0] = True  # Keep first row

        return df[mask]

    def _check_price_validity(self, df: pd.DataFrame) -> int:
        """Check for unrealistic prices."""
        price_cols = ['open', 'high', 'low', 'close']

        invalid = pd.Series(False, index=df.index)
        for col in price_cols:
            invalid |= (df[col] <= 0) | (df[col] < self.min_price) | (df[col] > self.max_price)

        return invalid.sum()

    def get_quality_score(self, df: pd.DataFrame) -> float:
        """
        Calculate a data quality score (0-100).

        Higher is better.
        """
        if df.empty:
            return 0.0

        _, issues = self.validate(df)

        # Start with perfect score
        score = 100.0

        for issue in issues:
            if issue['type'] == 'error':
                score -= 30  # Errors are severe
            elif issue['type'] == 'warning':
                score -= 10  # Warnings are minor

        return max(0.0, score)

    def generate_report(self, df: pd.DataFrame) -> str:
        """Generate a text report of data quality."""
        is_valid, issues = self.validate(df)

        lines = [
            "=" * 60,
            "DATA QUALITY REPORT",
            "=" * 60,
            "",
            f"Total Rows: {len(df)}",
        ]

        if not df.empty:
            lines.extend([
                f"Date Range: {df['time'].min()} to {df['time'].max()}",
                f"Price Range: {df['low'].min():.2f} - {df['high'].max():.2f}",
                f"Timeframe: {self.timeframe}",
            ])

        lines.append("")
        lines.append(f"Quality Score: {self.get_quality_score(df):.1f}/100")
        lines.append(f"Valid: {'Yes' if is_valid else 'No'}")
        lines.append("")

        if issues:
            lines.append("Issues Found:")
            for issue in issues:
                icon = "❌" if issue['type'] == 'error' else "⚠️"
                lines.append(f"  {icon} [{issue['type'].upper()}] {issue['message']}")
        else:
            lines.append("No issues found. Data quality is good.")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from config import get_config
    from data.data_fetcher import DataFetcher

    config = get_config()
    validator = DataValidator(config)
    fetcher = DataFetcher(config)

    # Fetch demo data
    from datetime import datetime
    start = datetime(2024, 1, 1)
    end = datetime(2024, 3, 1)
    df = fetcher.fetch_historical(start, end)

    # Validate
    print("Validating data...")
    is_valid, issues = validator.validate(df)
    print(f"Valid: {is_valid}")
    print(f"Issues: {len(issues)}")

    # Generate report
    print("\n" + validator.generate_report(df))

    # Test with deliberately bad data
    print("\nTesting with bad data...")
    df_bad = df.copy()
    df_bad.loc[5, 'high'] = df_bad.loc[5, 'low'] - 10  # Invalid OHLC
    df_bad = pd.concat([df_bad, df_bad.iloc[[10]]])  # Duplicate

    is_valid, issues = validator.validate(df_bad)
    print(f"Valid: {is_valid}")
    for issue in issues:
        print(f"  - {issue['type']}: {issue['message']}")

    # Clean bad data
    print("\nCleaning data...")
    df_clean = validator.clean(df_bad)
    print(f"Rows before: {len(df_bad)}, after: {len(df_clean)}")
