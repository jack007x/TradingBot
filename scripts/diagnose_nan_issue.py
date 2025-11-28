"""
Find which indicators are creating NaN values
Helps optimize data pipeline and reduce data loss
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.data.data_preprocessor import DataPreprocessor
from src.mt5.mt5_data import MT5DataFetcher
from src.mt5.mt5_connector import MT5Connector
import MetaTrader5 as mt5


def diagnose_nan():
    """Diagnose NaN generation in technical indicators"""

    print("=" * 80)
    print("🔍 DIAGNOSING NaN GENERATION")
    print("=" * 80)

    # Initialize MT5
    print("\n📡 Initializing MT5...")
    if not mt5.initialize():
        print(f"❌ Failed to initialize MT5: {mt5.last_error()}")
        return

    print("✅ MT5 initialized")

    # Create connector and data fetcher
    connector = MT5Connector()
    if not connector.connect():
        print("❌ Failed to connect to MT5")
        mt5.shutdown()
        return

    data_fetcher = MT5DataFetcher(connector)
    preprocessor = DataPreprocessor()

    # Fetch data
    symbol = 'XAUUSD'
    bars = 310

    print(f"\n📥 Fetching {bars} bars of {symbol}...")

    try:
        df = data_fetcher.fetch_ohlcv(
            symbol=symbol,
            timeframe='1h',
            count=bars
        )
    except Exception as e:
        print(f"❌ Failed to fetch data: {e}")
        mt5.shutdown()
        return

    print(f"\n" + "=" * 80)
    print("STEP 1: RAW DATA ANALYSIS")
    print("=" * 80)
    print(f"Original data: {len(df)} bars, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")

    nan_before = df.isnull().sum().sum()
    print(f"NaN count BEFORE indicators: {nan_before}")

    if nan_before > 0:
        print("\nNaN in raw data:")
        for col in df.columns:
            nan_count = df[col].isnull().sum()
            if nan_count > 0:
                print(f"  {col}: {nan_count} NaN")

    # Add indicators
    print(f"\n" + "=" * 80)
    print("STEP 2: ADDING TECHNICAL INDICATORS")
    print("=" * 80)

    df_with_indicators = preprocessor.add_technical_indicators(df)

    print(f"After indicators: {len(df_with_indicators)} bars, {len(df_with_indicators.columns)} columns")

    nan_after = df_with_indicators.isnull().sum().sum()
    print(f"NaN count AFTER indicators: {nan_after}")
    print(f"NEW NaN generated: {nan_after - nan_before}")

    # Find worst offenders
    print(f"\n" + "=" * 80)
    print("STEP 3: TOP NaN OFFENDERS")
    print("=" * 80)

    nan_by_column = df_with_indicators.isnull().sum()
    nan_columns = nan_by_column[nan_by_column > 0].sort_values(ascending=False)

    if len(nan_columns) > 0:
        print(f"\nTOP 20 COLUMNS WITH MOST NaN VALUES:")
        print("-" * 80)
        print(f"{'Column':<30} {'NaN Count':>10} {'Percentage':>12}")
        print("-" * 80)

        for col, count in nan_columns.head(20).items():
            pct = (count / len(df_with_indicators)) * 100
            print(f"{col:<30} {count:>10} {pct:>11.1f}%")
    else:
        print("✅ No NaN values found!")

    # Analyze data loss
    print(f"\n" + "=" * 80)
    print("STEP 4: DATA LOSS ANALYSIS")
    print("=" * 80)

    df_clean = df_with_indicators.dropna()
    rows_lost = len(df_with_indicators) - len(df_clean)
    pct_lost = (rows_lost / len(df_with_indicators)) * 100

    print(f"Rows before dropna(): {len(df_with_indicators)}")
    print(f"Rows after dropna():  {len(df_clean)}")
    print(f"Rows lost:            {rows_lost} ({pct_lost:.1f}%)")

    if rows_lost > 0:
        print(f"\n⚠️  LOSING {pct_lost:.1f}% OF DATA!")
        if pct_lost > 50:
            print("❌ CRITICAL: Over 50% data loss!")

    # Check specific problematic indicators
    print(f"\n" + "=" * 80)
    print("STEP 5: CHECKING COMMON CULPRITS")
    print("=" * 80)

    long_period_indicators = {
        'sma_200': 200,
        'ema_200': 200,
        'sma_100': 100,
        'ema_100': 100,
        'sma_50': 50,
        'ema_50': 50
    }

    for indicator, period in long_period_indicators.items():
        if indicator in df_with_indicators.columns:
            nan_count = df_with_indicators[indicator].isnull().sum()
            if nan_count > 0:
                print(f"  {indicator:15} (period {period:3}): {nan_count:4} NaN")

    # Recommendations
    print(f"\n" + "=" * 80)
    print("STEP 6: RECOMMENDATIONS")
    print("=" * 80)

    if pct_lost > 50:
        print("\n🔧 RECOMMENDED FIXES:")
        print("1. Use forward-fill (ffill) and backward-fill (bfill) instead of dropna()")
        print("2. Reduce long-period indicators (200 → 100 or 50)")
        print("3. Fetch more initial bars to compensate for NaN loss")
        print(f"   Current: {bars} bars")
        print(f"   Recommended: {bars + rows_lost + 50} bars")
    elif pct_lost > 30:
        print("\n⚠️  MODERATE DATA LOSS:")
        print("1. Consider forward-fill for some columns")
        print("2. Review if all long-period indicators are necessary")
    else:
        print("\n✅ Data loss is acceptable (< 30%)")

    # Summary
    print(f"\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total NaN values: {nan_after}")
    print(f"Rows with NaN: {rows_lost} ({pct_lost:.1f}%)")
    print(f"Clean rows: {len(df_clean)}")
    print(f"Required for 60-step sequence: 60")
    print(f"Available after cleaning: {len(df_clean)}")

    if len(df_clean) >= 60:
        print(f"✅ Sufficient data for predictions ({len(df_clean)} >= 60)")
    else:
        print(f"❌ INSUFFICIENT data for predictions ({len(df_clean)} < 60)")

    print("=" * 80)

    # Cleanup
    connector.disconnect()
    mt5.shutdown()


if __name__ == "__main__":
    diagnose_nan()
