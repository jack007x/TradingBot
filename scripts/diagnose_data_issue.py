"""
Diagnose why bot says "Insufficient data"
Run this to see exactly what's happening
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import MetaTrader5 as mt5
from src.mt5.mt5_data import MT5DataFetcher
from src.data.data_preprocessor import DataPreprocessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def diagnose():
    """Run diagnostic checks"""

    print("=" * 80)
    print("🔍 DIAGNOSING DATA ISSUE")
    print("=" * 80)

    # Initialize MT5
    print("\n📡 Initializing MT5...")
    if not mt5.initialize():
        print("❌ Failed to initialize MT5")
        print(f"   Error: {mt5.last_error()}")
        return

    print("✅ MT5 initialized")

    # Create MT5 connector and data fetcher
    from src.mt5.mt5_connector import MT5Connector

    connector = MT5Connector()
    if not connector.connect():
        print("❌ Failed to connect to MT5")
        mt5.shutdown()
        return

    print("✅ MT5 connected")

    data_fetcher = MT5DataFetcher(connector)
    preprocessor = DataPreprocessor()

    # Test data fetching
    symbol = 'XAUUSD'
    bars_to_fetch = 100

    print(f"\n📥 TEST 1: Fetching {bars_to_fetch} bars of {symbol}...")

    try:
        df = data_fetcher.fetch_historical_data(
            symbol=symbol,
            timeframe=mt5.TIMEFRAME_H1,
            bars=bars_to_fetch
        )

        print(f"✅ Fetched: {len(df)} bars")
        print(f"   Columns: {len(df.columns)}")
        if len(df) > 0:
            print(f"   Date range: {df.index[0]} to {df.index[-1]}")
        else:
            print("❌ NO DATA FETCHED!")
            mt5.shutdown()
            return

    except Exception as e:
        print(f"❌ Data fetch failed: {e}")
        import traceback
        traceback.print_exc()
        mt5.shutdown()
        return

    # Add indicators
    print(f"\n📊 TEST 2: Adding technical indicators...")
    try:
        df_with_indicators = preprocessor.add_technical_indicators(df)

        print(f"✅ After indicators: {len(df_with_indicators)} bars, {len(df_with_indicators.columns)} features")
        print(f"   Lost: {len(df) - len(df_with_indicators)} bars")

    except Exception as e:
        print(f"❌ Indicator calculation failed: {e}")
        import traceback
        traceback.print_exc()
        mt5.shutdown()
        return

    # Check NaN
    print(f"\n🔍 TEST 3: Checking for NaN values...")
    nan_count = df_with_indicators.isnull().sum().sum()
    print(f"   Total NaN values: {nan_count}")

    if nan_count > 0:
        print("\n   NaN by column (top 10):")
        nan_cols = df_with_indicators.isnull().sum()
        nan_cols = nan_cols[nan_cols > 0].sort_values(ascending=False)
        for col, count in nan_cols.head(10).items():
            print(f"     {col}: {count}")

    # Drop NaN
    df_clean = df_with_indicators.dropna()
    print(f"\n✅ After NaN removal: {len(df_clean)} bars")
    print(f"   Lost: {len(df_with_indicators) - len(df_clean)} bars to NaN")

    # Calculate how many bars we need
    sequence_length = 60
    prediction_horizon = 4
    min_required = sequence_length + prediction_horizon

    print(f"\n📏 TEST 4: Sequence requirements...")
    print(f"   Sequence length: {sequence_length}")
    print(f"   Prediction horizon: {prediction_horizon}")
    print(f"   Minimum required: {min_required} bars")
    print(f"   Available: {len(df_clean)} bars")

    if len(df_clean) < min_required:
        print(f"\n❌ INSUFFICIENT DATA!")
        print(f"   Have: {len(df_clean)} bars")
        print(f"   Need: {min_required} bars")
        print(f"   Shortfall: {min_required - len(df_clean)} bars")
        print(f"\n💡 SOLUTION: Fetch at least {bars_to_fetch + (min_required - len(df_clean)) + 50} bars initially")
    else:
        print(f"✅ Sufficient data: {len(df_clean)} >= {min_required}")

    # Try to create sequences
    print(f"\n🔄 TEST 5: Creating sequences...")

    try:
        # Check if prepare_regression_sequences exists
        if not hasattr(preprocessor, 'prepare_regression_sequences'):
            print("❌ ERROR: prepare_regression_sequences method not found!")
            print(f"   Available methods: {[m for m in dir(preprocessor) if not m.startswith('_')]}")
            mt5.shutdown()
            return

        # Get feature columns (exclude non-numeric)
        feature_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns.tolist()
        print(f"   Using {len(feature_cols)} numeric features")

        sequences, targets = preprocessor.prepare_regression_sequences(
            df_clean,
            sequence_length=sequence_length,
            target_col='close',  # Use close as target
            feature_cols=feature_cols
        )

        print(f"✅ Created {len(sequences)} sequences")
        if len(sequences) > 0:
            print(f"   Sequence shape: {sequences.shape}")
            print(f"   Target shape: {targets.shape}")
            print(f"   Latest sequence available: YES")
            print("\n✅ SUCCESS! Data pipeline works!")
        else:
            print("\n❌ FAILED! No sequences created!")
            print(f"   Input: {len(df_clean)} bars")
            print(f"   Required: {min_required} bars minimum")

    except Exception as e:
        print(f"\n❌ ERROR creating sequences: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 80)
    print(f"1. Data Fetch: {'✅ OK' if len(df) >= bars_to_fetch * 0.9 else '❌ FAILED'}")
    print(f"2. Indicators: {'✅ OK' if len(df_with_indicators) > 0 else '❌ FAILED'}")
    print(f"3. NaN Handling: {'✅ OK' if len(df_clean) > 0 else '❌ FAILED'}")
    print(f"4. Data Sufficiency: {'✅ OK' if len(df_clean) >= min_required else '❌ INSUFFICIENT'}")

    try:
        print(f"5. Sequence Creation: {'✅ OK' if len(sequences) > 0 else '❌ FAILED'}")
    except:
        print(f"5. Sequence Creation: ❌ FAILED")

    print("\n💡 RECOMMENDATIONS:")
    if len(df_clean) < min_required:
        recommended_bars = 300  # Safe amount
        print(f"   - Increase bars fetched from {bars_to_fetch} to {recommended_bars}")
        print(f"   - This accounts for indicator calculation and NaN removal")
    else:
        print(f"   - Data pipeline is working correctly!")
        print(f"   - Problem may be in get_trading_signal() logic")

    print("=" * 80)

    # Cleanup
    mt5.shutdown()


if __name__ == "__main__":
    diagnose()
