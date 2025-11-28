#!/usr/bin/env python3
"""
Quick Fix Script - Automated Diagnostics and Fixes
Quickly diagnose and fix common trading bot issues
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import MetaTrader5 as mt5
from src.mt5.mt5_connector import MT5Connector
from src.mt5.mt5_data import MT5DataFetcher
from src.data.data_preprocessor import DataPreprocessor
from src.utils.market_hours import MarketHoursChecker
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_mt5_connection():
    """Check MT5 connection status"""
    print("\n" + "=" * 80)
    print("✅ TEST 1: MT5 CONNECTION")
    print("=" * 80)

    if not mt5.initialize():
        print(f"❌ FAILED: Cannot initialize MT5")
        print(f"   Error: {mt5.last_error()}")
        return False

    connector = MT5Connector()
    if not connector.connect():
        print("❌ FAILED: Cannot connect to MT5")
        mt5.shutdown()
        return False

    account = connector.get_account_info()
    if account:
        print(f"✅ CONNECTED")
        print(f"   Account: {account.get('login', 'N/A')}")
        print(f"   Balance: ${account.get('balance', 0):.2f}")
        print(f"   Leverage: 1:{account.get('leverage', 0)}")
    else:
        print("⚠️  Connected but cannot get account info")

    return True


def check_market_hours():
    """Check if market is open"""
    print("\n" + "=" * 80)
    print("✅ TEST 2: MARKET HOURS")
    print("=" * 80)

    status = MarketHoursChecker.get_market_status()

    print(f"Current Time (UTC): {status['current_time_utc']}")
    print(f"Weekday: {status['weekday']}")
    print(f"Hour (UTC): {status['hour_utc']}")

    if status['is_open']:
        print(f"✅ MARKET OPEN: {status['reason']}")
        return True
    else:
        print(f"❌ MARKET CLOSED: {status['reason']}")
        print(f"   Next open: {status.get('next_open', 'N/A')}")

        wait_seconds = MarketHoursChecker.wait_time_until_open()
        wait_hours = wait_seconds / 3600
        print(f"   Wait time: {wait_hours:.1f} hours")
        return False


def check_data_pipeline(symbol='XAUUSD', bars=310):
    """Check data fetching and processing pipeline"""
    print("\n" + "=" * 80)
    print(f"✅ TEST 3: DATA PIPELINE ({symbol})")
    print("=" * 80)

    connector = MT5Connector()
    if not connector.connect():
        print("❌ FAILED: Cannot connect to MT5")
        return False

    data_fetcher = MT5DataFetcher(connector)
    preprocessor = DataPreprocessor()

    # Fetch data
    print(f"\n📥 Fetching {bars} bars of {symbol}...")
    try:
        df = data_fetcher.fetch_historical_data(
            symbol=symbol,
            timeframe=mt5.TIMEFRAME_H1,
            bars=bars
        )
        print(f"✅ Fetched: {len(df)} bars")
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

    # Add indicators
    print(f"\n📊 Adding technical indicators...")
    try:
        df = preprocessor.add_technical_indicators(df)
        print(f"✅ After indicators: {len(df)} bars")
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

    # Check NaN
    nan_count = df.isnull().sum().sum()
    print(f"\n🔍 NaN check: {nan_count} values")

    # Drop NaN
    df_clean = df.dropna()
    rows_lost = len(df) - len(df_clean)
    pct_lost = (rows_lost / len(df)) * 100 if len(df) > 0 else 0

    print(f"\n✅ After NaN removal: {len(df_clean)} bars")
    print(f"   Lost: {rows_lost} bars ({pct_lost:.1f}%)")

    # Check sufficiency
    sequence_length = 60
    if len(df_clean) >= sequence_length:
        print(f"✅ SUFFICIENT DATA: {len(df_clean)} >= {sequence_length}")
        return True
    else:
        print(f"❌ INSUFFICIENT DATA: {len(df_clean)} < {sequence_length}")
        print(f"   Increase bars from {bars} to {bars + 100}")
        return False


def check_positions():
    """Check open positions"""
    print("\n" + "=" * 80)
    print("✅ TEST 4: OPEN POSITIONS")
    print("=" * 80)

    connector = MT5Connector()
    if not connector.connect():
        print("❌ FAILED: Cannot connect to MT5")
        return False

    from src.mt5.mt5_trader import MT5Trader
    trader = MT5Trader(connector)

    positions = trader.get_positions()

    if not positions or len(positions) == 0:
        print("✅ No open positions")
        return True

    print(f"📍 {len(positions)} open position(s):")
    for pos in positions:
        ticket = pos.get('ticket', 'N/A')
        symbol = pos.get('symbol', 'N/A')
        pos_type = pos.get('type', 'N/A')
        volume = pos.get('volume', 0.0)
        price_open = pos.get('price_open', 0.0)
        profit = pos.get('profit', 0.0)

        print(f"\n   Position #{ticket}")
        print(f"   Symbol: {symbol} | Type: {pos_type}")
        print(f"   Volume: {volume} | Entry: {price_open:.2f}")
        print(f"   P&L: ${profit:.2f}")

    return True


def check_models():
    """Check if models are available"""
    print("\n" + "=" * 80)
    print("✅ TEST 5: MODEL FILES")
    print("=" * 80)

    models_dir = Path('saved_models')

    if not models_dir.exists():
        print("❌ FAILED: saved_models/ directory not found")
        print("   Run training first!")
        return False

    model_files = {
        'AttentionLSTM': models_dir / 'lstm_model.pt',
        'GRU': models_dir / 'gru_model.pt',
        'Metrics': models_dir / 'model_metrics.json'
    }

    all_found = True
    for name, path in model_files.items():
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"✅ {name}: {path} ({size_kb:.1f} KB)")
        else:
            print(f"❌ {name}: NOT FOUND")
            all_found = False

    if not all_found:
        print("\n⚠️  Some models missing - run training first!")
        return False

    print("\n✅ All model files found")
    return True


def main():
    """Run all diagnostic checks"""
    print("=" * 80)
    print("🔧 QUICK FIX - AUTOMATED DIAGNOSTICS")
    print("=" * 80)
    print("Running comprehensive diagnostic checks...")

    results = {
        'MT5 Connection': False,
        'Market Hours': False,
        'Data Pipeline': False,
        'Open Positions': False,
        'Model Files': False
    }

    # Run checks
    try:
        results['MT5 Connection'] = check_mt5_connection()
    except Exception as e:
        print(f"❌ MT5 Connection check failed: {e}")

    try:
        results['Market Hours'] = check_market_hours()
    except Exception as e:
        print(f"❌ Market Hours check failed: {e}")

    try:
        results['Data Pipeline'] = check_data_pipeline()
    except Exception as e:
        print(f"❌ Data Pipeline check failed: {e}")

    try:
        results['Open Positions'] = check_positions()
    except Exception as e:
        print(f"❌ Position check failed: {e}")

    try:
        results['Model Files'] = check_models()
    except Exception as e:
        print(f"❌ Model Files check failed: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 80)

    all_passed = True
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check_name:20s}: {status}")
        if not passed:
            all_passed = False

    print("=" * 80)

    if all_passed:
        print("\n🎉 ALL CHECKS PASSED - Bot should work!")
        print("\nRecommended next steps:")
        print("1. Start the bot: python run_mt5_bot.py")
        print("2. Monitor logs for trading activity")
        print("3. Check positions with /positions command")
    else:
        print("\n⚠️  ISSUES DETECTED - Fix problems above before trading!")
        print("\nCommon fixes:")
        print("1. MT5 not running? Start MetaTrader 5")
        print("2. Models missing? Run: python scripts/train_models.py")
        print("3. Market closed? Wait until Sunday 22:00 GMT")
        print("4. Data insufficient? Already fixed in latest code")

    # Cleanup
    mt5.shutdown()

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
