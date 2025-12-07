#!/usr/bin/env python3
"""
AI Trading Bot XAUUSD - Main Entry Point
==========================================

Usage:
    python main.py --mode <mode> [options]

Modes:
    fetch_data  - Download historical data from MT5
    train       - Train a new model
    backtest    - Run backtest simulation
    paper_trade - Run paper trading (demo account)
    live_trade  - Run live trading (real account)
    retrain     - Run retraining cycle
    status      - Show system status

Examples:
    python main.py --mode fetch_data --days 365
    python main.py --mode train
    python main.py --mode backtest --start 2023-01-01 --end 2024-01-01
    python main.py --mode paper_trade
    python main.py --mode retrain
"""

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.config_loader import get_config, get_timeframe_minutes
from monitoring.logger_setup import setup_logging, get_logger, log_trade, log_decision


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='AI Trading Bot for XAUUSD',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--mode', '-m',
        required=True,
        choices=['fetch_data', 'train', 'backtest', 'paper_trade', 'live_trade', 'retrain', 'status'],
        help='Operation mode'
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to config file'
    )

    parser.add_argument(
        '--days',
        type=int,
        default=None,
        help='Number of days of data to fetch'
    )

    parser.add_argument(
        '--start',
        type=str,
        default=None,
        help='Start date for backtest (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--end',
        type=str,
        default=None,
        help='End date for backtest (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force operation (e.g., force retrain)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def mode_fetch_data(config, args, logger):
    """Fetch historical data from MT5."""
    from data.data_fetcher import DataFetcher
    from data.data_store import DataStore
    from data.data_validator import DataValidator

    logger.info("=" * 60)
    logger.info("FETCH DATA MODE")
    logger.info("=" * 60)

    fetcher = DataFetcher(config)
    store = DataStore(config)
    validator = DataValidator(config)

    # Determine date range
    days = args.days or config.training.training_window_days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    logger.info(f"Fetching {config.trading.symbol} {config.trading.timeframe}")
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")

    # Connect and fetch
    if not fetcher.connect():
        logger.warning("Could not connect to MT5, using demo data")
    else:
        # Try to find the correct gold symbol
        gold_symbol = fetcher.find_gold_symbol()
        if gold_symbol and gold_symbol != config.trading.symbol:
            logger.info(f"Using symbol '{gold_symbol}' instead of '{config.trading.symbol}'")
            fetcher.symbol = gold_symbol

        # Show available gold-related symbols
        available = fetcher.list_available_symbols("XAU")
        if available:
            logger.info(f"Available gold symbols: {available}")

    df = fetcher.fetch_historical(start_date, end_date)
    logger.info(f"Fetched {len(df)} bars")

    if df.empty:
        logger.error("No data fetched! Check MT5 connection and symbol availability.")
        logger.info("Possible causes:")
        logger.info("  1. MT5 terminal not running or not logged in")
        logger.info("  2. Symbol 'XAUUSD' not available (try 'GOLD' or broker-specific name)")
        logger.info("  3. No historical data available for the requested period")
        fetcher.disconnect()
        return

    # Validate
    is_valid, issues = validator.validate(df)
    if not is_valid:
        logger.warning("Data validation issues found, cleaning...")
        df = validator.clean(df)

    # Save
    store.save_historical(df)

    # Print info
    info = store.get_data_info()
    logger.info(f"Data saved: {info['rows']} bars")
    if info.get('first_date') and info.get('last_date'):
        logger.info(f"Date range: {info['first_date']} to {info['last_date']}")

    fetcher.disconnect()


def mode_train(config, args, logger):
    """Train a new model."""
    from data.data_store import DataStore
    from features.feature_engineering import FeatureEngineer
    from models.model_trainer import ModelTrainer
    from models.model_evaluator import ModelEvaluator
    from models.model_manager import ModelManager

    logger.info("=" * 60)
    logger.info("TRAIN MODE")
    logger.info("=" * 60)

    store = DataStore(config)
    engineer = FeatureEngineer(config)
    trainer = ModelTrainer(config)
    evaluator = ModelEvaluator(config)
    manager = ModelManager(config)

    # Load data
    df = store.load_historical()
    if df.empty:
        logger.error("No data found. Run fetch_data first.")
        return

    logger.info(f"Loaded {len(df)} bars")

    # Feature engineering
    logger.info("Building features...")
    df_features = engineer.build_features(df, include_labels=True)
    X, y, feature_names = engineer.prepare_training_data(df_features)
    logger.info(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")

    # Train with CV
    logger.info("Training model with time-series CV...")
    model, cv_metrics = trainer.train_with_cv(X, y, feature_names)

    logger.info(f"CV Accuracy: {cv_metrics['mean_accuracy']:.4f} (+/- {cv_metrics['std_accuracy']:.4f})")

    # Evaluate on holdout
    val_size = int(len(X) * 0.2)
    df_clean = df_features.dropna(subset=feature_names + ['label', 'future_return'])
    future_returns = df_clean['future_return'].values

    metrics = evaluator.evaluate(
        model, X[-val_size:], y[-val_size:], future_returns[-val_size:]
    )

    logger.info(evaluator.generate_report(metrics))

    # Save model
    model_path = manager.save_model(
        model,
        metrics=metrics,
        tag="production",
        feature_names=feature_names
    )
    logger.info(f"Model saved: {model_path}")


def mode_backtest(config, args, logger):
    """Run backtest simulation."""
    from data.data_store import DataStore
    from features.feature_engineering import FeatureEngineer
    from models.model_manager import ModelManager
    from backtesting.backtester import Backtester

    logger.info("=" * 60)
    logger.info("BACKTEST MODE")
    logger.info("=" * 60)

    store = DataStore(config)
    engineer = FeatureEngineer(config)
    manager = ModelManager(config)
    backtester = Backtester(config)

    # Load model
    try:
        model, metadata = manager.load_production()
        feature_names = metadata.get('feature_names', [])
        logger.info(f"Loaded model: {metadata.get('model_name')}")
    except FileNotFoundError:
        logger.error("No production model found. Run train first.")
        return

    # Load and prepare data
    df = store.load_historical()
    df_features = engineer.build_features(df, include_labels=False)

    # Parse date range
    start_date = datetime.strptime(args.start, '%Y-%m-%d') if args.start else None
    end_date = datetime.strptime(args.end, '%Y-%m-%d') if args.end else None

    # Run backtest
    logger.info("Running backtest...")
    results = backtester.run(df_features, model, feature_names, start_date, end_date)

    # Print report
    logger.info("\n" + backtester.generate_report(results))

    # Save results
    saved = backtester.save_results(results)
    logger.info(f"Results saved to: {saved.get('report')}")


def mode_paper_trade(config, args, logger):
    """Run paper trading on demo account."""
    _run_trading_loop(config, logger, is_live=False)


def mode_live_trade(config, args, logger):
    """Run live trading on real account."""
    logger.warning("=" * 60)
    logger.warning("LIVE TRADING MODE - REAL MONEY AT RISK")
    logger.warning("=" * 60)

    # Confirmation
    confirm = input("Type 'CONFIRM' to start live trading: ")
    if confirm != 'CONFIRM':
        logger.info("Live trading cancelled")
        return

    _run_trading_loop(config, logger, is_live=True)


def _run_trading_loop(config, logger, is_live: bool):
    """Main trading loop for paper/live trading."""
    from data.data_fetcher import DataFetcher
    from features.feature_engineering import FeatureEngineer
    from models.model_manager import ModelManager
    from strategy.signal_generator import SignalGenerator
    from strategy.position_sizing import PositionSizer
    from risk.risk_manager import RiskManager, AccountState
    from execution.broker_mt5 import MT5Broker
    from monitoring.performance_tracker import PerformanceTracker

    mode_str = "LIVE" if is_live else "PAPER"
    logger.info(f"Starting {mode_str} trading loop...")

    # Initialize components
    fetcher = DataFetcher(config)
    engineer = FeatureEngineer(config)
    manager = ModelManager(config)
    signal_gen = SignalGenerator(config)
    sizer = PositionSizer(config)
    risk_mgr = RiskManager(config)
    broker = MT5Broker(config)
    tracker = PerformanceTracker(config)

    # Load model
    try:
        model, metadata = manager.load_production()
        feature_names = metadata.get('feature_names', [])
        logger.info(f"Loaded model: {metadata.get('model_name')}")
    except FileNotFoundError:
        logger.error("No production model found. Run train first.")
        return

    # Connect to broker
    if not broker.connect():
        logger.error("Failed to connect to MT5")
        return

    logger.info("Connected to MT5")
    account = broker.get_account_info()
    logger.info(f"Account: {account.login}, Balance: ${account.balance:,.2f}")

    # Calculate bar interval
    tf_minutes = get_timeframe_minutes(config.trading.timeframe)
    bar_seconds = tf_minutes * 60

    logger.info(f"Trading loop started. Timeframe: {config.trading.timeframe} ({tf_minutes} min)")
    logger.info("Press Ctrl+C to stop")

    try:
        while True:
            try:
                # Wait for next bar
                now = datetime.now()
                seconds_into_bar = (now.minute % tf_minutes) * 60 + now.second
                wait_seconds = bar_seconds - seconds_into_bar + 5  # 5 second buffer

                if wait_seconds > 0 and wait_seconds < bar_seconds:
                    logger.debug(f"Waiting {wait_seconds}s for next bar...")
                    time.sleep(wait_seconds)

                # Fetch latest data
                df = fetcher.fetch_latest_bars(engineer.get_minimum_bars_required())
                if df.empty:
                    logger.warning("No data received")
                    time.sleep(30)
                    continue

                # Build features for latest bar
                df_features = engineer.build_features(df, include_labels=False)
                X, _ = engineer.prepare_inference_data(df_features)

                if X.size == 0:
                    logger.warning("No valid features")
                    continue

                # Get prediction
                prediction = model.predict(X)[0]
                proba = model.predict_proba(X)[0]
                max_prob = proba.max()

                latest_bar = df_features.iloc[-1]

                # Generate signal
                market_conditions = {
                    'hour': latest_bar.get('hour'),
                    'atr': latest_bar.get('atr'),
                    'atr_percentile': latest_bar.get('vol_percentile'),
                    'ema_trend': 1 if latest_bar.get('ema_8_21_cross', 0) == 1 else -1,
                    'is_overlap_session': latest_bar.get('is_overlap_session', 0) == 1,
                }

                signal = signal_gen.generate_signal(prediction, max_prob, market_conditions)

                # Log decision
                log_decision(
                    bar_time=latest_bar['time'],
                    prediction=prediction,
                    probability=max_prob,
                    signal=signal.direction,
                    reason=signal.reason,
                    atr=market_conditions.get('atr', 0)
                )

                # Get account state
                account = broker.get_account_info()
                open_positions = broker.get_open_positions()

                account_state = AccountState(
                    balance=account.balance,
                    equity=account.equity,
                    margin_used=account.margin,
                    open_positions=len(open_positions),
                    open_lots=sum(p.volume for p in open_positions),
                    daily_pnl=tracker.get_daily_summary()['pnl'],
                    weekly_pnl=tracker.get_weekly_summary()['pnl'],
                    consecutive_losses=0,  # Would track this properly
                    starting_balance_today=account.balance,  # Should store this
                    starting_balance_week=account.balance,
                )

                # Record equity
                tracker.record_equity(
                    balance=account.balance,
                    equity=account.equity,
                    open_pnl=account.profit,
                    open_positions=len(open_positions)
                )

                # Check if we should trade
                if signal.is_tradeable:
                    # Risk check
                    risk_check = risk_mgr.can_open_trade(account_state)

                    if not risk_check.approved:
                        logger.info(f"Trade blocked: {risk_check.reason}")
                        continue

                    # Calculate position size
                    atr = latest_bar.get('atr', 5.0)
                    sl_distance = atr * config.risk.sl_atr_multiplier
                    position = sizer.calculate_lot_size(
                        balance=account.balance,
                        sl_distance=sl_distance,
                        signal_strength=signal.strength
                    )

                    if not position.is_valid:
                        logger.info(f"Position rejected: {position.rejection_reason}")
                        continue

                    # Calculate SL/TP
                    bid, ask = broker.get_current_price()
                    entry_price = ask if signal.direction == 1 else bid
                    sl, tp = sizer.calculate_sl_tp(entry_price, signal.direction, atr)

                    # Place order
                    logger.info(f"Placing {'BUY' if signal.direction == 1 else 'SELL'} "
                                f"{position.lot_size} @ {entry_price:.2f}, SL={sl:.2f}, TP={tp:.2f}")

                    result = broker.place_market_order(
                        direction=signal.direction,
                        lot_size=position.lot_size,
                        sl=sl,
                        tp=tp,
                        comment=f"AI_{datetime.now().strftime('%H%M')}"
                    )

                    if result.success:
                        log_trade(
                            action="OPEN",
                            symbol=config.trading.symbol,
                            direction="BUY" if signal.direction == 1 else "SELL",
                            lot_size=position.lot_size,
                            price=result.price,
                            sl=sl,
                            tp=tp,
                            ticket=result.ticket
                        )
                    else:
                        logger.error(f"Order failed: {result.message}")

                # Check for closed positions (would need proper tracking)
                # This is simplified - real implementation would track position lifecycle

                # Sleep until next check
                time.sleep(10)

            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.error(f"Trading loop error: {e}", exc_info=True)
                time.sleep(60)

    except KeyboardInterrupt:
        logger.info("Trading loop stopped by user")

    finally:
        broker.disconnect()
        logger.info("Disconnected from MT5")


def mode_retrain(config, args, logger):
    """Run model retraining cycle."""
    from retraining.retrain_scheduler import RetrainScheduler

    logger.info("=" * 60)
    logger.info("RETRAIN MODE")
    logger.info("=" * 60)

    scheduler = RetrainScheduler(config)
    result = scheduler.run_retrain_cycle(force=args.force)

    logger.info(f"Retrain completed: success={result['success']}, deployed={result['deployed']}")


def mode_status(config, args, logger):
    """Show system status."""
    from data.data_store import DataStore
    from models.model_manager import ModelManager
    from monitoring.performance_tracker import PerformanceTracker

    logger.info("=" * 60)
    logger.info("SYSTEM STATUS")
    logger.info("=" * 60)

    # Data status
    store = DataStore(config)
    data_info = store.get_data_info()
    print("\nDATA:")
    for k, v in data_info.items():
        print(f"  {k}: {v}")

    # Model status
    manager = ModelManager(config)
    print("\n" + manager.get_model_summary())

    # Performance status
    tracker = PerformanceTracker(config)
    tracker.load_trades_from_db(days=30)
    print("\nRECENT PERFORMANCE:")
    print(tracker.generate_report('all'))


def main():
    """Main entry point."""
    args = parse_args()

    # Load config
    config = get_config(args.config)

    # Setup logging
    level = "DEBUG" if args.verbose else config.logging.level
    setup_logging(config, level=level)
    logger = get_logger(__name__)

    logger.info(f"AI Trading Bot starting - Mode: {args.mode}")
    logger.info(f"Symbol: {config.trading.symbol}, Timeframe: {config.trading.timeframe}")

    # Dispatch to mode handler
    mode_handlers = {
        'fetch_data': mode_fetch_data,
        'train': mode_train,
        'backtest': mode_backtest,
        'paper_trade': mode_paper_trade,
        'live_trade': mode_live_trade,
        'retrain': mode_retrain,
        'status': mode_status,
    }

    handler = mode_handlers.get(args.mode)
    if handler:
        try:
            handler(config, args, logger)
        except Exception as e:
            logger.error(f"Error in {args.mode}: {e}", exc_info=True)
            sys.exit(1)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)

    logger.info("Done")


if __name__ == "__main__":
    main()
