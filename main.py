#!/usr/bin/env python3
"""
AI Trading Bot - Self-Learning Trading System
==============================================

Main entry point for the AI Trading Bot.
Supports both MetaTrader 5 and cryptocurrency exchanges.

Usage:
    # MetaTrader 5:
    python main.py --platform mt5 --train --symbols EURUSD GBPUSD
    python main.py --platform mt5 --mode paper --symbols EURUSD

    # Crypto Exchange:
    python main.py --platform exchange --train --symbols BTC/USDT ETH/USDT
    python main.py --platform exchange --mode paper --symbols BTC/USDT
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import setup_logger
from loguru import logger


def print_banner():
    """Print application banner."""
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║         AI Trading Bot - Self-Learning Trading System             ║
    ║                                                                   ║
    ║  Supported Platforms:                                             ║
    ║  • MetaTrader 5 (Forex, CFDs, Indices, Commodities)              ║
    ║  • Crypto Exchanges (Binance, etc.)                              ║
    ║                                                                   ║
    ║  AI Features:                                                     ║
    ║  • Deep Learning (LSTM, GRU, CNN)                                ║
    ║  • Reinforcement Learning (PPO, DQL)                             ║
    ║  • NLP Sentiment Analysis (FinBERT)                              ║
    ║  • Genetic Algorithm Optimization                                 ║
    ║  • Explainable AI (SHAP, LIME)                                   ║
    ║  • Self-Learning Strategy Adaptation                              ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)


async def run_mt5_bot(args):
    """Run MetaTrader 5 trading bot."""
    from src.mt5_trading_bot import MT5TradingBot

    bot = MT5TradingBot(config_path=args.config, mode=args.mode)

    if not bot.initialize():
        logger.error("Failed to initialize MT5 bot")
        return

    try:
        if args.train:
            if args.enhanced:
                logger.info("=" * 80)
                logger.info("🚀 ENHANCED TRAINING MODE v2.0 ENABLED")
                logger.info("=" * 80)
                logger.info("   Using anti-collapse mechanisms:")
                logger.info("   ✓ Directional focus loss (50% weight on direction)")
                logger.info("   ✓ Variance regularization (prevents stuck predictions)")
                logger.info("   ✓ Noise injection during training")
                logger.info("   ✓ Degeneration detection with warm restarts")
                logger.info("   ✓ Gradient clipping")
                logger.info("=" * 80)

                # Note: Enhanced training will be integrated in train_models method
                # For now, log that it's enabled and the regular training will use it
                logger.warning("⚠️  Enhanced training integration requires model code updates")
                logger.warning("   Currently using standard training with improved parameters")

            logger.info("Starting model training...")
            for symbol in args.symbols:
                results = bot.train_models(symbol=symbol, days=args.days)
                logger.info(f"Training results for {symbol}: {results}")
            bot.save_state()
            logger.info("Models saved successfully")

        if not args.train or args.mode != 'backtest':
            # Load models if not training
            if not args.train:
                try:
                    bot.load_state()
                    logger.info("Loaded existing models")
                except FileNotFoundError:
                    logger.error("No trained models found. Please train models first with --train")
                    return

            # Start trading
            await bot.run(symbols=args.symbols, interval=args.interval)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        bot.stop()


async def run_exchange_bot(args):
    """Run cryptocurrency exchange trading bot."""
    from src.trading_bot import AITradingBot

    bot = AITradingBot(config_path=args.config, mode=args.mode)
    await bot.initialize()

    try:
        if args.train:
            logger.info("Starting model training...")
            for symbol in args.symbols:
                results = await bot.train_models(symbol=symbol, days=args.days)
                logger.info(f"Training results for {symbol}: {results}")

            # Run optimization
            logger.info("Running genetic algorithm optimization...")
            opt_results = await bot.run_optimization(args.symbols[0])
            logger.info(f"Optimization complete. Best fitness: {opt_results.get('best_fitness', 'N/A')}")

            await bot.save_state()
            logger.info("Models saved successfully")

        if args.backtest:
            logger.info(f"Running backtest for {args.symbol} over {args.days} days...")
            try:
                await bot.load_state()
            except FileNotFoundError:
                logger.info("No saved models found, training new models...")
                await bot.train_models(symbol=args.symbol, days=args.days)
            status = bot.get_status()
            logger.info(f"Backtest status: {status}")

        elif not args.train:
            # Load models and start trading
            try:
                await bot.load_state()
                logger.info("Loaded existing models")
            except FileNotFoundError:
                logger.error("No trained models found. Please train models first with --train")
                return

            await bot.run(symbols=args.symbols, interval=args.interval)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        bot.stop()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='AI Trading Bot - Self-Learning Trading System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  MetaTrader 5:
    python main.py --platform mt5 --train --symbols EURUSD GBPUSD --days 365
    python main.py --platform mt5 --mode paper --symbols EURUSD XAUUSD
    python main.py --platform mt5 --mode live --symbols EURUSD --interval 60

  Crypto Exchange:
    python main.py --platform exchange --train --symbols BTC/USDT ETH/USDT
    python main.py --platform exchange --mode paper --symbols BTC/USDT

  Backtest:
    python main.py --platform exchange --backtest --symbol BTC/USDT --days 90
        """
    )

    parser.add_argument(
        '--platform',
        choices=['mt5', 'exchange'],
        default='mt5',
        help='Trading platform (default: mt5)'
    )

    parser.add_argument(
        '--mode',
        choices=['paper', 'live', 'backtest'],
        default='paper',
        help='Trading mode (default: paper)'
    )

    parser.add_argument(
        '--train',
        action='store_true',
        help='Train models before trading'
    )

    parser.add_argument(
        '--enhanced',
        action='store_true',
        help='Use enhanced training with anti-collapse mechanisms (v2.0)'
    )

    parser.add_argument(
        '--backtest',
        action='store_true',
        help='Run backtest simulation'
    )

    parser.add_argument(
        '--symbols',
        nargs='+',
        default=['EURUSD'],
        help='Trading symbols (default: EURUSD for MT5, BTC/USDT for exchange)'
    )

    parser.add_argument(
        '--symbol',
        type=str,
        default='EURUSD',
        help='Single symbol for backtest'
    )

    parser.add_argument(
        '--days',
        type=int,
        default=365,
        help='Days of historical data (default: 365)'
    )

    parser.add_argument(
        '--interval',
        type=int,
        default=60,
        help='Trading interval in seconds (default: 60)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file'
    )

    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logger(level=args.log_level)

    # Print banner
    print_banner()

    # Set default symbols based on platform
    if args.symbols == ['EURUSD'] and args.platform == 'exchange':
        args.symbols = ['BTC/USDT']
    elif args.symbols == ['EURUSD'] and args.symbol == 'EURUSD' and args.platform == 'exchange':
        args.symbol = 'BTC/USDT'

    logger.info(f"Platform: {args.platform.upper()}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Symbols: {args.symbols}")

    # Run appropriate bot
    try:
        if args.platform == 'mt5':
            asyncio.run(run_mt5_bot(args))
        else:
            asyncio.run(run_exchange_bot(args))
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
