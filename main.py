#!/usr/bin/env python3
"""
AI Trading Bot - Self-Learning Trading System
==============================================

Main entry point for the AI Trading Bot.

Usage:
    python main.py --mode paper --train --symbols BTC/USDT ETH/USDT
    python main.py --mode live --load-models
    python main.py --backtest --symbol BTC/USDT --days 90
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.trading_bot import AITradingBot
from src.utils.logger import setup_logger
from loguru import logger


async def run_training(bot: AITradingBot, symbols: list, days: int):
    """Train all models."""
    logger.info("Starting model training...")

    for symbol in symbols:
        results = await bot.train_models(symbol=symbol, days=days)
        logger.info(f"Training results for {symbol}:")
        for model, metrics in results.items():
            logger.info(f"  {model}: {metrics}")

    # Run optimization
    logger.info("Running genetic algorithm optimization...")
    opt_results = await bot.run_optimization(symbols[0])
    logger.info(f"Optimization complete. Best fitness: {opt_results.get('best_fitness', 'N/A')}")

    # Save models
    await bot.save_state()
    logger.info("Models saved successfully")


async def run_backtest(bot: AITradingBot, symbol: str, days: int):
    """Run backtest simulation."""
    logger.info(f"Running backtest for {symbol} over {days} days...")

    # Load models if available
    try:
        await bot.load_state()
    except FileNotFoundError:
        logger.info("No saved models found, training new models...")
        await bot.train_models(symbol=symbol, days=days)

    # Backtest logic would go here
    # For now, just show status
    status = bot.get_status()
    logger.info(f"Backtest status: {status}")


async def run_live_trading(bot: AITradingBot, symbols: list, interval: int):
    """Run live/paper trading."""
    logger.info(f"Starting {'paper' if bot.mode == 'paper' else 'live'} trading...")

    # Load models
    try:
        await bot.load_state()
        logger.info("Loaded existing models")
    except FileNotFoundError:
        logger.error("No trained models found. Please train models first.")
        return

    # Start trading loop
    await bot.run(symbols=symbols, interval=interval)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='AI Trading Bot - Self-Learning Trading System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train models:
    python main.py --train --symbols BTC/USDT ETH/USDT --days 365

  Paper trading:
    python main.py --mode paper --symbols BTC/USDT

  Live trading (use with caution):
    python main.py --mode live --symbols BTC/USDT --interval 60

  Backtest:
    python main.py --backtest --symbol BTC/USDT --days 90
        """
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
        '--backtest',
        action='store_true',
        help='Run backtest simulation'
    )

    parser.add_argument(
        '--symbols',
        nargs='+',
        default=['BTC/USDT'],
        help='Trading symbols (default: BTC/USDT)'
    )

    parser.add_argument(
        '--symbol',
        type=str,
        default='BTC/USDT',
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
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║           AI Trading Bot - Self-Learning System               ║
    ║                                                               ║
    ║  Features:                                                    ║
    ║  • Deep Learning (LSTM, GRU, CNN)                            ║
    ║  • Reinforcement Learning (PPO, DQL)                         ║
    ║  • NLP Sentiment Analysis                                     ║
    ║  • Genetic Algorithm Optimization                             ║
    ║  • Explainable AI (SHAP, LIME)                               ║
    ║  • Self-Learning Strategy Adaptation                          ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

    # Create bot
    bot = AITradingBot(config_path=args.config, mode=args.mode)

    # Run async main
    async def async_main():
        await bot.initialize()

        if args.train:
            await run_training(bot, args.symbols, args.days)

        if args.backtest:
            await run_backtest(bot, args.symbol, args.days)
        elif not args.train:
            await run_live_trading(bot, args.symbols, args.interval)

    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        bot.stop()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
