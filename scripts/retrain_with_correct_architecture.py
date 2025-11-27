"""
Retrain models with correct, consistent architecture
Fixes the layer mismatch issue

This script:
1. Backs up old models
2. Retrains all models with consistent architecture
3. Verifies models can be loaded
4. Provides detailed training summary
"""

import asyncio
import logging
import sys
from pathlib import Path
import shutil
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def retrain_models():
    """Retrain all models with consistent architecture"""

    logger.info("=" * 90)
    logger.info("🔄 RETRAINING MODELS WITH CORRECT ARCHITECTURE")
    logger.info("=" * 90)
    logger.info("")
    logger.info("This will:")
    logger.info("1. Backup existing models")
    logger.info("2. Train new models with consistent architecture")
    logger.info("3. Verify new models load correctly")
    logger.info("4. Save models with correct metadata (v2.1)")
    logger.info("")

    # Check if backup is needed
    models_dir = Path("saved_models")
    if models_dir.exists() and any(models_dir.glob("*.pt")):
        # Create backup
        backup_dir = Path(f"saved_models_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        backup_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"📦 Backing up old models to {backup_dir}")
        for model_file in models_dir.glob("*"):
            if model_file.is_file():
                dest = backup_dir / model_file.name
                shutil.copy2(model_file, dest)
                logger.info(f"   Backed up: {model_file.name}")
        logger.info("✅ Backup complete")
        logger.info("")
    else:
        logger.info("No existing models found - starting fresh")
        logger.info("")

    # Import bot (after adding to path)
    try:
        from src.mt5_trading_bot import MT5TradingBot
        from src.config import Config
    except ImportError as e:
        logger.error(f"Failed to import: {e}")
        logger.error("Make sure you're running from the project root directory")
        return

    # Initialize bot
    logger.info("🤖 Initializing trading bot...")
    config = Config()

    bot = MT5TradingBot(
        symbol='XAUUSD',
        initial_balance=10000,
        paper_trading=True
    )

    # Initialize MT5 connection
    try:
        await bot.initialize()
        logger.info("✅ Bot initialized")
        logger.info("")
    except Exception as e:
        logger.error(f"Failed to initialize bot: {e}")
        logger.error("Make sure MetaTrader 5 is running and credentials are configured")
        return

    # Train with CONSISTENT architecture
    logger.info("=" * 90)
    logger.info("🚀 STARTING TRAINING WITH FIXED ARCHITECTURE")
    logger.info("=" * 90)
    logger.info("")
    logger.info("Training parameters:")
    logger.info(f"  Symbol: XAUUSD")
    logger.info(f"  Days of data: 365")
    logger.info(f"  AttentionLSTM: 3 layers, 128 hidden, attention heads: 4")
    logger.info(f"  GRU: 2 layers, 128 hidden")
    logger.info(f"  DQL: Compact state space")
    logger.info("")

    try:
        results = bot.train_models(
            symbol='XAUUSD',
            days=365  # 1 year of data
        )

        logger.info("")
        logger.info("=" * 90)
        logger.info("✅ RETRAINING COMPLETE")
        logger.info("=" * 90)
        logger.info("")
        logger.info("Training Results:")
        logger.info(f"  AttentionLSTM:")
        logger.info(f"    - Directional Accuracy: {results['lstm']['directional_accuracy']:.2%}")
        logger.info(f"    - Correlation: {results['lstm']['correlation']:.4f}")
        logger.info(f"    - MSE: {results['lstm']['mse']:.6f}")
        logger.info("")
        logger.info(f"  GRU:")
        logger.info(f"    - Directional Accuracy: {results['gru']['directional_accuracy']:.2%}")
        logger.info(f"    - Correlation: {results['gru']['correlation']:.4f}")
        logger.info(f"    - MSE: {results['gru']['mse']:.6f}")
        logger.info("")
        logger.info(f"  DQL:")
        logger.info(f"    - Mean Return: {results['dql']['mean_return']:.2%}")
        logger.info(f"    - Win Rate: {results['dql']['mean_win_rate']:.2%}")
        logger.info(f"    - Sharpe Ratio: {results['dql']['mean_sharpe']:.4f}")
        logger.info("")
        logger.info("=" * 90)

    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error("", exc_info=True)
        await bot.stop()
        return

    # Verify models can be loaded
    logger.info("")
    logger.info("🔍 VERIFYING MODELS CAN BE LOADED...")
    logger.info("")

    try:
        # Create fresh bot instance
        verification_bot = MT5TradingBot(
            symbol='XAUUSD',
            initial_balance=10000,
            paper_trading=True
        )

        # Load models
        verification_bot.load_state()

        logger.info("✅ All models loaded successfully!")
        logger.info("")

        # Check compatibility
        from src.utils.model_compatibility import ModelCompatibilityChecker

        logger.info("Running compatibility checks...")
        results = ModelCompatibilityChecker.check_all_models()

        all_compatible = all(r['compatible'] for r in results.values())
        if all_compatible:
            logger.info("✅ All models passed compatibility checks!")
        else:
            logger.warning("⚠️  Some models have compatibility warnings")

    except Exception as e:
        logger.error(f"Verification failed: {e}")
        logger.error("", exc_info=True)
        await bot.stop()
        return

    # Cleanup
    await bot.stop()

    logger.info("")
    logger.info("=" * 90)
    logger.info("🎉 RETRAINING COMPLETED SUCCESSFULLY")
    logger.info("=" * 90)
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Review training results above")
    logger.info("2. Start live trading with: python main.py --mode live")
    logger.info("3. Monitor predictions and signals in logs")
    logger.info("")
    logger.info("Models saved to: saved_models/")
    logger.info(f"Backup available at: {backup_dir if 'backup_dir' in locals() else 'N/A'}")
    logger.info("=" * 90)


if __name__ == "__main__":
    try:
        asyncio.run(retrain_models())
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\n❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)
