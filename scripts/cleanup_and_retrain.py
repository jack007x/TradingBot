#!/usr/bin/env python3
"""
Clean Up Old Models and Retrain
================================

CRITICAL: Old models have issues:
1. No TradingLoss (still conservative)
2. Old format (compatibility issues)
3. Poor performance (GRU < 50% accuracy)

This script:
1. Backs up old models
2. Cleans up old checkpoints
3. Retrains with TradingLoss enabled
4. Validates new models
"""

import shutil
from pathlib import Path
from datetime import datetime
from loguru import logger
import sys
import os

# Add parent directory to path
# Script is in scripts/ so we need to go up one level to get to project root
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

# Add project root to Python path
sys.path.insert(0, str(project_root))

# Change to project root directory so relative paths work
os.chdir(project_root)

def backup_old_models():
    """Backup old models before deletion."""
    saved_models_dir = Path('saved_models')

    if not saved_models_dir.exists():
        logger.info("No saved_models directory found - nothing to backup")
        return

    # Create backup directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = Path(f'saved_models_backup_{timestamp}')

    logger.info(f"Backing up old models to {backup_dir}")

    # Copy entire directory
    shutil.copytree(saved_models_dir, backup_dir)

    logger.info(f"✅ Backup complete: {backup_dir}")
    logger.info(f"   You can restore from backup if needed")

    return backup_dir


def clean_old_models():
    """Delete old model checkpoints."""
    saved_models_dir = Path('saved_models')

    if not saved_models_dir.exists():
        logger.info("No saved_models directory - nothing to clean")
        return

    # List of old model files to delete
    old_models = [
        'lstm_model.pt',
        'gru_model.pt',
        'dql_agent.pt',
        'cnn_model.pt',
        'directional_lstm.pt',
        'directional_gru.pt'
    ]

    deleted_count = 0
    for model_file in old_models:
        model_path = saved_models_dir / model_file
        if model_path.exists():
            logger.info(f"Deleting: {model_path}")
            model_path.unlink()
            deleted_count += 1

    logger.info(f"✅ Cleaned up {deleted_count} old model files")


def validate_new_models():
    """Check if new models were created successfully."""
    saved_models_dir = Path('saved_models')

    required_models = [
        'lstm_model.pt',
        'gru_model.pt'
    ]

    all_present = True
    for model_file in required_models:
        model_path = saved_models_dir / model_file
        if model_path.exists():
            logger.info(f"✅ Found: {model_path}")
        else:
            logger.error(f"❌ Missing: {model_path}")
            all_present = False

    return all_present


def print_instructions():
    """Print retraining instructions."""
    print("\n" + "="*60)
    print("NEXT STEPS: Retrain Models with TradingLoss")
    print("="*60)
    print()
    print("Run the following command to retrain:")
    print()
    print("  python main.py --platform mt5 \\")
    print("                 --train \\")
    print("                 --symbols XAUUSD \\")
    print("                 --days 3650  # 10 years of data")
    print()
    print("This will:")
    print("  ✅ Use TradingLoss (fixes conservatism)")
    print("  ✅ Train on 10 years of data")
    print("  ✅ Save models in new v2.0 format")
    print("  ✅ Enable signal filtering")
    print("  ✅ Enable enhanced risk management")
    print()
    print("Expected improvements:")
    print("  - Predicted std ratio: 0.05 → 0.60-0.80")
    print("  - Directional accuracy: 50% → 55-58%")
    print("  - Better capture of large moves")
    print()
    print("Monitor logs for:")
    print("  - 'Pred/Target Std Ratio' should be 0.6-0.9")
    print("  - 'Directional Accuracy' should be >52%")
    print("  - 'Using TRADING loss' confirmation")
    print()
    print("="*60)
    print()


def main():
    """Main cleanup and retrain workflow."""
    logger.info("="*60)
    logger.info("MODEL CLEANUP & RETRAIN UTILITY")
    logger.info("="*60)

    # Step 1: Backup old models
    logger.info("\n📦 Step 1: Backing up old models...")
    backup_dir = backup_old_models()

    # Step 2: Clean old models
    logger.info("\n🗑️  Step 2: Cleaning old model files...")
    clean_old_models()

    # Step 3: Print instructions
    logger.info("\n📋 Step 3: Retrain models with new settings...")
    print_instructions()

    # Ask for confirmation
    print("Would you like to start retraining now? (y/n): ", end='')
    response = input().strip().lower()

    if response == 'y':
        logger.info("Starting retraining...")

        # Import and run training
        from src.mt5_trading_bot import MT5TradingBot

        bot = MT5TradingBot(mode='paper')

        if not bot.initialize():
            logger.error("Failed to initialize bot")
            return 1

        # Train models
        results = bot.train_models(
            symbol='XAUUSD',
            timeframe='1h',
            days=3650  # 10 years
        )

        # Validate results
        logger.info("\n📊 Training Results:")
        logger.info(f"  LSTM: {results.get('lstm', {})}")
        logger.info(f"  GRU: {results.get('gru', {})}")

        # Save models
        bot.save_state()

        # Validate saved models
        if validate_new_models():
            logger.info("\n✅ SUCCESS: New models trained and saved!")
            logger.info("   Models are now using TradingLoss")
            logger.info("   Ready for paper trading")
        else:
            logger.error("\n❌ ERROR: Some models failed to save")
            return 1
    else:
        logger.info("Skipping automatic retraining")
        logger.info("Run the command manually when ready")

    return 0


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
