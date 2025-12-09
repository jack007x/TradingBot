"""
Retrain Scheduler Module
=========================
Handles scheduled model retraining with validation and safe deployment.

Usage:
    from retraining import RetrainScheduler
    from config import get_config

    config = get_config()
    scheduler = RetrainScheduler(config)

    # Run retrain cycle
    result = scheduler.run_retrain_cycle()

    # Schedule daily retraining
    scheduler.start_scheduler()
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
import json

from config.config_loader import Config
from data.data_fetcher import DataFetcher
from data.data_store import DataStore
from data.data_validator import DataValidator
from features.feature_engineering import FeatureEngineer
from models.model_trainer import ModelTrainer
from models.model_evaluator import ModelEvaluator
from models.model_manager import ModelManager
from models.model_baseline import BaselineModel

logger = logging.getLogger(__name__)


class RetrainScheduler:
    """
    Manages scheduled model retraining.

    Process:
    1. Update data (fetch new bars)
    2. Train new model on rolling window
    3. Evaluate new model
    4. Compare with current production model
    5. Deploy if new model passes criteria
    """

    def __init__(self, config: Config):
        """
        Initialize retrain scheduler.

        Args:
            config: Configuration object
        """
        self.config = config

        # Initialize components
        self.fetcher = DataFetcher(config)
        self.store = DataStore(config)
        self.validator = DataValidator(config)
        self.engineer = FeatureEngineer(config)
        self.trainer = ModelTrainer(config)
        self.evaluator = ModelEvaluator(config)
        self.manager = ModelManager(config)

        # Retraining parameters
        self.training_window_days = config.training.training_window_days
        self.retrain_frequency = config.training.retrain_frequency_days
        self.retrain_hour = config.training.retrain_hour_utc

        # Last retrain tracking
        self._last_retrain: Optional[datetime] = None
        self._retrain_history: list = []

    def run_retrain_cycle(self, force: bool = False) -> Dict[str, Any]:
        """
        Run a complete retrain cycle.

        Args:
            force: Force retrain even if not scheduled

        Returns:
            Dict with retrain results
        """
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("Starting retrain cycle")
        logger.info("=" * 60)

        result = {
            'started_at': start_time.isoformat(),
            'success': False,
            'deployed': False,
            'steps': {}
        }

        try:
            # Step 1: Update data
            logger.info("Step 1: Updating data...")
            data_result = self._update_data()
            result['steps']['data_update'] = data_result

            if not data_result['success']:
                result['message'] = "Data update failed"
                return result

            # Step 2: Load and prepare data
            logger.info("Step 2: Loading and preparing data...")
            df = self.store.load_historical()

            # Use rolling window
            end_date = df['time'].max()
            start_date = end_date - timedelta(days=self.training_window_days)
            df = df[df['time'] >= start_date].copy()

            logger.info(f"Training data: {len(df)} bars from {start_date.date()} to {end_date.date()}")

            # Step 3: Feature engineering
            logger.info("Step 3: Building features...")
            df_features = self.engineer.build_features(df, include_labels=True)

            # Prepare training data
            X, y, feature_names = self.engineer.prepare_training_data(df_features)

            if len(X) < 1000:
                result['message'] = f"Insufficient data: {len(X)} samples"
                return result

            result['steps']['feature_engineering'] = {
                'samples': len(X),
                'features': len(feature_names)
            }

            # Step 4: Train new model
            logger.info("Step 4: Training new model...")
            new_model, cv_metrics = self.trainer.train_with_cv(X, y, feature_names)

            result['steps']['training'] = {
                'mean_accuracy': cv_metrics['mean_accuracy'],
                'std_accuracy': cv_metrics['std_accuracy'],
            }

            # Step 5: Evaluate new model
            logger.info("Step 5: Evaluating new model...")
            # Get future returns for trading metrics
            df_clean = df_features.dropna(subset=feature_names + ['label', 'future_return'])
            future_returns = df_clean['future_return'].values

            # Use last 20% as validation
            val_size = int(len(X) * 0.2)
            X_val = X[-val_size:]
            y_val = y[-val_size:]
            returns_val = future_returns[-val_size:]

            new_metrics = self.evaluator.evaluate(new_model, X_val, y_val, returns_val)
            result['steps']['evaluation'] = {
                'accuracy': new_metrics['accuracy'],
                'sharpe_ratio': new_metrics.get('sharpe_ratio', 0),
                'win_rate': new_metrics.get('win_rate', 0),
                'profit_factor': new_metrics.get('profit_factor', 0),
                'max_drawdown': new_metrics.get('max_drawdown', 0),
            }

            # Step 6: Compare with current model
            logger.info("Step 6: Comparing with current model...")
            deploy_decision = self._compare_and_decide(new_model, new_metrics, X_val, y_val, returns_val)
            result['steps']['comparison'] = deploy_decision

            # Step 7: Deploy if approved
            if deploy_decision['should_deploy']:
                logger.info("Step 7: Deploying new model...")
                model_path = self.manager.save_model(
                    new_model,
                    metrics=new_metrics,
                    tag="production",
                    feature_names=feature_names
                )
                result['deployed'] = True
                result['model_path'] = str(model_path)
                logger.info(f"New model deployed: {model_path}")
            else:
                logger.info("Step 7: Keeping current model (new model did not pass criteria)")
                # Save as checkpoint anyway for reference
                self.manager.save_model(
                    new_model,
                    metrics=new_metrics,
                    tag="checkpoint",
                    feature_names=feature_names
                )

            result['success'] = True
            result['message'] = "Retrain cycle completed"

        except Exception as e:
            logger.error(f"Retrain cycle failed: {e}", exc_info=True)
            result['message'] = str(e)
            result['success'] = False

        finally:
            result['completed_at'] = datetime.now().isoformat()
            result['duration_seconds'] = (datetime.now() - start_time).total_seconds()

            # Record in history
            self._retrain_history.append(result)
            self._last_retrain = datetime.now()

            # Cleanup old models
            self.manager.cleanup_old_models()

        logger.info(f"Retrain cycle completed in {result['duration_seconds']:.1f}s")
        logger.info(f"Success: {result['success']}, Deployed: {result['deployed']}")

        return result

    def _update_data(self) -> Dict[str, Any]:
        """Update historical data with latest bars."""
        try:
            # Get last timestamp in storage
            last_ts = self.store.get_last_timestamp()

            if last_ts is None:
                # No data, fetch from scratch
                start_date = datetime.now() - timedelta(days=self.training_window_days + 30)
            else:
                # Fetch from last timestamp
                start_date = last_ts - timedelta(hours=1)  # Small overlap

            end_date = datetime.now()

            # Fetch new data
            df_new = self.fetcher.fetch_historical(start_date, end_date)

            if df_new.empty:
                return {'success': True, 'message': 'No new data', 'new_bars': 0}

            # Validate
            is_valid, issues = self.validator.validate(df_new)
            if not is_valid:
                df_new = self.validator.clean(df_new)

            # Append to storage
            df_combined = self.store.append_data(df_new)

            return {
                'success': True,
                'new_bars': len(df_new),
                'total_bars': len(df_combined),
                'message': f'Added {len(df_new)} new bars'
            }

        except Exception as e:
            logger.error(f"Data update failed: {e}")
            return {'success': False, 'message': str(e)}

    def _compare_and_decide(
            self,
            new_model: BaselineModel,
            new_metrics: Dict[str, float],
            X_val: Any,
            y_val: Any,
            returns_val: Any
    ) -> Dict[str, Any]:
        """
        Compare new model with current and decide on deployment.
        """
        result = {
            'should_deploy': False,
            'reason': '',
            'new_metrics': new_metrics,
            'current_metrics': None,
        }

        try:
            # Load current production model
            current_model, current_meta = self.manager.load_production()
            current_metrics = current_meta.get('metrics', {})
            result['current_metrics'] = current_metrics

            # Use evaluator to compare
            should_deploy, comparison = self.evaluator.compare_models(
                current_metrics, new_metrics
            )

            result['should_deploy'] = should_deploy
            result['comparison'] = comparison
            result['reason'] = comparison.get('reason', '')

        except FileNotFoundError:
            # No current model, deploy if meets absolute thresholds
            logger.info("No current model found, checking absolute thresholds...")

            min_sharpe = self.config.retraining.min_sharpe_ratio
            max_dd = self.config.retraining.max_allowed_drawdown
            min_wr = self.config.retraining.min_win_rate
            min_pf = self.config.retraining.min_profit_factor

            checks = {
                'sharpe_ratio': new_metrics.get('sharpe_ratio', 0) >= min_sharpe,
                'max_drawdown': new_metrics.get('max_drawdown', 1) <= max_dd,
                'win_rate': new_metrics.get('win_rate', 0) >= min_wr,
                'profit_factor': new_metrics.get('profit_factor', 0) >= min_pf,
            }

            all_passed = all(checks.values())
            result['should_deploy'] = all_passed
            result['absolute_checks'] = checks
            result['reason'] = "First model - meets thresholds" if all_passed else "Does not meet minimum thresholds"

        return result

    def should_retrain(self) -> Tuple[bool, str]:
        """
        Check if retraining should be performed now.

        Returns:
            Tuple[should_retrain, reason]
        """
        now = datetime.utcnow()

        # Check if it's the right hour
        if now.hour != self.retrain_hour:
            return False, f"Not retrain hour (current: {now.hour}, scheduled: {self.retrain_hour})"

        # Check if we already retrained today
        if self._last_retrain is not None:
            if (now - self._last_retrain).total_seconds() < 3600:  # Within 1 hour
                return False, "Already retrained recently"

        # Check frequency
        if self._last_retrain is not None:
            days_since = (now - self._last_retrain).days
            if days_since < self.retrain_frequency:
                return False, f"Not due yet (last: {days_since} days ago, freq: {self.retrain_frequency})"

        return True, "Scheduled retrain time"

    def start_scheduler(self, blocking: bool = True):
        """
        Start the retraining scheduler.

        Args:
            blocking: If True, blocks and runs continuously
        """
        import schedule
        import time

        logger.info(f"Starting retrain scheduler (hour: {self.retrain_hour} UTC, "
                    f"frequency: every {self.retrain_frequency} days)")

        def retrain_job():
            should, reason = self.should_retrain()
            if should:
                logger.info(f"Retrain triggered: {reason}")
                self.run_retrain_cycle()
            else:
                logger.debug(f"Retrain skipped: {reason}")

        # Schedule job to run every hour
        schedule.every().hour.at(":00").do(retrain_job)

        if blocking:
            logger.info("Scheduler running. Press Ctrl+C to stop.")
            while True:
                schedule.run_pending()
                time.sleep(60)

    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status."""
        return {
            'last_retrain': self._last_retrain.isoformat() if self._last_retrain else None,
            'retrain_count': len(self._retrain_history),
            'next_retrain_hour': self.retrain_hour,
            'retrain_frequency_days': self.retrain_frequency,
            'training_window_days': self.training_window_days,
            'recent_results': self._retrain_history[-5:] if self._retrain_history else [],
        }


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from config import get_config

    config = get_config()
    scheduler = RetrainScheduler(config)

    print("Testing Retrain Scheduler...")
    print("=" * 60)

    # Run a retrain cycle
    print("\nRunning retrain cycle...")
    result = scheduler.run_retrain_cycle()

    print("\nResult:")
    print(f"  Success: {result['success']}")
    print(f"  Deployed: {result['deployed']}")
    print(f"  Duration: {result.get('duration_seconds', 0):.1f}s")

    if 'steps' in result:
        print("\nSteps:")
        for step, data in result['steps'].items():
            print(f"  {step}: {data}")

    # Get status
    print("\nScheduler Status:")
    status = scheduler.get_status()
    for key, value in status.items():
        if key != 'recent_results':
            print(f"  {key}: {value}")
