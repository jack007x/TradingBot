"""
Model Manager Module
=====================
Handles model persistence, versioning, and deployment.

Usage:
    from models import ModelManager
    from config import get_config

    config = get_config()
    manager = ModelManager(config)

    # Save model
    model_path = manager.save_model(model, metrics, "production")

    # Load latest model
    model, metadata = manager.load_latest()

    # Rollback to previous version
    manager.rollback()
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import joblib

from config.config_loader import Config
from models.model_baseline import BaselineModel

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages model lifecycle: save, load, version, and deploy.

    Features:
    - Automatic versioning with timestamps
    - Metadata storage (metrics, training info)
    - Production/staging model slots
    - Rollback capability
    - Model comparison and selection
    """

    def __init__(self, config: Config):
        """
        Initialize model manager.

        Args:
            config: Configuration object
        """
        self.config = config
        self.models_dir = config.get_models_path()

        # Ensure directory exists
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Model file patterns
        self.model_prefix = "model"
        self.metadata_suffix = "_metadata.json"

    def save_model(
            self,
            model: BaselineModel,
            metrics: Optional[Dict[str, Any]] = None,
            tag: str = "checkpoint",
            feature_names: Optional[List[str]] = None
    ) -> Path:
        """
        Save model to disk with metadata.

        Args:
            model: Trained model
            metrics: Evaluation metrics
            tag: Model tag (e.g., "production", "staging", "checkpoint")
            feature_names: Feature names used for training

        Returns:
            Path: Path to saved model
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{self.model_prefix}_{tag}_{timestamp}"

        model_path = self.models_dir / f"{model_name}.pkl"
        metadata_path = self.models_dir / f"{model_name}{self.metadata_suffix}"

        # Save model
        joblib.dump(model, model_path)
        logger.info(f"Saved model to {model_path}")

        # Prepare metadata
        metadata = {
            "model_name": model_name,
            "tag": tag,
            "timestamp": timestamp,
            "created_at": datetime.now().isoformat(),
            "model_type": "LightGBM",
            "config": {
                "symbol": self.config.trading.symbol,
                "timeframe": self.config.trading.timeframe,
                "prediction_horizon": self.config.model.prediction_horizon_bars,
            },
            "training_info": model.training_info if hasattr(model, 'training_info') else {},
            "metrics": metrics or {},
            "feature_names": feature_names or model._feature_names or [],
        }

        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Saved metadata to {metadata_path}")

        # If tag is "production", update the production symlink/reference
        if tag == "production":
            self._set_production_model(model_path)

        return model_path

    def load_model(
            self,
            model_path: Optional[Path] = None,
            model_name: Optional[str] = None
    ) -> Tuple[BaselineModel, Dict[str, Any]]:
        """
        Load a specific model.

        Args:
            model_path: Full path to model file
            model_name: Model name (without extension)

        Returns:
            Tuple[model, metadata]
        """
        if model_path is None and model_name is None:
            raise ValueError("Either model_path or model_name must be provided")

        if model_path is None:
            model_path = self.models_dir / f"{model_name}.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Load model
        model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")

        # Load metadata
        metadata_path = model_path.parent / (model_path.stem + self.metadata_suffix)
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {"model_path": str(model_path)}
            logger.warning(f"Metadata not found for {model_path}")

        return model, metadata

    def load_latest(
            self,
            tag: Optional[str] = None
    ) -> Tuple[BaselineModel, Dict[str, Any]]:
        """
        Load the latest model, optionally filtered by tag.

        Args:
            tag: Filter by tag (e.g., "production")

        Returns:
            Tuple[model, metadata]
        """
        models = self.list_models(tag=tag)

        if not models:
            raise FileNotFoundError(f"No models found" + (f" with tag '{tag}'" if tag else ""))

        # Sort by timestamp (newest first)
        models.sort(key=lambda x: x['timestamp'], reverse=True)
        latest = models[0]

        return self.load_model(model_path=Path(latest['path']))

    def load_production(self) -> Tuple[BaselineModel, Dict[str, Any]]:
        """
        Load the current production model.

        Returns:
            Tuple[model, metadata]
        """
        prod_pointer = self.models_dir / "production_model.txt"

        if prod_pointer.exists():
            with open(prod_pointer, 'r') as f:
                model_path = Path(f.read().strip())
            return self.load_model(model_path=model_path)
        else:
            # Fallback to latest production-tagged model
            return self.load_latest(tag="production")

    def list_models(
            self,
            tag: Optional[str] = None,
            limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        List all saved models.

        Args:
            tag: Filter by tag
            limit: Maximum number of models to return

        Returns:
            List of model info dictionaries
        """
        models = []

        for model_file in self.models_dir.glob("*.pkl"):
            if model_file.name == "production_model.pkl":
                continue

            # Parse model info from filename
            # Format: model_{tag}_{timestamp}.pkl
            parts = model_file.stem.split('_')
            if len(parts) >= 3:
                model_tag = parts[1]
                timestamp = '_'.join(parts[2:])
            else:
                model_tag = "unknown"
                timestamp = ""

            # Filter by tag if specified
            if tag is not None and model_tag != tag:
                continue

            # Load metadata if available
            metadata_path = model_file.parent / (model_file.stem + self.metadata_suffix)
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {}

            models.append({
                "name": model_file.stem,
                "path": str(model_file),
                "tag": model_tag,
                "timestamp": timestamp,
                "size_mb": model_file.stat().st_size / (1024 * 1024),
                "metrics": metadata.get("metrics", {}),
                "created_at": metadata.get("created_at", ""),
            })

        # Sort by timestamp and limit
        models.sort(key=lambda x: x['timestamp'], reverse=True)
        return models[:limit]

    def _set_production_model(self, model_path: Path):
        """Set the production model pointer."""
        prod_pointer = self.models_dir / "production_model.txt"
        with open(prod_pointer, 'w') as f:
            f.write(str(model_path))
        logger.info(f"Set production model to {model_path}")

    def deploy_model(
            self,
            model_name: str
    ) -> Path:
        """
        Deploy a model to production.

        Args:
            model_name: Name of model to deploy

        Returns:
            Path to deployed model
        """
        model_path = self.models_dir / f"{model_name}.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self._set_production_model(model_path)
        logger.info(f"Deployed {model_name} to production")

        return model_path

    def rollback(self, steps: int = 1) -> Tuple[BaselineModel, Dict[str, Any]]:
        """
        Rollback to a previous production model.

        Args:
            steps: Number of versions to roll back

        Returns:
            Tuple[model, metadata] of the restored model
        """
        production_models = self.list_models(tag="production")

        if len(production_models) <= steps:
            raise ValueError(f"Not enough production models to rollback {steps} steps")

        # Get the model to restore (skip 'steps' most recent)
        target_model = production_models[steps]

        # Deploy it as current production
        self._set_production_model(Path(target_model['path']))
        logger.info(f"Rolled back to {target_model['name']}")

        return self.load_model(model_path=Path(target_model['path']))

    def cleanup_old_models(
            self,
            keep_production: int = 5,
            keep_checkpoints: int = 10
    ) -> int:
        """
        Remove old model files.

        Args:
            keep_production: Number of production models to keep
            keep_checkpoints: Number of checkpoint models to keep

        Returns:
            Number of models deleted
        """
        deleted = 0

        for tag, keep_count in [("production", keep_production), ("checkpoint", keep_checkpoints)]:
            models = self.list_models(tag=tag)

            # Delete models beyond keep count
            for model in models[keep_count:]:
                model_path = Path(model['path'])
                metadata_path = model_path.parent / (model_path.stem + self.metadata_suffix)

                try:
                    model_path.unlink()
                    if metadata_path.exists():
                        metadata_path.unlink()
                    deleted += 1
                    logger.info(f"Deleted old model: {model['name']}")
                except Exception as e:
                    logger.error(f"Failed to delete {model['name']}: {e}")

        return deleted

    def get_model_summary(self) -> str:
        """Generate a summary of saved models."""
        lines = [
            "=" * 60,
            "MODEL INVENTORY",
            "=" * 60,
            "",
        ]

        # Current production model
        try:
            _, prod_meta = self.load_production()
            lines.append(f"Current Production: {prod_meta.get('model_name', 'Unknown')}")
            if 'metrics' in prod_meta:
                metrics = prod_meta['metrics']
                lines.append(f"  Sharpe: {metrics.get('sharpe_ratio', 0):.3f}, "
                             f"WinRate: {metrics.get('win_rate', 0):.2%}")
            lines.append("")
        except FileNotFoundError:
            lines.append("Current Production: None")
            lines.append("")

        # List models by tag
        for tag in ["production", "checkpoint"]:
            models = self.list_models(tag=tag, limit=5)
            lines.append(f"{tag.upper()} MODELS ({len(models)} shown):")
            for m in models:
                metrics = m.get('metrics', {})
                lines.append(f"  - {m['name']}")
                lines.append(f"    Created: {m['created_at']}, Size: {m['size_mb']:.2f} MB")
                if metrics:
                    lines.append(f"    Sharpe: {metrics.get('sharpe_ratio', 0):.3f}, "
                                 f"Accuracy: {metrics.get('accuracy', 0):.3f}")
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
    from features.feature_engineering import FeatureEngineer
    from models.model_trainer import ModelTrainer

    config = get_config()
    fetcher = DataFetcher(config)
    engineer = FeatureEngineer(config)
    trainer = ModelTrainer(config)
    manager = ModelManager(config)

    # Prepare and train a model
    from datetime import datetime
    start = datetime(2023, 1, 1)
    end = datetime(2024, 6, 1)
    df = fetcher.fetch_historical(start, end)
    df_features = engineer.build_features(df)
    X, y, feature_names = engineer.prepare_training_data(df_features)

    model, metrics = trainer.train_simple(X, y, feature_names)

    # Save model
    print("\nSaving model...")
    model_path = manager.save_model(
        model,
        metrics=metrics,
        tag="production",
        feature_names=feature_names
    )
    print(f"Saved to: {model_path}")

    # List models
    print("\nListing models:")
    for m in manager.list_models():
        print(f"  - {m['name']} ({m['tag']})")

    # Load production model
    print("\nLoading production model...")
    loaded_model, loaded_meta = manager.load_production()
    print(f"Loaded: {loaded_meta.get('model_name')}")

    # Print summary
    print("\n" + manager.get_model_summary())
