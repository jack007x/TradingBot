"""
Model compatibility checker
Prevents loading mismatched architectures and provides diagnostic information
"""

import torch
from pathlib import Path
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ModelCompatibilityChecker:
    """Check if saved model is compatible with current code"""

    @staticmethod
    def check_compatibility(checkpoint_path: str) -> Dict:
        """
        Check model compatibility before loading.

        Validates:
        - Architecture consistency (layers in config vs weights)
        - Version compatibility
        - Required configuration fields
        - State dict integrity

        Args:
            checkpoint_path: Path to model checkpoint file

        Returns:
            dict: {
                'compatible': bool,
                'issues': list[str],
                'warnings': list[str],
                'config': dict,
                'detected_layers': int,
                'version': str
            }
        """
        result = {
            'compatible': True,
            'issues': [],
            'warnings': [],
            'config': {},
            'detected_layers': None,
            'version': None
        }

        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

            # Extract config
            config = checkpoint.get('config', {})
            result['config'] = config

            # Extract version
            version = checkpoint.get('version', '1.0')
            result['version'] = version

            # Detect layers from state_dict
            state_dict = checkpoint.get('model_state', checkpoint.get('model_state_dict', {}))
            detected_layers = ModelCompatibilityChecker._detect_layers(state_dict)
            result['detected_layers'] = detected_layers

            # Check version
            if version < '2.0':
                result['warnings'].append(
                    f"Old model version: {version} (current: 2.1). "
                    "Recommend retraining for TradingLoss and better performance."
                )

            # Check layer consistency
            config_layers = config.get('num_layers')
            if config_layers and config_layers != detected_layers:
                result['issues'].append(
                    f"❌ ARCHITECTURE MISMATCH: "
                    f"Config says {config_layers} layers, "
                    f"but weights have {detected_layers} layers. "
                    f"This will cause loading errors!"
                )
                result['compatible'] = False
            elif detected_layers:
                result['warnings'].append(
                    f"✅ Architecture consistent: {detected_layers} layers"
                )

            # Check required fields
            if version >= '2.0':
                required_fields = ['input_size', 'hidden_size', 'model_type', 'num_layers']
                missing = [f for f in required_fields if f not in config]
                if missing:
                    result['issues'].append(
                        f"Missing required config fields: {missing}"
                    )
                    result['compatible'] = False

            # Check state dict integrity
            if not state_dict:
                result['issues'].append("No model state_dict found in checkpoint!")
                result['compatible'] = False

        except Exception as e:
            result['compatible'] = False
            result['issues'].append(f"Failed to load checkpoint: {e}")

        return result

    @staticmethod
    def _detect_layers(state_dict: dict) -> Optional[int]:
        """
        Detect number of layers from state dict.

        Args:
            state_dict: Model state dictionary

        Returns:
            Number of layers detected, or None if cannot detect
        """
        max_layer = -1
        for key in state_dict.keys():
            if '_l' in key:
                parts = key.split('_l')
                if len(parts) > 1:
                    layer_str = parts[1].split('_')[0].split('.')[0]
                    if layer_str.isdigit():
                        max_layer = max(max_layer, int(layer_str))

        return max_layer + 1 if max_layer >= 0 else None

    @staticmethod
    def print_report(checkpoint_path: str) -> bool:
        """
        Print detailed compatibility report.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            True if compatible, False otherwise
        """
        print("=" * 80)
        print("MODEL COMPATIBILITY REPORT")
        print("=" * 80)
        print(f"File: {checkpoint_path}")
        print("")

        result = ModelCompatibilityChecker.check_compatibility(checkpoint_path)

        # Version info
        print(f"Version: {result['version']}")
        if result['detected_layers']:
            print(f"Detected Layers: {result['detected_layers']}")
        print("")

        # Config info
        if result['config']:
            print("Configuration:")
            for key, value in result['config'].items():
                if key != 'device':  # Skip device as it's not important
                    print(f"  {key}: {value}")
            print("")

        # Compatibility status
        if result['compatible']:
            print("✅ STATUS: COMPATIBLE")
        else:
            print("❌ STATUS: INCOMPATIBLE")
        print("")

        # Issues
        if result['issues']:
            print("🔴 CRITICAL ISSUES:")
            for issue in result['issues']:
                print(f"  - {issue}")
            print("")

        # Warnings
        if result['warnings']:
            print("⚠️  WARNINGS:")
            for warning in result['warnings']:
                print(f"  - {warning}")
            print("")

        print("=" * 80)

        return result['compatible']

    @staticmethod
    def check_all_models(models_dir: str = 'saved_models') -> Dict[str, Dict]:
        """
        Check compatibility of all models in directory.

        Args:
            models_dir: Directory containing model checkpoints

        Returns:
            Dict mapping model name to compatibility result
        """
        models_dir = Path(models_dir)
        results = {}

        print("=" * 80)
        print(f"CHECKING ALL MODELS IN: {models_dir}")
        print("=" * 80)
        print("")

        for model_file in models_dir.glob("*.pt"):
            if model_file.name == 'dql_agent.pt':
                continue  # Skip DQL for now (different format)

            model_name = model_file.stem
            print(f"Checking {model_name}...")
            result = ModelCompatibilityChecker.check_compatibility(str(model_file))
            results[model_name] = result

            if result['compatible']:
                print(f"  ✅ Compatible (v{result['version']}, {result['detected_layers']} layers)")
            else:
                print(f"  ❌ Incompatible - {len(result['issues'])} issue(s)")
            print("")

        # Summary
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        compatible = sum(1 for r in results.values() if r['compatible'])
        total = len(results)
        print(f"Compatible: {compatible}/{total} models")

        if compatible < total:
            print("\n⚠️  Some models have compatibility issues!")
            print("Recommend retraining or fixing architecture mismatches.")

        print("=" * 80)

        return results


def check_before_loading(model_path: str) -> bool:
    """
    Quick compatibility check before loading model.

    Args:
        model_path: Path to model checkpoint

    Returns:
        True if compatible, False otherwise
    """
    checker = ModelCompatibilityChecker()
    result = checker.check_compatibility(model_path)

    logger.info("=" * 70)
    logger.info("MODEL COMPATIBILITY CHECK")
    logger.info("=" * 70)
    logger.info(f"File: {model_path}")
    logger.info(f"Version: {result['version']}")
    logger.info(f"Compatible: {result['compatible']}")

    if result['issues']:
        logger.warning("Issues found:")
        for issue in result['issues']:
            logger.warning(f"  - {issue}")

    if result['warnings']:
        logger.info("Warnings:")
        for warning in result['warnings']:
            logger.info(f"  - {warning}")

    if result['detected_layers']:
        logger.info(f"Detected layers: {result['detected_layers']}")

    logger.info("=" * 70)

    return result['compatible']


if __name__ == "__main__":
    # CLI interface for quick checks
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m src.utils.model_compatibility <checkpoint_path>")
        print("  python -m src.utils.model_compatibility --check-all [models_dir]")
        sys.exit(1)

    if sys.argv[1] == '--check-all':
        models_dir = sys.argv[2] if len(sys.argv) > 2 else 'saved_models'
        ModelCompatibilityChecker.check_all_models(models_dir)
    else:
        checkpoint_path = sys.argv[1]
        is_compatible = ModelCompatibilityChecker.print_report(checkpoint_path)
        sys.exit(0 if is_compatible else 1)
