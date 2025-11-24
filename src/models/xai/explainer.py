"""
Explainable AI (XAI) module for model transparency and interpretability.
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import json
from datetime import datetime
from loguru import logger

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False


class ModelExplainer:
    """
    Provides explainability for trading model decisions using SHAP and LIME.
    """

    def __init__(
        self,
        feature_names: List[str],
        model_type: str = 'neural_network',
        use_shap: bool = True,
        use_lime: bool = True
    ):
        """
        Initialize model explainer.

        Args:
            feature_names: Names of input features
            model_type: Type of model ('neural_network', 'tree', 'linear')
            use_shap: Whether to use SHAP explanations
            use_lime: Whether to use LIME explanations
        """
        self.feature_names = feature_names
        self.model_type = model_type
        self.use_shap = use_shap and SHAP_AVAILABLE
        self.use_lime = use_lime and LIME_AVAILABLE

        self.shap_explainer = None
        self.lime_explainer = None
        self.explanation_history: List[Dict] = []

        logger.info(f"ModelExplainer initialized (SHAP={self.use_shap}, LIME={self.use_lime})")

    def setup_shap(
        self,
        model_predict: Callable,
        background_data: np.ndarray,
        algorithm: str = 'auto'
    ) -> None:
        """
        Setup SHAP explainer.

        Args:
            model_predict: Model prediction function
            background_data: Background dataset for SHAP
            algorithm: SHAP algorithm ('auto', 'deep', 'kernel', 'tree')
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available")
            return

        try:
            if algorithm == 'deep' or self.model_type == 'neural_network':
                self.shap_explainer = shap.KernelExplainer(
                    model_predict,
                    shap.sample(background_data, min(100, len(background_data)))
                )
            elif algorithm == 'tree' or self.model_type == 'tree':
                self.shap_explainer = shap.TreeExplainer(model_predict)
            else:
                self.shap_explainer = shap.KernelExplainer(
                    model_predict,
                    shap.sample(background_data, min(100, len(background_data)))
                )

            logger.info("SHAP explainer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize SHAP: {e}")
            self.use_shap = False

    def setup_lime(
        self,
        training_data: np.ndarray,
        mode: str = 'regression',
        categorical_features: Optional[List[int]] = None
    ) -> None:
        """
        Setup LIME explainer.

        Args:
            training_data: Training data for LIME
            mode: 'regression' or 'classification'
            categorical_features: Indices of categorical features
        """
        if not LIME_AVAILABLE:
            logger.warning("LIME not available")
            return

        try:
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data,
                feature_names=self.feature_names,
                mode=mode,
                categorical_features=categorical_features
            )
            logger.info("LIME explainer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize LIME: {e}")
            self.use_lime = False

    def explain_prediction(
        self,
        model_predict: Callable,
        instance: np.ndarray,
        num_features: int = 10
    ) -> Dict[str, Any]:
        """
        Explain a single prediction.

        Args:
            model_predict: Model prediction function
            instance: Input instance to explain
            num_features: Number of top features to show

        Returns:
            Explanation dictionary
        """
        explanation = {
            'timestamp': datetime.utcnow().isoformat(),
            'prediction': float(model_predict(instance.reshape(1, -1))[0]),
            'explanations': {}
        }

        # SHAP explanation
        if self.use_shap and self.shap_explainer is not None:
            try:
                shap_values = self.shap_explainer.shap_values(instance.reshape(1, -1))
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]

                shap_values = shap_values.flatten()
                feature_importance = dict(zip(self.feature_names, shap_values.tolist()))

                # Sort by absolute importance
                sorted_features = sorted(
                    feature_importance.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )[:num_features]

                explanation['explanations']['shap'] = {
                    'feature_importance': dict(sorted_features),
                    'base_value': float(self.shap_explainer.expected_value) if hasattr(self.shap_explainer, 'expected_value') else 0.0
                }
            except Exception as e:
                logger.error(f"SHAP explanation error: {e}")

        # LIME explanation
        if self.use_lime and self.lime_explainer is not None:
            try:
                lime_exp = self.lime_explainer.explain_instance(
                    instance,
                    model_predict,
                    num_features=num_features
                )

                lime_importance = dict(lime_exp.as_list())
                explanation['explanations']['lime'] = {
                    'feature_importance': lime_importance,
                    'intercept': float(lime_exp.intercept[0]) if hasattr(lime_exp, 'intercept') else 0.0
                }
            except Exception as e:
                logger.error(f"LIME explanation error: {e}")

        # Store in history
        self.explanation_history.append(explanation)

        return explanation

    def get_feature_importance(
        self,
        model_predict: Callable,
        data: np.ndarray,
        num_samples: int = 100
    ) -> Dict[str, float]:
        """
        Calculate global feature importance.

        Args:
            model_predict: Model prediction function
            data: Dataset for importance calculation
            num_samples: Number of samples to use

        Returns:
            Feature importance dictionary
        """
        if not self.use_shap or self.shap_explainer is None:
            # Fallback: permutation importance
            return self._permutation_importance(model_predict, data, num_samples)

        try:
            sample_data = data[:num_samples] if len(data) > num_samples else data
            shap_values = self.shap_explainer.shap_values(sample_data)

            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            # Mean absolute SHAP values
            importance = np.abs(shap_values).mean(axis=0)
            importance_dict = dict(zip(self.feature_names, importance.tolist()))

            # Normalize
            total = sum(importance_dict.values())
            if total > 0:
                importance_dict = {k: v / total for k, v in importance_dict.items()}

            return importance_dict

        except Exception as e:
            logger.error(f"Feature importance error: {e}")
            return self._permutation_importance(model_predict, data, num_samples)

    def _permutation_importance(
        self,
        model_predict: Callable,
        data: np.ndarray,
        num_samples: int = 100
    ) -> Dict[str, float]:
        """Calculate permutation-based feature importance."""
        sample_data = data[:num_samples] if len(data) > num_samples else data
        baseline_pred = model_predict(sample_data)

        importance = {}
        for i, feature in enumerate(self.feature_names):
            permuted_data = sample_data.copy()
            np.random.shuffle(permuted_data[:, i])
            permuted_pred = model_predict(permuted_data)

            # Importance is the change in prediction
            imp = np.mean(np.abs(baseline_pred - permuted_pred))
            importance[feature] = float(imp)

        # Normalize
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}

        return importance

    def explain_trading_decision(
        self,
        model_predict: Callable,
        instance: np.ndarray,
        decision: str,
        confidence: float
    ) -> Dict[str, Any]:
        """
        Generate human-readable explanation for trading decision.

        Args:
            model_predict: Model prediction function
            instance: Input features
            decision: Trading decision ('buy', 'sell', 'hold')
            confidence: Confidence level

        Returns:
            Human-readable explanation
        """
        # Get feature explanation
        explanation = self.explain_prediction(model_predict, instance)

        # Build narrative
        top_features = []
        if 'shap' in explanation.get('explanations', {}):
            top_features = list(explanation['explanations']['shap']['feature_importance'].items())[:5]
        elif 'lime' in explanation.get('explanations', {}):
            top_features = list(explanation['explanations']['lime']['feature_importance'].items())[:5]

        # Generate narrative
        narrative_parts = []
        for feature, importance in top_features:
            direction = "positively" if importance > 0 else "negatively"
            narrative_parts.append(f"{feature} ({direction} influencing)")

        narrative = f"The model decided to {decision.upper()} with {confidence:.1%} confidence. "
        narrative += "Key factors: " + ", ".join(narrative_parts) if narrative_parts else "Unable to determine key factors."

        return {
            'decision': decision,
            'confidence': confidence,
            'narrative': narrative,
            'top_factors': dict(top_features),
            'full_explanation': explanation
        }

    def generate_report(
        self,
        model_name: str,
        period: str = 'daily'
    ) -> Dict[str, Any]:
        """
        Generate XAI report for a period.

        Args:
            model_name: Name of the model
            period: Report period

        Returns:
            XAI report dictionary
        """
        if not self.explanation_history:
            return {'error': 'No explanations available'}

        # Aggregate feature importance
        all_importance = {}
        for exp in self.explanation_history:
            for method in ['shap', 'lime']:
                if method in exp.get('explanations', {}):
                    for feature, imp in exp['explanations'][method]['feature_importance'].items():
                        if feature not in all_importance:
                            all_importance[feature] = []
                        all_importance[feature].append(abs(imp))

        avg_importance = {
            feature: np.mean(values)
            for feature, values in all_importance.items()
        }

        # Normalize
        total = sum(avg_importance.values())
        if total > 0:
            avg_importance = {k: v / total for k, v in avg_importance.items()}

        # Sort by importance
        sorted_importance = dict(
            sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
        )

        return {
            'model_name': model_name,
            'period': period,
            'generated_at': datetime.utcnow().isoformat(),
            'total_explanations': len(self.explanation_history),
            'average_feature_importance': sorted_importance,
            'top_5_features': dict(list(sorted_importance.items())[:5]),
            'explanation_methods_used': {
                'shap': self.use_shap,
                'lime': self.use_lime
            }
        }

    def save_report(self, filepath: Union[str, Path], model_name: str) -> None:
        """Save XAI report to file."""
        report = self.generate_report(model_name)
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"XAI report saved to {filepath}")

    def clear_history(self) -> None:
        """Clear explanation history."""
        self.explanation_history.clear()
        logger.info("Explanation history cleared")
