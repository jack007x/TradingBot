"""
Ensemble Strategy combining multiple AI models.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from loguru import logger


class EnsembleStrategy:
    """
    Ensemble strategy that combines predictions from multiple AI models
    (LSTM, GRU, CNN, RL agents, NLP sentiment) for trading decisions.
    """

    def __init__(
        self,
        models: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None,
        voting_method: str = 'weighted',
        confidence_threshold: float = 0.6
    ):
        """
        Initialize ensemble strategy.

        Args:
            models: Dictionary of model instances {name: model}
            weights: Initial weights for each model
            voting_method: 'weighted', 'majority', or 'unanimous'
            confidence_threshold: Minimum confidence for action
        """
        self.models = models
        self.voting_method = voting_method
        self.confidence_threshold = confidence_threshold

        # Initialize equal weights if not provided
        if weights:
            self.weights = weights
        else:
            self.weights = {name: 1.0 / len(models) for name in models}

        # Prediction history for performance tracking
        self.prediction_history: List[Dict] = []

        logger.info(f"EnsembleStrategy initialized with {len(models)} models")

    def get_predictions(
        self,
        features: np.ndarray,
        sentiment_data: Optional[Dict] = None,
        current_price: float = 0.0
    ) -> Dict[str, Dict]:
        """
        Get predictions from all models.

        Args:
            features: Feature array for prediction
            sentiment_data: NLP sentiment data
            current_price: Current market price

        Returns:
            Dictionary of predictions from each model
        """
        predictions = {}

        for name, model in self.models.items():
            try:
                if 'lstm' in name.lower() or 'gru' in name.lower():
                    pred = self._get_nn_prediction(model, features, current_price)
                elif 'cnn' in name.lower():
                    pred = self._get_cnn_prediction(model, features)
                elif 'ppo' in name.lower() or 'dql' in name.lower():
                    pred = self._get_rl_prediction(model, features)
                elif 'sentiment' in name.lower() or 'nlp' in name.lower():
                    pred = self._get_sentiment_prediction(model, sentiment_data)
                else:
                    pred = {'signal': 'hold', 'confidence': 0.5}

                predictions[name] = pred

            except Exception as e:
                logger.error(f"Error getting prediction from {name}: {e}")
                predictions[name] = {'signal': 'hold', 'confidence': 0.0, 'error': str(e)}

        return predictions

    def _get_nn_prediction(
        self,
        model,
        features: np.ndarray,
        current_price: float
    ) -> Dict:
        """
        Get prediction from LSTM/GRU regression model.

        UPDATED FOR REGRESSION APPROACH:
        - Model now predicts RETURNS (e.g., +0.0025 = +0.25% gain)
        - Not absolute prices!
        - Larger return magnitude = higher confidence
        """
        try:
            prediction = model.predict(features)

            # Extract predicted return (continuous value like +0.0025 or -0.0015)
            predicted_return = float(prediction[0]) if hasattr(prediction[0], '__iter__') else float(prediction)

            # Confidence based on return magnitude (larger magnitude = more confident)
            # Scale: 0.5% return → 0.7 confidence, 1% → 0.8, 2% → 0.9
            return_magnitude = abs(predicted_return)
            confidence = min(0.5 + (return_magnitude / 0.02) * 0.4, 0.95)  # Cap at 0.95

            # Determine signal based on predicted return
            # Use adaptive threshold based on magnitude
            threshold = 0.003  # 0.3% default threshold

            if predicted_return > threshold:
                signal = 'buy'
            elif predicted_return < -threshold:
                signal = 'sell'
            else:
                signal = 'hold'

            # Calculate predicted price for compatibility
            predicted_price = current_price * (1 + predicted_return) if current_price > 0 else 0

            return {
                'signal': signal,
                'confidence': confidence,
                'predicted_return': predicted_return,  # New: return prediction
                'predicted_price': predicted_price,    # Legacy: for backward compatibility
                'price_change': predicted_return       # Now same as predicted_return
            }
        except Exception as e:
            logger.error(f"NN prediction error: {e}")
            return {'signal': 'hold', 'confidence': 0.0}

    def _get_cnn_prediction(self, model, features: np.ndarray) -> Dict:
        """Get prediction from CNN model."""
        try:
            predictions = model.predict(features)

            if len(predictions.shape) > 1:
                # Classification output
                class_idx = np.argmax(predictions[0])
                confidence = float(predictions[0][class_idx])

                if class_idx == 1:  # Bullish
                    signal = 'buy'
                elif class_idx == 0:  # Bearish
                    signal = 'sell'
                else:
                    signal = 'hold'
            else:
                signal = 'hold'
                confidence = 0.5

            return {
                'signal': signal,
                'confidence': confidence,
                'probabilities': predictions[0].tolist() if len(predictions.shape) > 1 else []
            }
        except Exception as e:
            logger.error(f"CNN prediction error: {e}")
            return {'signal': 'hold', 'confidence': 0.0}

    def _get_rl_prediction(self, model, features: np.ndarray) -> Dict:
        """Get prediction from RL agent."""
        try:
            action, confidence = model.predict(features, deterministic=True)

            # Map action to signal
            action_map = {0: 'hold', 1: 'buy', 2: 'sell'}
            signal = action_map.get(action, 'hold')

            return {
                'signal': signal,
                'confidence': confidence,
                'action': int(action)
            }
        except Exception as e:
            logger.error(f"RL prediction error: {e}")
            return {'signal': 'hold', 'confidence': 0.0}

    def _get_sentiment_prediction(
        self,
        model,
        sentiment_data: Optional[Dict]
    ) -> Dict:
        """Get prediction from sentiment analyzer."""
        if sentiment_data is None:
            return {'signal': 'hold', 'confidence': 0.0}

        try:
            sentiment = sentiment_data.get('sentiment', 'neutral')
            score = sentiment_data.get('score', 0.0)
            confidence = sentiment_data.get('confidence', 0.5)

            if sentiment == 'bullish' or score > 0.2:
                signal = 'buy'
            elif sentiment == 'bearish' or score < -0.2:
                signal = 'sell'
            else:
                signal = 'hold'

            return {
                'signal': signal,
                'confidence': confidence,
                'sentiment': sentiment,
                'score': score
            }
        except Exception as e:
            logger.error(f"Sentiment prediction error: {e}")
            return {'signal': 'hold', 'confidence': 0.0}

    def combine_predictions(
        self,
        predictions: Dict[str, Dict]
    ) -> Dict:
        """
        Combine predictions using the configured voting method.

        Args:
            predictions: Dictionary of model predictions

        Returns:
            Combined prediction
        """
        if self.voting_method == 'weighted':
            return self._weighted_voting(predictions)
        elif self.voting_method == 'majority':
            return self._majority_voting(predictions)
        elif self.voting_method == 'unanimous':
            return self._unanimous_voting(predictions)
        else:
            return self._weighted_voting(predictions)

    def _weighted_voting(self, predictions: Dict[str, Dict]) -> Dict:
        """Weighted voting combination."""
        scores = {'buy': 0.0, 'sell': 0.0, 'hold': 0.0}

        for name, pred in predictions.items():
            weight = self.weights.get(name, 0.0)
            signal = pred.get('signal', 'hold')
            confidence = pred.get('confidence', 0.5)

            scores[signal] += weight * confidence

        # Normalize
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}

        best_signal = max(scores, key=scores.get)
        best_confidence = scores[best_signal]

        return {
            'signal': best_signal,
            'confidence': best_confidence,
            'scores': scores,
            'method': 'weighted'
        }

    def _majority_voting(self, predictions: Dict[str, Dict]) -> Dict:
        """Simple majority voting."""
        votes = {'buy': 0, 'sell': 0, 'hold': 0}

        for pred in predictions.values():
            signal = pred.get('signal', 'hold')
            votes[signal] += 1

        best_signal = max(votes, key=votes.get)
        best_count = votes[best_signal]
        total = len(predictions)

        return {
            'signal': best_signal,
            'confidence': best_count / total if total > 0 else 0,
            'votes': votes,
            'method': 'majority'
        }

    def _unanimous_voting(self, predictions: Dict[str, Dict]) -> Dict:
        """Unanimous voting - all models must agree."""
        signals = [p.get('signal', 'hold') for p in predictions.values()]

        if len(set(signals)) == 1:
            signal = signals[0]
            avg_confidence = np.mean([p.get('confidence', 0.5) for p in predictions.values()])
            return {
                'signal': signal,
                'confidence': avg_confidence,
                'unanimous': True,
                'method': 'unanimous'
            }

        return {
            'signal': 'hold',
            'confidence': 0.0,
            'unanimous': False,
            'method': 'unanimous',
            'reason': 'Models disagreed'
        }

    def get_trading_signal(
        self,
        features: np.ndarray,
        sentiment_data: Optional[Dict] = None,
        current_price: float = 0.0
    ) -> Dict:
        """
        Get final trading signal.

        Args:
            features: Feature array
            sentiment_data: Sentiment analysis data
            current_price: Current price

        Returns:
            Trading signal with confidence and details
        """
        # Get individual predictions
        predictions = self.get_predictions(features, sentiment_data, current_price)

        # Combine predictions
        combined = self.combine_predictions(predictions)

        # Apply confidence threshold
        if combined['confidence'] < self.confidence_threshold:
            combined['signal'] = 'hold'
            combined['reason'] = f"Below confidence threshold ({self.confidence_threshold})"

        # Add timestamp
        combined['timestamp'] = datetime.utcnow().isoformat()
        combined['individual_predictions'] = predictions

        # Store in history
        self.prediction_history.append(combined)

        return combined

    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """Update model weights."""
        self.weights.update(new_weights)
        logger.info(f"Weights updated: {self.weights}")

    def get_model_performance(self) -> Dict[str, Dict]:
        """Analyze performance of individual models."""
        if len(self.prediction_history) < 10:
            return {}

        performance = {}
        # This would need actual outcome data to calculate properly
        # For now, return weight distribution
        for name in self.models:
            performance[name] = {
                'weight': self.weights.get(name, 0),
                'predictions_made': len(self.prediction_history)
            }

        return performance
