"""
Experience Buffer for Continuous Learning.
Stores trade outcomes for model fine-tuning.
"""

import numpy as np
import json
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from pathlib import Path
from loguru import logger


@dataclass
class TradeExperience:
    """
    Single trade experience record.

    Attributes:
        prediction_id: Unique ID linking prediction to outcome
        timestamp: When prediction was made
        symbol: Trading symbol
        state: Market state (features) at prediction time
        predicted_return: Model's predicted return
        predicted_signal: 'buy', 'sell', or 'hold'
        confidence: Model confidence (0-1)
        model_name: Which model made prediction

        # Trade execution details
        trade_executed: Whether trade was actually executed
        trade_ticket: MT5 ticket number (if executed)
        entry_price: Execution price (if executed)
        exit_price: Close price (if closed)
        exit_timestamp: When position closed

        # Outcome
        actual_return: Realized return (if trade closed)
        profit_loss: P&L in dollars (if trade closed)
        outcome_recorded: Whether outcome is finalized

        # For learning
        directional_correct: Whether direction was correct
        prediction_error: abs(predicted - actual) return
    """
    prediction_id: str
    timestamp: datetime
    symbol: str
    state: np.ndarray
    predicted_return: float
    predicted_signal: str
    confidence: float
    model_name: str

    # Trade execution
    trade_executed: bool = False
    trade_ticket: Optional[int] = None
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None

    # Outcome
    actual_return: Optional[float] = None
    profit_loss: Optional[float] = None
    outcome_recorded: bool = False

    # Metrics
    directional_correct: Optional[bool] = None
    prediction_error: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'prediction_id': self.prediction_id,
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'state': self.state.tolist() if isinstance(self.state, np.ndarray) else self.state,
            'predicted_return': float(self.predicted_return),
            'predicted_signal': self.predicted_signal,
            'confidence': float(self.confidence),
            'model_name': self.model_name,
            'trade_executed': self.trade_executed,
            'trade_ticket': self.trade_ticket,
            'entry_price': float(self.entry_price) if self.entry_price else None,
            'exit_price': float(self.exit_price) if self.exit_price else None,
            'exit_timestamp': self.exit_timestamp.isoformat() if self.exit_timestamp else None,
            'actual_return': float(self.actual_return) if self.actual_return else None,
            'profit_loss': float(self.profit_loss) if self.profit_loss else None,
            'outcome_recorded': self.outcome_recorded,
            'directional_correct': self.directional_correct,
            'prediction_error': float(self.prediction_error) if self.prediction_error else None
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'TradeExperience':
        """Create from dictionary."""
        return cls(
            prediction_id=data['prediction_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            symbol=data['symbol'],
            state=np.array(data['state']),
            predicted_return=data['predicted_return'],
            predicted_signal=data['predicted_signal'],
            confidence=data['confidence'],
            model_name=data['model_name'],
            trade_executed=data['trade_executed'],
            trade_ticket=data.get('trade_ticket'),
            entry_price=data.get('entry_price'),
            exit_price=data.get('exit_price'),
            exit_timestamp=datetime.fromisoformat(data['exit_timestamp']) if data.get('exit_timestamp') else None,
            actual_return=data.get('actual_return'),
            profit_loss=data.get('profit_loss'),
            outcome_recorded=data['outcome_recorded'],
            directional_correct=data.get('directional_correct'),
            prediction_error=data.get('prediction_error')
        )


class ExperienceBuffer:
    """
    Buffer for storing trade experiences for continuous learning.

    Features:
    - Stores predictions and outcomes
    - Links predictions to trade results
    - Provides samples for model fine-tuning
    - Tracks model performance over time
    - Persists to disk for recovery
    """

    def __init__(
        self,
        max_size: int = 10000,
        buffer_path: str = 'data_cache/experience_buffer.json'
    ):
        """
        Initialize experience buffer.

        Args:
            max_size: Maximum number of experiences to keep
            buffer_path: Path to save buffer to disk
        """
        self.max_size = max_size
        self.buffer_path = Path(buffer_path)

        # Main storage: completed experiences with outcomes
        self.completed_experiences: deque = deque(maxlen=max_size)

        # Pending predictions waiting for outcomes
        self.pending_predictions: Dict[str, TradeExperience] = {}

        # Mapping: ticket -> prediction_id for linking trades to predictions
        self.ticket_to_prediction: Dict[int, str] = {}

        # Performance tracking per model
        self.model_performance: Dict[str, Dict] = {}

        logger.info(f"ExperienceBuffer initialized (max_size={max_size})")
        logger.info(f"  Save path: {self.buffer_path}")

    def add_prediction(
        self,
        prediction_id: str,
        symbol: str,
        state: np.ndarray,
        predicted_return: float,
        predicted_signal: str,
        confidence: float,
        model_name: str
    ) -> TradeExperience:
        """
        Record a prediction made by a model.

        Args:
            prediction_id: Unique ID for this prediction
            symbol: Trading symbol
            state: Market state (features)
            predicted_return: Model's predicted return
            predicted_signal: 'buy', 'sell', or 'hold'
            confidence: Model confidence
            model_name: Name of model

        Returns:
            TradeExperience object
        """
        experience = TradeExperience(
            prediction_id=prediction_id,
            timestamp=datetime.utcnow(),
            symbol=symbol,
            state=state,
            predicted_return=predicted_return,
            predicted_signal=predicted_signal,
            confidence=confidence,
            model_name=model_name
        )

        self.pending_predictions[prediction_id] = experience

        logger.debug(f"📝 Prediction recorded: {prediction_id}")
        logger.debug(f"   Model: {model_name} | Signal: {predicted_signal} | Confidence: {confidence:.2f}")

        return experience

    def link_trade_to_prediction(
        self,
        prediction_id: str,
        trade_ticket: int,
        entry_price: float
    ):
        """
        Link an executed trade to its prediction.

        Args:
            prediction_id: ID of the prediction
            trade_ticket: MT5 ticket number
            entry_price: Execution price
        """
        if prediction_id not in self.pending_predictions:
            logger.warning(f"⚠️  Prediction {prediction_id} not found in pending predictions")
            return

        experience = self.pending_predictions[prediction_id]
        experience.trade_executed = True
        experience.trade_ticket = trade_ticket
        experience.entry_price = entry_price

        # Map ticket to prediction for outcome recording
        self.ticket_to_prediction[trade_ticket] = prediction_id

        logger.info(f"🔗 Trade linked to prediction:")
        logger.info(f"   Prediction: {prediction_id}")
        logger.info(f"   Ticket: {trade_ticket}")
        logger.info(f"   Entry: {entry_price:.5f}")

    def record_outcome(
        self,
        trade_ticket: int,
        exit_price: float,
        profit_loss: float
    ):
        """
        Record the outcome of a closed trade.

        Args:
            trade_ticket: MT5 ticket number
            exit_price: Close price
            profit_loss: P&L in dollars
        """
        # Find prediction linked to this ticket
        if trade_ticket not in self.ticket_to_prediction:
            logger.warning(f"⚠️  No prediction found for ticket {trade_ticket}")
            return

        prediction_id = self.ticket_to_prediction[trade_ticket]

        if prediction_id not in self.pending_predictions:
            logger.warning(f"⚠️  Prediction {prediction_id} not in pending list")
            return

        experience = self.pending_predictions[prediction_id]

        # Calculate actual return
        if experience.entry_price:
            actual_return = (exit_price - experience.entry_price) / experience.entry_price

            # Adjust for short positions
            if experience.predicted_signal == 'sell':
                actual_return = -actual_return
        else:
            actual_return = 0.0

        # Update experience
        experience.exit_price = exit_price
        experience.exit_timestamp = datetime.utcnow()
        experience.actual_return = actual_return
        experience.profit_loss = profit_loss
        experience.outcome_recorded = True

        # Calculate metrics
        predicted_direction = 1 if experience.predicted_return > 0 else -1
        actual_direction = 1 if actual_return > 0 else -1
        experience.directional_correct = (predicted_direction == actual_direction)
        experience.prediction_error = abs(experience.predicted_return - actual_return)

        # Move to completed experiences
        self.completed_experiences.append(experience)
        del self.pending_predictions[prediction_id]
        del self.ticket_to_prediction[trade_ticket]

        # Update model performance
        self._update_model_performance(experience)

        logger.info(f"✅ Outcome recorded for ticket {trade_ticket}")
        logger.info(f"   Predicted: {experience.predicted_return:.4f} | Actual: {actual_return:.4f}")
        logger.info(f"   Direction correct: {experience.directional_correct}")
        logger.info(f"   P&L: ${profit_loss:.2f}")

    def _update_model_performance(self, experience: TradeExperience):
        """Update performance metrics for a model."""
        model = experience.model_name

        if model not in self.model_performance:
            self.model_performance[model] = {
                'total_predictions': 0,
                'correct_directions': 0,
                'total_error': 0.0,
                'total_profit': 0.0,
                'win_count': 0
            }

        perf = self.model_performance[model]
        perf['total_predictions'] += 1

        if experience.directional_correct:
            perf['correct_directions'] += 1

        if experience.prediction_error:
            perf['total_error'] += experience.prediction_error

        if experience.profit_loss:
            perf['total_profit'] += experience.profit_loss
            if experience.profit_loss > 0:
                perf['win_count'] += 1

    def get_recent_experiences(
        self,
        n: int = 100,
        model_name: Optional[str] = None,
        min_confidence: float = 0.0
    ) -> List[TradeExperience]:
        """
        Get recent completed experiences for training.

        Args:
            n: Number of experiences to return
            model_name: Filter by specific model (optional)
            min_confidence: Minimum confidence threshold

        Returns:
            List of TradeExperience objects
        """
        experiences = list(self.completed_experiences)

        # Filter by model
        if model_name:
            experiences = [e for e in experiences if e.model_name == model_name]

        # Filter by confidence
        experiences = [e for e in experiences if e.confidence >= min_confidence]

        # Return most recent n
        return experiences[-n:] if len(experiences) > n else experiences

    def get_training_batch(
        self,
        batch_size: int = 32,
        model_name: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a batch of experiences for model training.

        Args:
            batch_size: Number of samples
            model_name: Filter by model (optional)

        Returns:
            Tuple of (states, targets) as numpy arrays
        """
        experiences = self.get_recent_experiences(
            n=batch_size * 2,  # Get more to sample from
            model_name=model_name
        )

        if len(experiences) < batch_size:
            logger.warning(f"⚠️  Only {len(experiences)} experiences available (requested {batch_size})")
            batch_size = len(experiences)

        # Random sample
        indices = np.random.choice(len(experiences), size=batch_size, replace=False)
        sampled = [experiences[i] for i in indices]

        # Extract states and actual returns
        states = np.array([e.state for e in sampled])
        targets = np.array([e.actual_return for e in sampled])

        return states, targets

    def get_model_performance(self, model_name: str) -> Dict:
        """
        Get performance metrics for a specific model.

        Args:
            model_name: Name of model

        Returns:
            Dict with performance metrics
        """
        if model_name not in self.model_performance:
            return {
                'accuracy': 0.0,
                'avg_error': 0.0,
                'total_profit': 0.0,
                'win_rate': 0.0,
                'total_trades': 0
            }

        perf = self.model_performance[model_name]
        total = perf['total_predictions']

        if total == 0:
            return {
                'accuracy': 0.0,
                'avg_error': 0.0,
                'total_profit': 0.0,
                'win_rate': 0.0,
                'total_trades': 0
            }

        return {
            'accuracy': perf['correct_directions'] / total,
            'avg_error': perf['total_error'] / total,
            'total_profit': perf['total_profit'],
            'win_rate': perf['win_count'] / total,
            'total_trades': total
        }

    def save(self):
        """Save buffer to disk."""
        try:
            self.buffer_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                'completed_experiences': [e.to_dict() for e in self.completed_experiences],
                'pending_predictions': {k: v.to_dict() for k, v in self.pending_predictions.items()},
                'ticket_to_prediction': self.ticket_to_prediction,
                'model_performance': self.model_performance
            }

            with open(self.buffer_path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"✅ Experience buffer saved to {self.buffer_path}")
            logger.info(f"   Completed: {len(self.completed_experiences)}")
            logger.info(f"   Pending: {len(self.pending_predictions)}")

        except Exception as e:
            logger.error(f"❌ Failed to save experience buffer: {e}")

    def load(self):
        """Load buffer from disk."""
        try:
            if not self.buffer_path.exists():
                logger.info("No saved buffer found - starting fresh")
                return

            with open(self.buffer_path, 'r') as f:
                data = json.load(f)

            # Load completed experiences
            self.completed_experiences = deque(
                [TradeExperience.from_dict(e) for e in data['completed_experiences']],
                maxlen=self.max_size
            )

            # Load pending predictions
            self.pending_predictions = {
                k: TradeExperience.from_dict(v)
                for k, v in data['pending_predictions'].items()
            }

            # Load mappings
            self.ticket_to_prediction = {
                int(k): v for k, v in data['ticket_to_prediction'].items()
            }

            # Load performance
            self.model_performance = data.get('model_performance', {})

            logger.info(f"✅ Experience buffer loaded from {self.buffer_path}")
            logger.info(f"   Completed: {len(self.completed_experiences)}")
            logger.info(f"   Pending: {len(self.pending_predictions)}")

        except Exception as e:
            logger.error(f"❌ Failed to load experience buffer: {e}")

    def get_stats(self) -> Dict:
        """Get buffer statistics."""
        return {
            'completed_count': len(self.completed_experiences),
            'pending_count': len(self.pending_predictions),
            'total_capacity': self.max_size,
            'utilization': len(self.completed_experiences) / self.max_size,
            'models_tracked': list(self.model_performance.keys())
        }
