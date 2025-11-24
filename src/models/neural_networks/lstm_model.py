"""
LSTM (Long Short-Term Memory) model for price prediction.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from loguru import logger


class LSTMNetwork(nn.Module):
    """
    LSTM neural network architecture for time series prediction.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        bidirectional: bool = False,
        output_size: int = 1,
        task: str = 'regression'
    ):
        """
        Initialize LSTM network.

        Args:
            input_size: Number of input features
            hidden_size: Hidden layer size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Use bidirectional LSTM
            output_size: Output size (1 for regression, n for classification)
            task: 'regression' or 'classification'
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.task = task

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Attention mechanism
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size),
            nn.Tanh(),
            nn.Linear(lstm_output_size, 1),
            nn.Softmax(dim=1)
        )

        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(lstm_output_size)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, features)
            return_attention: Whether to return attention weights

        Returns:
            Output predictions, optionally with attention weights
        """
        # LSTM forward
        lstm_out, _ = self.lstm(x)

        # Layer normalization
        lstm_out = self.layer_norm(lstm_out)

        # Attention
        attention_weights = self.attention(lstm_out)
        context = torch.sum(attention_weights * lstm_out, dim=1)

        # Output
        output = self.fc(context)

        if self.task == 'classification':
            output = torch.softmax(output, dim=-1)

        if return_attention:
            return output, attention_weights.squeeze(-1)
        return output


class LSTMPredictor:
    """
    LSTM-based price predictor with training, evaluation, and self-learning capabilities.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        task: str = 'regression',
        device: Optional[str] = None
    ):
        """
        Initialize LSTM predictor.

        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            learning_rate: Learning rate
            task: 'regression' or 'classification'
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.task = task

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Initialize model
        output_size = 1 if task == 'regression' else 3  # up, neutral, down
        self.model = LSTMNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            output_size=output_size,
            task=task
        ).to(self.device)

        # Loss and optimizer
        if task == 'regression':
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )

        # Training history
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'train_metric': [],
            'val_metric': []
        }

        # Self-learning parameters
        self.prediction_history: List[Dict] = []
        self.performance_threshold = 0.1  # 10% improvement threshold

        logger.info(f"LSTMPredictor initialized on {self.device}")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        early_stopping: int = 20,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the LSTM model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size
            early_stopping: Early stopping patience
            verbose: Print training progress

        Returns:
            Training history dictionary
        """
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)

        if self.task == 'classification':
            y_train_t = y_train_t.long()

        # Create data loader
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Validation data
        if X_val is not None:
            X_val_t = torch.FloatTensor(X_val).to(self.device)
            y_val_t = torch.FloatTensor(y_val).to(self.device)
            if self.task == 'classification':
                y_val_t = y_val_t.long()

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_losses = []

            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)

                if self.task == 'regression':
                    loss = self.criterion(outputs.squeeze(), batch_y)
                else:
                    loss = self.criterion(outputs, batch_y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)
            self.history['train_loss'].append(avg_train_loss)

            # Validation phase
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_t)
                    if self.task == 'regression':
                        val_loss = self.criterion(val_outputs.squeeze(), y_val_t).item()
                    else:
                        val_loss = self.criterion(val_outputs, y_val_t).item()

                self.history['val_loss'].append(val_loss)
                self.scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self._save_best_weights()
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    self._load_best_weights()
                    break

                if verbose and (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch + 1}/{epochs} - "
                        f"Train Loss: {avg_train_loss:.6f} - "
                        f"Val Loss: {val_loss:.6f}"
                    )
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.6f}")

        return self.history

    def predict(
        self,
        X: np.ndarray,
        return_attention: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions.

        Args:
            X: Input features
            return_attention: Return attention weights

        Returns:
            Predictions, optionally with attention weights
        """
        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            if return_attention:
                outputs, attention = self.model(X_t, return_attention=True)
                return outputs.cpu().numpy(), attention.cpu().numpy()
            else:
                outputs = self.model(X_t)
                return outputs.cpu().numpy()

    def predict_with_confidence(
        self,
        X: np.ndarray,
        n_samples: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with confidence estimates using MC Dropout.

        Args:
            X: Input features
            n_samples: Number of Monte Carlo samples

        Returns:
            Tuple of (mean predictions, standard deviations)
        """
        self.model.train()  # Enable dropout
        X_t = torch.FloatTensor(X).to(self.device)

        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                outputs = self.model(X_t)
                predictions.append(outputs.cpu().numpy())

        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)

        self.model.eval()
        return mean_pred, std_pred

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(X_test)

        if self.task == 'regression':
            mse = np.mean((predictions.squeeze() - y_test) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions.squeeze() - y_test))

            # Direction accuracy
            if len(y_test) > 1:
                pred_direction = np.sign(np.diff(predictions.squeeze()))
                actual_direction = np.sign(np.diff(y_test))
                direction_acc = np.mean(pred_direction == actual_direction)
            else:
                direction_acc = 0.0

            return {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'direction_accuracy': float(direction_acc)
            }
        else:
            pred_classes = np.argmax(predictions, axis=1)
            accuracy = np.mean(pred_classes == y_test)
            return {'accuracy': float(accuracy)}

    def self_learn(
        self,
        recent_predictions: List[Dict],
        actual_outcomes: List[float],
        retrain_threshold: float = 0.1
    ) -> bool:
        """
        Self-learning mechanism to adapt to recent market conditions.

        Args:
            recent_predictions: List of recent prediction dictionaries
            actual_outcomes: Actual outcomes for those predictions
            retrain_threshold: Threshold for triggering retraining

        Returns:
            Whether model was updated
        """
        if len(recent_predictions) < 10:
            return False

        # Calculate recent accuracy
        correct = 0
        for pred, actual in zip(recent_predictions, actual_outcomes):
            pred_direction = 1 if pred['prediction'] > 0 else -1
            actual_direction = 1 if actual > 0 else -1
            if pred_direction == actual_direction:
                correct += 1

        recent_accuracy = correct / len(recent_predictions)

        # Check if performance degraded
        if hasattr(self, 'baseline_accuracy'):
            performance_drop = self.baseline_accuracy - recent_accuracy

            if performance_drop > retrain_threshold:
                logger.info(f"Performance dropped by {performance_drop:.2%}. Triggering self-learning...")

                # Prepare retraining data from recent predictions
                X_new = np.array([p['features'] for p in recent_predictions])
                y_new = np.array(actual_outcomes)

                # Fine-tune model with lower learning rate
                original_lr = self.optimizer.param_groups[0]['lr']
                self.optimizer.param_groups[0]['lr'] = original_lr * 0.1

                self.train(X_new, y_new, epochs=10, batch_size=8, verbose=False)

                self.optimizer.param_groups[0]['lr'] = original_lr
                logger.info("Self-learning update completed")
                return True

        # Update baseline
        self.baseline_accuracy = recent_accuracy
        return False

    def get_feature_importance(
        self,
        X: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """
        Get feature importance using gradient-based method.

        Args:
            X: Input features
            feature_names: Names of features

        Returns:
            Dictionary mapping feature names to importance scores
        """
        X_t = torch.FloatTensor(X).to(self.device)
        X_t.requires_grad = True

        self.model.eval()
        outputs = self.model(X_t)
        outputs.sum().backward()

        # Average gradients across samples and time steps
        importance = np.abs(X_t.grad.cpu().numpy()).mean(axis=(0, 1))

        # Normalize
        importance = importance / importance.sum()

        return dict(zip(feature_names, importance.tolist()))

    def _save_best_weights(self) -> None:
        """Save best model weights."""
        self._best_weights = {
            k: v.cpu().clone() for k, v in self.model.state_dict().items()
        }

    def _load_best_weights(self) -> None:
        """Load best model weights."""
        if hasattr(self, '_best_weights'):
            self.model.load_state_dict(self._best_weights)

    def save(self, filepath: Union[str, Path]) -> None:
        """Save model to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'config': {
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'learning_rate': self.learning_rate,
                'task': self.task
            }
        }, filepath)

        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: Union[str, Path]) -> None:
        """Load model from file."""
        filepath = Path(filepath)
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']

        logger.info(f"Model loaded from {filepath}")

    @classmethod
    def from_checkpoint(
        cls,
        filepath: Union[str, Path],
        device: Optional[str] = None
    ) -> 'LSTMPredictor':
        """Create predictor from saved checkpoint."""
        checkpoint = torch.load(filepath, map_location='cpu')
        config = checkpoint['config']

        predictor = cls(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            learning_rate=config['learning_rate'],
            task=config['task'],
            device=device
        )

        predictor.load(filepath)
        return predictor
