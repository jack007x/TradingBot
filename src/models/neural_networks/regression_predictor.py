"""
Regression Predictor - LSTM/GRU model for predicting CONTINUOUS returns.

WHY REGRESSION INSTEAD OF CLASSIFICATION:
1. Markets are continuous, not discrete (returns are real numbers, not categories)
2. NO class imbalance issues (returns follow continuous distribution)
3. Captures magnitude of moves, not just direction
4. Natural confidence scores (prediction magnitude = confidence)
5. More information in labels (0.5% up is different from 2% up)
6. NO need for SMOTE or artificial balancing
7. Train-test distribution match (both use real continuous returns)

FIXES THESE CRITICAL BUGS:
- ❌ Class imbalance (97.4% neutral) → ✅ Continuous distribution
- ❌ SMOTE unrealistic data → ✅ Real market returns only
- ❌ Train-test mismatch → ✅ Same distribution
- ❌ Threshold dependency → ✅ No thresholds needed
- ❌ Binary predictions → ✅ Probabilistic predictions
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from loguru import logger
from sklearn.metrics import mean_squared_error, mean_absolute_error


class RegressionLSTM(nn.Module):
    """
    LSTM for continuous return prediction.

    Key differences from classification:
    - Output: Single continuous value (not 3 classes)
    - Loss: MSE (not CrossEntropy/Focal Loss)
    - No softmax, no class probabilities
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Bidirectional LSTM for sequence processing
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True  # Capture patterns from both directions
        )

        # Attention mechanism (same as before, helps focus on important timesteps)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # Regression head (MUCH SIMPLER than classification head!)
        # No need for complex multi-layer classifier
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)  # ← KEY CHANGE: Single output, not 3 classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch_size, sequence_length, input_size)

        Returns:
            predictions: (batch_size, 1) - continuous return predictions
        """
        # LSTM processing
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden*2)

        # Attention weights
        attention_weights = self.attention(lstm_out)  # (batch, seq, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)

        # Weighted sum
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden*2)

        # Regression output (NO SOFTMAX! Just linear output)
        prediction = self.regressor(context)  # (batch, 1)

        return prediction


class RegressionGRU(nn.Module):
    """
    GRU for continuous return prediction.
    Alternative to LSTM, often faster and similarly effective.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(x)

        attention_weights = self.attention(gru_out)
        attention_weights = torch.softmax(attention_weights, dim=1)

        context = torch.sum(attention_weights * gru_out, dim=1)
        prediction = self.regressor(context)

        return prediction


class RegressionPredictor:
    """
    Main regression predictor class.

    CRITICAL ADVANTAGES OVER CLASSIFICATION:
    1. NO class imbalance - returns are continuous
    2. NO SMOTE needed - all data is real
    3. Train-test distribution match - both use real returns
    4. Captures magnitude - 0.5% vs 2% move distinction
    5. Natural confidence - larger predictions = more confident
    6. Simpler training - just MSE loss, no Focal Loss complexity
    """

    def __init__(
        self,
        input_size: int,
        model_type: str = 'lstm',
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        device: Optional[str] = None
    ):
        """
        Initialize regression predictor.

        Args:
            input_size: Number of input features
            model_type: 'lstm' or 'gru'
            hidden_size: Hidden layer size
            num_layers: Number of RNN layers
            dropout: Dropout rate
            learning_rate: Learning rate
            weight_decay: L2 regularization
            device: Device to use
        """
        self.input_size = input_size
        self.model_type = model_type

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Create model
        if model_type == 'lstm':
            self.model = RegressionLSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout
            ).to(self.device)
        elif model_type == 'gru':
            self.model = RegressionGRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # MSE Loss (MUCH SIMPLER than Focal Loss!)
        # NO class weights needed, NO balancing needed
        self.criterion = nn.MSELoss()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler (OneCycleLR works great for regression too)
        self.scheduler = None  # Will be created in train()

        # Training history
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'train_direction_acc': [],
            'val_direction_acc': [],
            'train_correlation': [],
            'val_correlation': []
        }

        logger.info(f"RegressionPredictor ({model_type.upper()}) initialized on {self.device}")
        logger.info(f"Input size: {input_size}, Hidden: {hidden_size}, Layers: {num_layers}")
        logger.info(f"Using MSE loss for continuous prediction (NO class imbalance issues!)")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 64,
        early_stopping_patience: int = 20,
        min_delta: float = 0.00001
    ) -> Dict[str, List[float]]:
        """
        Train the regression model.

        KEY DIFFERENCES from classification training:
        - NO SMOTE resampling
        - NO class weights
        - NO stratified split
        - NO balanced batch sampling
        - Just simple, straightforward MSE minimization

        Args:
            X_train: Training sequences (samples, seq_len, features)
            y_train: Training returns (samples,) - CONTINUOUS values
            X_val: Validation sequences
            y_val: Validation returns
            epochs: Number of epochs
            batch_size: Batch size
            early_stopping_patience: Patience for early stopping
            min_delta: Minimum improvement for early stopping

        Returns:
            Training history
        """
        # Log return distribution (continuous, not discrete classes!)
        logger.info(f"Training samples: {len(y_train)}")
        logger.info(f"Return statistics:")
        logger.info(f"  Mean: {y_train.mean():.6f}")
        logger.info(f"  Std:  {y_train.std():.6f}")
        logger.info(f"  Min:  {y_train.min():.6f}")
        logger.info(f"  Max:  {y_train.max():.6f}")
        logger.info(f"  Median: {np.median(y_train):.6f}")

        # NO SMOTE, NO class balancing - just use real data!
        # This is a HUGE advantage of regression

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)

        train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)

        # NO WeightedRandomSampler needed! Just simple random shuffle
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,  # Simple shuffle, no fancy sampling
            drop_last=True
        )

        # Create scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.optimizer.param_groups[0]['lr'] * 2,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            anneal_strategy='cos'
        )

        if X_val is not None and y_val is not None:
            X_val_t = torch.FloatTensor(X_val).to(self.device)
            y_val_t = torch.FloatTensor(y_val).to(self.device)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_losses = []
            train_predictions = []
            train_actuals = []

            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()

                predictions = self.model(batch_X).squeeze()
                loss = self.criterion(predictions, batch_y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()  # Step after each batch

                train_losses.append(loss.item())
                train_predictions.extend(predictions.detach().cpu().numpy())
                train_actuals.extend(batch_y.detach().cpu().numpy())

            train_loss = np.mean(train_losses)

            # Calculate directional accuracy (for trading evaluation)
            train_direction_acc = self._calculate_directional_accuracy(
                train_predictions, train_actuals
            )

            # Calculate correlation (key metric for regression)
            train_correlation = np.corrcoef(train_predictions, train_actuals)[0, 1]

            self.history['train_loss'].append(train_loss)
            self.history['train_direction_acc'].append(train_direction_acc)
            self.history['train_correlation'].append(train_correlation)

            # Validation
            if X_val is not None:
                self.model.eval()
                val_losses = []
                val_predictions = []
                val_actuals = []

                with torch.no_grad():
                    # Process validation data in batches
                    for i in range(0, len(X_val_t), batch_size):
                        batch_X = X_val_t[i:i+batch_size]
                        batch_y = y_val_t[i:i+batch_size]

                        predictions = self.model(batch_X).squeeze()
                        loss = self.criterion(predictions, batch_y)

                        val_losses.append(loss.item())
                        val_predictions.extend(predictions.cpu().numpy())
                        val_actuals.extend(batch_y.cpu().numpy())

                val_loss = np.mean(val_losses)
                val_direction_acc = self._calculate_directional_accuracy(
                    val_predictions, val_actuals
                )
                val_correlation = np.corrcoef(val_predictions, val_actuals)[0, 1]

                self.history['val_loss'].append(val_loss)
                self.history['val_direction_acc'].append(val_direction_acc)
                self.history['val_correlation'].append(val_correlation)

                # Logging (every 5 epochs)
                if (epoch + 1) % 5 == 0:
                    logger.info(
                        f"Epoch {epoch + 1}/{epochs} - "
                        f"Train Loss: {train_loss:.6f}, "
                        f"Val Loss: {val_loss:.6f}"
                    )
                    logger.info(
                        f"  Train Dir Acc: {train_direction_acc:.4f}, "
                        f"Val Dir Acc: {val_direction_acc:.4f}"
                    )
                    logger.info(
                        f"  Train Corr: {train_correlation:.4f}, "
                        f"Val Corr: {val_correlation:.4f}"
                    )

                    # Show prediction distribution
                    logger.info(
                        f"  Val Predictions - Mean: {np.mean(val_predictions):.6f}, "
                        f"Std: {np.std(val_predictions):.6f}"
                    )

                # Early stopping based on validation loss
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    self.best_state = {
                        'model': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'epoch': epoch,
                        'val_loss': val_loss,
                        'val_direction_acc': val_direction_acc,
                        'val_correlation': val_correlation
                    }
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        # Restore best model
                        if hasattr(self, 'best_state'):
                            self.model.load_state_dict(self.best_state['model'])
                            self.optimizer.load_state_dict(self.best_state['optimizer'])
                        break
            else:
                if (epoch + 1) % 5 == 0:
                    logger.info(
                        f"Epoch {epoch + 1}/{epochs} - "
                        f"Train Loss: {train_loss:.6f}, "
                        f"Dir Acc: {train_direction_acc:.4f}, "
                        f"Corr: {train_correlation:.4f}"
                    )

        return self.history

    def _calculate_directional_accuracy(
        self,
        predictions: List[float],
        actuals: List[float]
    ) -> float:
        """
        Calculate directional accuracy (for trading evaluation).

        This measures: "Did we predict the correct direction?"
        - If actual > 0 and prediction > 0: Correct (both positive)
        - If actual < 0 and prediction < 0: Correct (both negative)
        - Otherwise: Wrong

        This is THE KEY METRIC for trading!
        """
        correct = sum(
            (p > 0 and a > 0) or (p < 0 and a < 0)
            for p, a in zip(predictions, actuals)
        )
        return correct / len(predictions)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict continuous returns.

        Args:
            X: Input sequences (samples, seq_len, features)

        Returns:
            predictions: (samples,) continuous return predictions
        """
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_t).squeeze()
            return predictions.cpu().numpy()

    def predict_single(self, x: np.ndarray) -> float:
        """
        Predict single sample.

        Args:
            x: Single sequence (seq_len, features)

        Returns:
            prediction: Single continuous return prediction
        """
        prediction = self.predict(x[np.newaxis, :, :])
        return float(prediction[0])

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Returns comprehensive metrics for regression models.
        """
        predictions = self.predict(X_test)

        # Statistical metrics
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mse)

        # Correlation (key indicator of predictive power)
        correlation = np.corrcoef(predictions, y_test)[0, 1]

        # Directional accuracy (for trading)
        direction_acc = self._calculate_directional_accuracy(
            predictions.tolist(), y_test.tolist()
        )

        # R-squared
        ss_res = np.sum((y_test - predictions) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        results = {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'r_squared': float(r_squared),
            'correlation': float(correlation),
            'directional_accuracy': float(direction_acc)
        }

        logger.info(f"Evaluation Results:")
        logger.info(f"  MSE: {mse:.6f}")
        logger.info(f"  MAE: {mae:.6f}")
        logger.info(f"  RMSE: {rmse:.6f}")
        logger.info(f"  R²: {r_squared:.4f}")
        logger.info(f"  Correlation: {correlation:.4f}")
        logger.info(f"  Directional Accuracy: {direction_acc:.4f}")

        return results

    def save(self, filepath: Union[str, Path]):
        """Save model to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_type': self.model_type,
            'input_size': self.input_size,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'history': self.history
        }, filepath)

        logger.info(f"RegressionPredictor saved to {filepath}")

    def load(self, filepath: Union[str, Path]):
        """Load model from file."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.history = checkpoint['history']

        logger.info(f"RegressionPredictor loaded from {filepath}")
