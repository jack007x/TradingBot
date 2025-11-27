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
        loss_fn: str = 'mse',
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
            loss_fn: Loss function ('mse', 'trading', 'variance', 'huber')
            device: Device to use
        """
        self.input_size = input_size
        self.model_type = model_type
        self.loss_fn_name = loss_fn

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
        elif model_type == 'attention_lstm':
            # Advanced LSTM with multi-head attention
            from .attention_predictor import AttentionRegressionLSTM
            self.model = AttentionRegressionLSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_heads=4,  # 4-head attention
                dropout=dropout
            ).to(self.device)
        elif model_type == 'attention_gru':
            # Advanced GRU with multi-head attention
            from .attention_predictor import AttentionRegressionGRU
            self.model = AttentionRegressionGRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_heads=4,  # 4-head attention
                dropout=dropout
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {model_type}. Choose from: lstm, gru, attention_lstm, attention_gru")

        # Select loss function
        # Import custom losses
        from ...losses.trading_losses import (
            TradingLoss, VarianceMatchingLoss, HuberLoss
        )

        if loss_fn == 'mse':
            # Standard MSE (baseline)
            self.criterion = nn.MSELoss()
            self.use_custom_loss = False
        elif loss_fn == 'trading':
            # Trading-focused loss (directional + variance)
            self.criterion = TradingLoss()
            self.use_custom_loss = True
        elif loss_fn == 'variance':
            # Variance matching loss (anti-conservatism)
            self.criterion = VarianceMatchingLoss()
            self.use_custom_loss = True
        elif loss_fn == 'huber':
            # Huber loss (robust to outliers)
            self.criterion = HuberLoss()
            self.use_custom_loss = True
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}. Choose from: mse, trading, variance, huber")

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

        # Loss component history (for custom losses)
        if self.use_custom_loss:
            self.history['loss_components'] = []

        logger.info(f"RegressionPredictor ({model_type.upper()}) initialized on {self.device}")
        logger.info(f"Input size: {input_size}, Hidden: {hidden_size}, Layers: {num_layers}")
        logger.info(f"Using {loss_fn.upper()} loss for continuous prediction")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 64,
        early_stopping_patience: int = 20,
        min_delta: float = 0.00001,
        min_epochs_before_check: int = 15,
        negative_corr_streak_threshold: int = 5
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
            min_epochs_before_check: Minimum epochs before checking negative correlation (default 15)
            negative_corr_streak_threshold: Consecutive negative epochs before stopping (default 5)

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
        best_val_correlation = float('-inf')  # Track best correlation
        patience_counter = 0
        correlation_patience_counter = 0  # Separate patience for correlation
        negative_corr_streak = 0  # Track consecutive negative correlation epochs

        logger.info(f"Smart early stopping configured:")
        logger.info(f"  - Minimum epochs before negative check: {min_epochs_before_check}")
        logger.info(f"  - Negative correlation streak threshold: {negative_corr_streak_threshold}")
        logger.info(f"  - Correlation-based patience: {early_stopping_patience}")

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_losses = []
            train_predictions = []
            train_actuals = []
            epoch_loss_components = []  # For custom losses

            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()

                predictions = self.model(batch_X).squeeze()

                # Handle custom losses that return (loss, components)
                if self.use_custom_loss:
                    loss, components = self.criterion(predictions, batch_y)
                    epoch_loss_components.append(components)
                else:
                    loss = self.criterion(predictions, batch_y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()  # Step after each batch

                train_losses.append(loss.item())
                train_predictions.extend(predictions.detach().cpu().numpy())
                train_actuals.extend(batch_y.detach().cpu().numpy())

            train_loss = np.mean(train_losses)

            # Average loss components for this epoch
            if self.use_custom_loss and epoch_loss_components:
                avg_components = {}
                for key in epoch_loss_components[0].keys():
                    avg_components[key] = np.mean([c[key] for c in epoch_loss_components])
                self.history['loss_components'].append(avg_components)

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

                        # Handle custom losses
                        if self.use_custom_loss:
                            loss, _ = self.criterion(predictions, batch_y)
                        else:
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

                    # Log custom loss components
                    if self.use_custom_loss and epoch_loss_components:
                        comp = avg_components
                        if 'std_ratio' in comp:
                            logger.info(
                                f"  Pred/Target Std Ratio: {comp['std_ratio']:.4f} "
                                f"(Target: match 1.0 for full variance)"
                            )
                        if 'mse' in comp and 'direction' in comp:
                            logger.info(
                                f"  Loss Components - MSE: {comp['mse']:.6f}, "
                                f"Direction: {comp['direction']:.6f}, "
                                f"Variance: {comp.get('variance', 0):.6f}"
                            )

                # 🚨 SMART NEGATIVE CORRELATION DETECTION
                # Only check after minimum epochs to give model time to learn
                if epoch >= min_epochs_before_check:
                    # Track negative correlation streak
                    if val_correlation < -0.05:  # Significantly negative
                        negative_corr_streak += 1
                        logger.warning(
                            f"⚠️  Negative correlation: {val_correlation:.4f} "
                            f"(streak: {negative_corr_streak}/{negative_corr_streak_threshold})"
                        )
                    else:
                        # Reset streak if correlation becomes positive
                        if negative_corr_streak > 0:
                            logger.info(
                                f"✅ Correlation recovered to {val_correlation:.4f} "
                                f"(streak reset from {negative_corr_streak})"
                            )
                        negative_corr_streak = 0

                    # Emergency stop: Severely negative correlation
                    if val_correlation < -0.15:
                        logger.error(f"🚨 SEVERE NEGATIVE CORRELATION: {val_correlation:.4f}")
                        logger.error(f"   Model is strongly predicting OPPOSITE direction!")
                        logger.error(f"   This indicates major training issues - stopping immediately!")
                        # Restore best model (if any)
                        if hasattr(self, 'best_state'):
                            logger.info(f"   Restoring best model (epoch {self.best_state['epoch']+1})")
                            logger.info(f"   Best correlation: {self.best_state['val_correlation']:.4f}")
                            self.model.load_state_dict(self.best_state['model'])
                            self.optimizer.load_state_dict(self.best_state['optimizer'])
                        break

                    # Stop if consistently negative (streak threshold reached)
                    if negative_corr_streak >= negative_corr_streak_threshold:
                        logger.error(
                            f"🚨 PERSISTENT NEGATIVE CORRELATION: "
                            f"{negative_corr_streak} consecutive epochs below -0.05"
                        )
                        logger.error(f"   Model consistently predicting opposite direction!")
                        logger.error(f"   Current correlation: {val_correlation:.4f}")
                        logger.error(f"   Stopping training to prevent further degradation")
                        # Restore best model
                        if hasattr(self, 'best_state'):
                            logger.info(f"   Restoring best model (epoch {self.best_state['epoch']+1})")
                            logger.info(f"   Best correlation: {self.best_state['val_correlation']:.4f}")
                            self.model.load_state_dict(self.best_state['model'])
                            self.optimizer.load_state_dict(self.best_state['optimizer'])
                        break
                else:
                    # Still in minimum epochs window - just warn
                    if val_correlation < 0:
                        logger.info(
                            f"ℹ️  Early epoch {epoch+1}/{min_epochs_before_check}: "
                            f"Negative correlation {val_correlation:.4f} (allowing model to stabilize)"
                        )

                # Warning for low correlation (anytime)
                if val_correlation < 0.05 and val_correlation >= 0:
                    logger.warning(f"⚠️  Low correlation: {val_correlation:.4f} (target: >0.08)")
                    logger.warning(f"   Model predictions weakly correlated with actual returns")

                # Early stopping based on CORRELATION (primary metric for trading!)
                # Correlation is more important than raw loss for trading performance
                if val_correlation > best_val_correlation + min_delta:
                    best_val_correlation = val_correlation
                    correlation_patience_counter = 0
                    # Save best model based on correlation
                    self.best_state = {
                        'model': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'epoch': epoch,
                        'val_loss': val_loss,
                        'val_direction_acc': val_direction_acc,
                        'val_correlation': val_correlation
                    }
                    logger.debug(f"📊 New best correlation: {val_correlation:.4f} (epoch {epoch+1})")
                else:
                    correlation_patience_counter += 1

                # Also track best loss (secondary metric)
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Early stopping: Use correlation patience as primary, loss as backup
                if correlation_patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1} (correlation not improving)")
                    # Restore best model (based on correlation!)
                    if hasattr(self, 'best_state'):
                        logger.info(f"Restoring best model from epoch {self.best_state['epoch']}")
                        logger.info(f"  Best correlation: {self.best_state['val_correlation']:.4f}")
                        logger.info(f"  Best val loss: {self.best_state['val_loss']:.6f}")
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
        """
        Save model to file with version and config.

        CRITICAL: Now saves actual architecture from model (not hardcoded values!)
        Includes version tracking for future compatibility.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Extract actual architecture from model
        model = self.model

        # Detect RNN module (LSTM or GRU)
        if hasattr(model, 'lstm'):
            rnn_module = model.lstm
        elif hasattr(model, 'gru'):
            rnn_module = model.gru
        else:
            rnn_module = None

        # Extract architecture parameters from actual model
        if rnn_module:
            num_layers = rnn_module.num_layers
            hidden_size = rnn_module.hidden_size
            dropout = rnn_module.dropout if num_layers > 1 else 0.0
        else:
            # Fallback to stored values
            num_layers = getattr(self, 'num_layers', 2)
            hidden_size = getattr(self, 'hidden_size', 128)
            dropout = 0.3

        # Get attention heads if applicable
        attention_heads = getattr(model, 'num_heads', 4) if hasattr(model, 'attention') else None

        logger.info(f"Saving model architecture: {self.model_type}, {num_layers} layers, {hidden_size} hidden")

        # Save with version and ACTUAL config from model
        checkpoint = {
            'version': '2.1',  # Version 2.1: Fixed architecture saving
            'model_type': self.model_type,
            'input_size': self.input_size,
            'loss_fn': self.loss_fn_name,
            'config': {
                'input_size': self.input_size,
                'model_type': self.model_type,
                'hidden_size': hidden_size,  # ✅ From actual model
                'num_layers': num_layers,    # ✅ From actual model (CRITICAL FIX!)
                'dropout': dropout,           # ✅ From actual model
                'attention_heads': attention_heads,  # For attention models
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'loss_fn': self.loss_fn_name
            },
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'history': self.history,
            'device': str(self.device)
        }

        torch.save(checkpoint, filepath)
        logger.info(f"✅ RegressionPredictor v2.1 saved to {filepath}")
        logger.info(f"   Architecture: {self.model_type}, Layers: {num_layers}, Hidden: {hidden_size}")

    def load(self, filepath: Union[str, Path]):
        """
        Load model from file with backward compatibility.

        Handles both:
        - Old format (no version, no config)
        - New format (with version and config)
        """
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        # Check version for compatibility
        version = checkpoint.get('version', '1.0')

        if version == '1.0':
            # Old format - no config, no version
            logger.warning(f"Loading old format checkpoint (v1.0) from {filepath}")
            logger.warning("Consider retraining model with new TradingLoss for better performance!")

            # Load what we can from old format
            if 'model_state' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state'])
            elif 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Very old format - just weights
                self.model.load_state_dict(checkpoint)

            if 'optimizer_state' in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state'])
                except Exception as e:
                    logger.warning(f"Could not load optimizer state: {e}")

            if 'history' in checkpoint:
                self.history = checkpoint['history']

        else:
            # New format (v2.0+) - has config and version
            logger.info(f"Loading checkpoint v{version} from {filepath}")

            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.history = checkpoint['history']

        logger.info(f"RegressionPredictor loaded from {filepath}")

    @classmethod
    def from_checkpoint(
        cls,
        filepath: Union[str, Path],
        device: Optional[str] = None
    ) -> 'RegressionPredictor':
        """
        Create new RegressionPredictor from saved checkpoint.

        Handles backward compatibility with old model formats.

        Args:
            filepath: Path to saved checkpoint
            device: Device to load model on

        Returns:
            Loaded RegressionPredictor instance
        """
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)

        version = checkpoint.get('version', '1.0')

        if version == '1.0':
            # Old format - infer config from checkpoint
            logger.warning(f"Loading old format checkpoint (v1.0) from {filepath}")
            logger.warning("⚠️  Old models may not have TradingLoss enabled!")
            logger.warning("⚠️  Recommend deleting and retraining for better performance!")

            model_type = checkpoint.get('model_type', 'lstm')
            input_size = checkpoint.get('input_size', None)

            # Try to infer input_size from model state
            if input_size is None:
                model_state = checkpoint.get('model_state', checkpoint.get('model_state_dict', checkpoint))
                input_size = cls._infer_input_size_from_state(model_state, model_type)

            # Create with default config (MSE loss!)
            predictor = cls(
                input_size=input_size,
                model_type=model_type,
                loss_fn='mse',  # Old models used MSE
                device=device
            )

        else:
            # New format (v2.0+) - has config
            logger.info(f"Loading checkpoint v{version} from {filepath}")

            config = checkpoint['config']
            model_state = checkpoint.get('model_state', checkpoint.get('model_state_dict'))

            # CRITICAL: Detect actual layer count from weights (don't trust config!)
            detected_layers = cls._detect_num_layers(model_state)
            config_layers = config.get('num_layers', 2)

            logger.info("=" * 70)
            logger.info("MODEL ARCHITECTURE VALIDATION")
            logger.info("=" * 70)
            logger.info(f"Config says: {config_layers} layers")
            logger.info(f"Weights have: {detected_layers} layers")

            # Use detected layers (trust the weights, not the config!)
            if detected_layers != config_layers:
                logger.warning(f"⚠️  ARCHITECTURE MISMATCH DETECTED!")
                logger.warning(f"⚠️  Config: {config_layers} layers, Weights: {detected_layers} layers")
                logger.warning(f"⚠️  Using detected value: {detected_layers} layers")
                num_layers = detected_layers
            else:
                logger.info(f"✅ Architecture consistent: {detected_layers} layers")
                num_layers = detected_layers

            predictor = cls(
                input_size=config['input_size'],
                model_type=config['model_type'],
                hidden_size=config.get('hidden_size', 128),
                num_layers=num_layers,  # Use detected layers!
                dropout=config.get('dropout', 0.3),
                learning_rate=config.get('learning_rate', 1e-3),
                loss_fn=config.get('loss_fn', 'mse'),
                device=device
            )

        # Load model weights with flexible loading
        model_state = checkpoint.get('model_state', checkpoint.get('model_state_dict'))
        if model_state:
            try:
                predictor.model.load_state_dict(model_state, strict=True)
                logger.info("✅ Model weights loaded successfully (strict mode)")
            except RuntimeError as e:
                logger.warning(f"⚠️  Strict loading failed: {e}")
                logger.info("🔄 Attempting flexible loading (strict=False)...")
                predictor.model.load_state_dict(model_state, strict=False)
                logger.info("✅ Model weights loaded (flexible mode - some weights may be skipped)")

            logger.info("=" * 70)

        # Load optimizer state (if available)
        if 'optimizer_state' in checkpoint:
            try:
                predictor.optimizer.load_state_dict(checkpoint['optimizer_state'])
            except Exception as e:
                logger.warning(f"Could not load optimizer state: {e}")

        # Load history (if available)
        if 'history' in checkpoint:
            predictor.history = checkpoint['history']

        return predictor

    @staticmethod
    def _detect_num_layers(state_dict: dict) -> int:
        """
        Detect number of LSTM/GRU layers from state_dict keys.

        CRITICAL: This prevents architecture mismatch errors by detecting
        the actual layer count from saved weights instead of trusting config.

        Example keys:
        - lstm.weight_ih_l0, lstm.weight_ih_l1, lstm.weight_ih_l2
        - Layer indices: l0, l1, l2 → 3 layers (0-indexed)

        Args:
            state_dict: Model state dictionary

        Returns:
            Number of layers detected from weights
        """
        layer_indices = set()

        for key in state_dict.keys():
            # Look for patterns like "lstm.weight_ih_l2" or "gru.weight_hh_l1"
            if '_l' in key:
                # Extract layer index (e.g., "lstm.weight_ih_l2" -> "2")
                parts = key.split('_l')
                if len(parts) > 1:
                    # Get the layer number (might have suffix like "_reverse")
                    layer_idx_str = parts[1].split('_')[0].split('.')[0]
                    if layer_idx_str.isdigit():
                        layer_indices.add(int(layer_idx_str))

        if layer_indices:
            num_layers = max(layer_indices) + 1  # Convert 0-indexed to count
            logger.debug(f"Detected {num_layers} layers from state_dict (indices: {sorted(layer_indices)})")
            return num_layers

        # Fallback: assume 2 layers (common default)
        logger.warning("Could not detect layer count from state_dict, assuming 2 layers")
        return 2

    @staticmethod
    def _infer_input_size_from_state(state_dict: dict, model_type: str) -> int:
        """
        Infer input size from model state dict.

        Args:
            state_dict: Model state dictionary
            model_type: 'lstm' or 'gru'

        Returns:
            Inferred input size
        """
        # Try to find input layer weights
        if model_type == 'lstm':
            weight_key = 'lstm.weight_ih_l0'
        elif model_type == 'gru':
            weight_key = 'gru.weight_ih_l0'
        else:
            weight_key = None

        if weight_key and weight_key in state_dict:
            weight = state_dict[weight_key]
            # For LSTM/GRU: weight shape is (4*hidden_size or 3*hidden_size, input_size)
            input_size = weight.shape[1]
            logger.info(f"Inferred input_size={input_size} from state dict")
            return input_size

        # Fallback default
        logger.warning("Could not infer input_size from state dict, using default=78")
        return 78
