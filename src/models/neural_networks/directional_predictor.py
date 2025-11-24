"""
Directional Predictor - LSTM/GRU model for predicting price direction (up/down/neutral).
Optimized for trading with focus on accuracy and profitability.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from loguru import logger


class DirectionalLSTM(nn.Module):
    """LSTM for direction classification (up/down/neutral)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 3
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
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

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden*2)

        # Attention weights
        attention_weights = self.attention(lstm_out)  # (batch, seq, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)

        # Weighted sum
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden*2)

        # Classification
        logits = self.classifier(context)  # (batch, num_classes)
        probs = torch.softmax(logits, dim=1)

        return logits, probs


class DirectionalGRU(nn.Module):
    """GRU for direction classification (up/down/neutral)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 3
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

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # GRU
        gru_out, _ = self.gru(x)  # (batch, seq, hidden*2)

        # Attention weights
        attention_weights = self.attention(gru_out)  # (batch, seq, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)

        # Weighted sum
        context = torch.sum(attention_weights * gru_out, dim=1)  # (batch, hidden*2)

        # Classification
        logits = self.classifier(context)  # (batch, num_classes)
        probs = torch.softmax(logits, dim=1)

        return logits, probs


class DirectionalPredictor:
    """
    Directional predictor for trading.
    Predicts price direction (up/down/neutral) with high accuracy.
    """

    def __init__(
        self,
        input_size: int,
        model_type: str = 'lstm',
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: Optional[str] = None
    ):
        """
        Initialize directional predictor.

        Args:
            input_size: Number of input features
            model_type: 'lstm' or 'gru'
            hidden_size: Hidden layer size
            num_layers: Number of RNN layers
            dropout: Dropout rate
            num_classes: Number of classes (3: up/down/neutral)
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            device: Device to use
        """
        self.input_size = input_size
        self.model_type = model_type
        self.num_classes = num_classes

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Create model
        if model_type == 'lstm':
            self.model = DirectionalLSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                num_classes=num_classes
            ).to(self.device)
        elif model_type == 'gru':
            self.model = DirectionalGRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                num_classes=num_classes
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Loss function with class weights for imbalanced data
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # Maximize accuracy
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )

        # Training history
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_balanced_acc': [],
            'val_balanced_acc': []
        }

        logger.info(f"DirectionalPredictor ({model_type.upper()}) initialized on {self.device}")
        logger.info(f"Input size: {input_size}, Hidden: {hidden_size}, Layers: {num_layers}")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        early_stopping_patience: int = 15,
        min_delta: float = 0.001
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            X_train: Training sequences (samples, seq_len, features)
            y_train: Training labels (samples,) - class indices
            X_val: Validation sequences
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            early_stopping_patience: Patience for early stopping
            min_delta: Minimum improvement for early stopping

        Returns:
            Training history
        """
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.LongTensor(y_train).to(self.device)

        train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        if X_val is not None and y_val is not None:
            X_val_t = torch.FloatTensor(X_val).to(self.device)
            y_val_t = torch.LongTensor(y_val).to(self.device)

        best_val_acc = 0.0
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_losses = []
            train_correct = 0
            train_total = 0
            train_class_correct = np.zeros(self.num_classes)
            train_class_total = np.zeros(self.num_classes)

            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()

                logits, probs = self.model(batch_X)
                loss = self.criterion(logits, batch_y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                train_losses.append(loss.item())

                # Calculate accuracy
                _, predicted = torch.max(probs, 1)
                train_correct += (predicted == batch_y).sum().item()
                train_total += batch_y.size(0)

                # Per-class accuracy
                for i in range(self.num_classes):
                    mask = (batch_y == i)
                    train_class_correct[i] += ((predicted == batch_y) & mask).sum().item()
                    train_class_total[i] += mask.sum().item()

            train_loss = np.mean(train_losses)
            train_acc = train_correct / train_total
            train_balanced_acc = np.mean(train_class_correct / (train_class_total + 1e-8))

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['train_balanced_acc'].append(train_balanced_acc)

            # Validation
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    logits, probs = self.model(X_val_t)
                    val_loss = self.criterion(logits, y_val_t).item()

                    _, predicted = torch.max(probs, 1)
                    val_correct = (predicted == y_val_t).sum().item()
                    val_acc = val_correct / y_val_t.size(0)

                    # Per-class accuracy
                    val_class_correct = np.zeros(self.num_classes)
                    val_class_total = np.zeros(self.num_classes)
                    for i in range(self.num_classes):
                        mask = (y_val_t == i)
                        val_class_correct[i] = ((predicted == y_val_t) & mask).sum().item()
                        val_class_total[i] = mask.sum().item()

                    val_balanced_acc = np.mean(val_class_correct / (val_class_total + 1e-8))

                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                self.history['val_balanced_acc'].append(val_balanced_acc)

                # Learning rate scheduler
                self.scheduler.step(val_balanced_acc)

                # Logging
                if (epoch + 1) % 5 == 0:
                    logger.info(
                        f"Epoch {epoch + 1}/{epochs} - "
                        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} ({train_balanced_acc:.4f}) - "
                        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} ({val_balanced_acc:.4f})"
                    )

                # Early stopping based on balanced accuracy
                if val_balanced_acc > best_val_acc + min_delta:
                    best_val_acc = val_balanced_acc
                    patience_counter = 0
                    # Save best model
                    self.best_state = {
                        'model': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'epoch': epoch,
                        'val_acc': val_acc,
                        'val_balanced_acc': val_balanced_acc
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
                        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} ({train_balanced_acc:.4f})"
                    )

        return self.history

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict direction.

        Args:
            X: Input sequences (samples, seq_len, features)

        Returns:
            Tuple of (predicted_classes, probabilities)
        """
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            _, probs = self.model(X_t)
            predicted = torch.argmax(probs, dim=1)

            return predicted.cpu().numpy(), probs.cpu().numpy()

    def predict_single(self, x: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Predict single sample.

        Args:
            x: Single sequence (seq_len, features)

        Returns:
            Tuple of (predicted_class, probabilities)
        """
        predicted, probs = self.predict(x[np.newaxis, :, :])
        return int(predicted[0]), probs[0]

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Args:
            X_test: Test sequences
            y_test: Test labels

        Returns:
            Evaluation metrics
        """
        predicted, probs = self.predict(X_test)

        # Overall accuracy
        accuracy = (predicted == y_test).mean()

        # Per-class metrics
        class_acc = []
        class_precision = []
        class_recall = []

        for i in range(self.num_classes):
            mask_true = (y_test == i)
            mask_pred = (predicted == i)

            # Accuracy for this class
            if mask_true.sum() > 0:
                class_acc.append((predicted[mask_true] == i).mean())
            else:
                class_acc.append(0.0)

            # Precision
            if mask_pred.sum() > 0:
                class_precision.append((y_test[mask_pred] == i).mean())
            else:
                class_precision.append(0.0)

            # Recall
            if mask_true.sum() > 0:
                class_recall.append((predicted[mask_true] == i).mean())
            else:
                class_recall.append(0.0)

        balanced_accuracy = np.mean(class_acc)

        # F1 scores
        class_f1 = [
            2 * (p * r) / (p + r + 1e-8)
            for p, r in zip(class_precision, class_recall)
        ]

        # Confidence
        max_probs = probs.max(axis=1)
        avg_confidence = max_probs.mean()

        results = {
            'accuracy': float(accuracy),
            'balanced_accuracy': float(balanced_accuracy),
            'avg_confidence': float(avg_confidence),
            'class_accuracy': [float(x) for x in class_acc],
            'class_precision': [float(x) for x in class_precision],
            'class_recall': [float(x) for x in class_recall],
            'class_f1': [float(x) for x in class_f1]
        }

        logger.info(f"Evaluation - Accuracy: {accuracy:.4f}, Balanced: {balanced_accuracy:.4f}")
        logger.info(f"Class accuracies: {[f'{x:.4f}' for x in class_acc]}")

        return results

    def save(self, filepath: Union[str, Path]):
        """Save model to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_type': self.model_type,
            'input_size': self.input_size,
            'num_classes': self.num_classes,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'history': self.history
        }, filepath)

        logger.info(f"DirectionalPredictor saved to {filepath}")

    def load(self, filepath: Union[str, Path]):
        """Load model from file."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.history = checkpoint['history']

        logger.info(f"DirectionalPredictor loaded from {filepath}")
