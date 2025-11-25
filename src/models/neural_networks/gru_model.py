"""
GRU (Gated Recurrent Unit) model for price prediction.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from loguru import logger


class GRUNetwork(nn.Module):
    """
    GRU neural network architecture for time series prediction.
    More efficient than LSTM with comparable performance.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        output_size: int = 1,
        task: str = 'regression'
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.task = task

        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        gru_output_size = hidden_size * 2 if bidirectional else hidden_size

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(gru_output_size, gru_output_size // 2),
            nn.Tanh(),
            nn.Linear(gru_output_size // 2, 1),
            nn.Softmax(dim=1)
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(gru_output_size)

        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(gru_output_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_size // 4, output_size)
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # GRU forward
        gru_out, _ = self.gru(x)

        # Layer normalization
        gru_out = self.layer_norm(gru_out)

        # Attention
        attention_weights = self.attention(gru_out)
        context = torch.sum(attention_weights * gru_out, dim=1)

        # Output
        output = self.fc(context)

        if self.task == 'classification':
            output = torch.softmax(output, dim=-1)

        if return_attention:
            return output, attention_weights.squeeze(-1)
        return output


class GRUPredictor:
    """
    GRU-based price predictor with self-learning capabilities.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        task: str = 'regression',
        device: Optional[str] = None
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.task = task

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        output_size = 1 if task == 'regression' else 3
        self.model = GRUNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            output_size=output_size,
            task=task
        ).to(self.device)

        if task == 'regression':
            self.criterion = nn.HuberLoss()  # More robust to outliers
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

        self.history: Dict[str, List[float]] = {
            'train_loss': [], 'val_loss': []
        }

        logger.info(f"GRUPredictor initialized on {self.device}")

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
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        if self.task == 'classification':
            y_train_t = y_train_t.long()

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if X_val is not None:
            X_val_t = torch.FloatTensor(X_val).to(self.device)
            y_val_t = torch.FloatTensor(y_val).to(self.device)
            if self.task == 'classification':
                y_val_t = y_val_t.long()

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
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

            self.scheduler.step()
            avg_train_loss = np.mean(train_losses)
            self.history['train_loss'].append(avg_train_loss)

            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_t)
                    if self.task == 'regression':
                        val_loss = self.criterion(val_outputs.squeeze(), y_val_t).item()
                    else:
                        val_loss = self.criterion(val_outputs, y_val_t).item()

                self.history['val_loss'].append(val_loss)

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
                    logger.info(f"Epoch {epoch + 1}/{epochs} - Train: {avg_train_loss:.6f} - Val: {val_loss:.6f}")

        return self.history

    def predict(self, X: np.ndarray, return_attention: bool = False):
        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            if return_attention:
                outputs, attention = self.model(X_t, return_attention=True)
                return outputs.cpu().numpy(), attention.cpu().numpy()
            else:
                outputs = self.model(X_t)
                return outputs.cpu().numpy()

    def predict_with_uncertainty(self, X: np.ndarray, n_samples: int = 50):
        """Monte Carlo Dropout for uncertainty estimation."""
        self.model.train()
        X_t = torch.FloatTensor(X).to(self.device)

        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                outputs = self.model(X_t)
                predictions.append(outputs.cpu().numpy())

        predictions = np.array(predictions)
        self.model.eval()
        return np.mean(predictions, axis=0), np.std(predictions, axis=0)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        predictions = self.predict(X_test)

        if self.task == 'regression':
            mse = np.mean((predictions.squeeze() - y_test) ** 2)
            mae = np.mean(np.abs(predictions.squeeze() - y_test))

            pred_direction = np.sign(np.diff(predictions.squeeze()))
            actual_direction = np.sign(np.diff(y_test))
            direction_acc = np.mean(pred_direction == actual_direction) if len(y_test) > 1 else 0

            return {
                'mse': float(mse),
                'rmse': float(np.sqrt(mse)),
                'mae': float(mae),
                'direction_accuracy': float(direction_acc)
            }
        else:
            pred_classes = np.argmax(predictions, axis=1)
            accuracy = np.mean(pred_classes == y_test)
            return {'accuracy': float(accuracy)}

    def _save_best_weights(self):
        self._best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

    def _load_best_weights(self):
        if hasattr(self, '_best_weights'):
            self.model.load_state_dict(self._best_weights)

    def save(self, filepath: Union[str, Path]):
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
        logger.info(f"GRU model saved to {filepath}")

    def load(self, filepath: Union[str, Path]):
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        logger.info(f"GRU model loaded from {filepath}")
