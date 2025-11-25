"""
CNN (Convolutional Neural Network) for chart pattern recognition.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from loguru import logger


class CNNNetwork(nn.Module):
    """
    CNN architecture for recognizing chart patterns.
    """

    def __init__(
        self,
        input_channels: int = 1,
        image_size: Tuple[int, int] = (64, 64),
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()

        self.image_size = image_size

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)
        self.dropout2d = nn.Dropout2d(dropout / 2)

        # Calculate flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, *image_size)
            dummy = self._forward_conv(dummy)
            flat_size = dummy.view(1, -1).size(1)

        # Fully connected layers
        self.fc1 = nn.Linear(flat_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

        # Global Average Pooling alternative
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def _forward_conv(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout2d(x)

        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2d(x)

        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout2d(x)

        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        return x

    def forward(self, x, return_features: bool = False):
        # Convolutional layers
        conv_out = self._forward_conv(x)

        # Flatten
        features = conv_out.view(conv_out.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(features))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        output = self.fc3(x)

        if return_features:
            return output, features
        return output


class CNNPatternRecognizer:
    """
    CNN-based pattern recognizer for chart analysis.
    """

    def __init__(
        self,
        input_channels: int = 1,
        image_size: Tuple[int, int] = (64, 64),
        num_classes: int = 2,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        device: Optional[str] = None
    ):
        self.input_channels = input_channels
        self.image_size = image_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.learning_rate = learning_rate

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = CNNNetwork(
            input_channels=input_channels,
            image_size=image_size,
            num_classes=num_classes,
            dropout=dropout
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=30, gamma=0.1
        )

        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        self.pattern_names = ['bearish', 'bullish']  # Default for binary

        logger.info(f"CNNPatternRecognizer initialized on {self.device}")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        early_stopping: int = 15,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.LongTensor(y_train).to(self.device)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if X_val is not None:
            X_val_t = torch.FloatTensor(X_val).to(self.device)
            y_val_t = torch.LongTensor(y_val).to(self.device)

        best_val_acc = 0
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            train_losses, train_correct, train_total = [], 0, 0

            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                train_losses.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()

            self.scheduler.step()

            train_acc = train_correct / train_total
            self.history['train_loss'].append(np.mean(train_losses))
            self.history['train_acc'].append(train_acc)

            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_t)
                    val_loss = self.criterion(val_outputs, y_val_t).item()
                    _, val_predicted = torch.max(val_outputs.data, 1)
                    val_acc = (val_predicted == y_val_t).sum().item() / len(y_val_t)

                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    self._save_best_weights()
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    self._load_best_weights()
                    break

                if verbose and (epoch + 1) % 5 == 0:
                    logger.info(f"Epoch {epoch + 1}/{epochs} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")

        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_t)
            probabilities = F.softmax(outputs, dim=1)
            return probabilities.cpu().numpy()

    def predict_class(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict(X)
        return np.argmax(probabilities, axis=1)

    def predict_pattern(self, X: np.ndarray) -> List[Dict]:
        """Predict patterns with names and confidence."""
        probabilities = self.predict(X)
        predictions = []

        for prob in probabilities:
            class_idx = np.argmax(prob)
            predictions.append({
                'pattern': self.pattern_names[class_idx] if class_idx < len(self.pattern_names) else f'class_{class_idx}',
                'confidence': float(prob[class_idx]),
                'probabilities': {
                    self.pattern_names[i] if i < len(self.pattern_names) else f'class_{i}': float(p)
                    for i, p in enumerate(prob)
                }
            })

        return predictions

    def extract_features(self, X: np.ndarray) -> np.ndarray:
        """Extract CNN features for ensemble methods."""
        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            _, features = self.model(X_t, return_features=True)
            return features.cpu().numpy()

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        predictions = self.predict_class(X_test)
        accuracy = np.mean(predictions == y_test)

        # Per-class metrics
        per_class_acc = {}
        for i in range(self.num_classes):
            mask = y_test == i
            if np.sum(mask) > 0:
                class_acc = np.mean(predictions[mask] == y_test[mask])
                class_name = self.pattern_names[i] if i < len(self.pattern_names) else f'class_{i}'
                per_class_acc[class_name] = float(class_acc)

        return {
            'accuracy': float(accuracy),
            'per_class_accuracy': per_class_acc
        }

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
            'history': self.history,
            'pattern_names': self.pattern_names,
            'config': {
                'input_channels': self.input_channels,
                'image_size': self.image_size,
                'num_classes': self.num_classes,
                'dropout': self.dropout,
                'learning_rate': self.learning_rate
            }
        }, filepath)
        logger.info(f"CNN model saved to {filepath}")

    def load(self, filepath: Union[str, Path]):
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint['history']
        self.pattern_names = checkpoint.get('pattern_names', self.pattern_names)
        logger.info(f"CNN model loaded from {filepath}")
