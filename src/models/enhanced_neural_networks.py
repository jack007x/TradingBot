"""
Enhanced Neural Network Training v2.0
=====================================

CRITICAL FIXES:
1. Anti-collapse regularization (forces prediction variance)
2. Noise injection during training
3. Directional focus loss (50% weight on direction)
4. Degeneration detection with warm restarts
5. Gradient clipping
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple
from loguru import logger


class AntiCollapseRegularizer(nn.Module):
    """Penalty for low prediction variance"""
    def __init__(self, target_std: float = 0.003, weight: float = 0.1):
        super().__init__()
        self.target_std = target_std
        self.weight = weight

    def forward(self, predictions: torch.Tensor) -> torch.Tensor:
        pred_std = predictions.std()
        if pred_std < self.target_std:
            return self.weight * (self.target_std - pred_std) ** 2
        return torch.tensor(0.0, device=predictions.device)


class DirectionalFocusLoss(nn.Module):
    """
    Loss function prioritizing directional accuracy.

    Weights: MSE=0.3, Direction=0.5, Variance=0.15, Correlation=0.05
    """
    def __init__(self):
        super().__init__()
        self.anti_collapse = AntiCollapseRegularizer()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        predictions = predictions.squeeze()
        targets = targets.squeeze()

        # MSE
        mse_loss = F.mse_loss(predictions, targets)

        # Directional
        direction_mismatch = (torch.sign(predictions) != torch.sign(targets)).float()
        direction_loss = (direction_mismatch * torch.abs(targets)).mean()

        # Variance
        variance_loss = self.anti_collapse(predictions)

        # Correlation
        if predictions.std() > 1e-8:
            pred_c = predictions - predictions.mean()
            tgt_c = targets - targets.mean()
            corr = (pred_c * tgt_c).mean() / (predictions.std() * targets.std() + 1e-10)
            corr_loss = 1 - corr
        else:
            corr_loss = torch.tensor(1.0, device=predictions.device)

        total = 0.3 * mse_loss + 0.5 * direction_loss + 0.15 * variance_loss + 0.05 * corr_loss

        return total, {
            'mse': mse_loss.item(),
            'direction': direction_loss.item(),
            'variance': variance_loss.item(),
            'correlation': corr_loss.item()
        }


class EnhancedLSTMNetwork(nn.Module):
    """LSTM with anti-collapse mechanisms"""

    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.3, noise_std: float = 0.01):
        super().__init__()
        self.noise_std = noise_std

        self.input_proj = nn.Linear(input_size, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.lstm_norm = nn.LayerNorm(hidden_size)

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softmax(dim=1)
        )

        self.output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor, add_noise: bool = False) -> torch.Tensor:
        x = self.input_norm(self.input_proj(x))

        if add_noise and self.training:
            x = x + torch.randn_like(x) * self.noise_std

        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_norm(lstm_out)

        attn = self.attention(lstm_out)
        context = torch.sum(attn * lstm_out, dim=1)

        return self.output(context)


class DegenerationDetector:
    """Detects training collapse early"""

    def __init__(self, variance_threshold: float = 1e-5, patience: int = 5):
        self.variance_threshold = variance_threshold
        self.patience = patience
        self.variance_history = []
        self.degen_count = 0

    def check(self, predictions: torch.Tensor, grad_norm: float) -> Tuple[bool, str]:
        var = predictions.var().item()
        self.variance_history.append(var)

        if var < self.variance_threshold:
            self.degen_count += 1
            if self.degen_count >= self.patience:
                return True, f"Variance {var:.2e} < threshold"
        else:
            self.degen_count = max(0, self.degen_count - 1)

        if grad_norm < 1e-8:
            return True, f"Gradient vanished: {grad_norm:.2e}"

        return False, ""


class EnhancedTrainer:
    """Training with anti-collapse mechanisms"""

    def __init__(self, model: nn.Module, learning_rate: float = 0.001,
                 grad_clip: float = 1.0, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.grad_clip = grad_clip

        self.criterion = DirectionalFocusLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)
        self.degen_detector = DegenerationDetector()

        self.history = {'train_loss': [], 'val_loss': [], 'train_dir_acc': [],
                       'val_dir_acc': [], 'train_correlation': [], 'val_correlation': [],
                       'pred_variance': []}

    def _apply_noise_perturbation(self):
        """Escape collapse by adding noise to weights"""
        with torch.no_grad():
            for param in self.model.parameters():
                if param.requires_grad:
                    param.add_(torch.randn_like(param) * 0.001)
        logger.info("Applied noise perturbation to escape collapse")

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, early_stopping=20):
        train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
                                  batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
                               batch_size=batch_size)

        best_val_loss = float('inf')
        best_state = None
        patience = 0

        logger.info("=" * 60)
        logger.info("TRAINING WITH ANTI-COLLAPSE MECHANISMS")
        logger.info("=" * 60)

        for epoch in range(epochs):
            # Train
            self.model.train()
            train_preds, train_targets = [], []
            train_loss = 0

            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(X, add_noise=True)
                loss, _ = self.criterion(pred, y)
                loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                train_loss += loss.item()
                train_preds.extend(pred.detach().cpu().numpy().flatten())
                train_targets.extend(y.cpu().numpy().flatten())

                # Check degeneration
                is_degen, reason = self.degen_detector.check(pred.detach(), grad_norm.item())
                if is_degen:
                    logger.warning(f"⚠️ Degeneration at epoch {epoch}: {reason}")
                    self._apply_noise_perturbation()

            # Validate
            self.model.eval()
            val_preds, val_targets = [], []
            val_loss = 0

            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    pred = self.model(X)
                    loss, _ = self.criterion(pred, y)
                    val_loss += loss.item()
                    val_preds.extend(pred.cpu().numpy().flatten())
                    val_targets.extend(y.cpu().numpy().flatten())

            # Metrics
            train_preds, train_targets = np.array(train_preds), np.array(train_targets)
            val_preds, val_targets = np.array(val_preds), np.array(val_targets)

            train_dir_acc = np.mean(np.sign(train_preds) == np.sign(train_targets))
            val_dir_acc = np.mean(np.sign(val_preds) == np.sign(val_targets))
            val_corr = np.corrcoef(val_preds, val_targets)[0, 1] if np.std(val_preds) > 1e-8 else 0
            val_var = np.var(val_preds)

            self.scheduler.step()

            # Record
            self.history['train_loss'].append(train_loss / len(train_loader))
            self.history['val_loss'].append(val_loss / len(val_loader))
            self.history['train_dir_acc'].append(train_dir_acc)
            self.history['val_dir_acc'].append(val_dir_acc)
            self.history['val_correlation'].append(val_corr)
            self.history['pred_variance'].append(val_var)

            # Log
            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch+1:3d} | Loss: {train_loss/len(train_loader):.6f}/{val_loss/len(val_loader):.6f} | "
                           f"DirAcc: {train_dir_acc:.2%}/{val_dir_acc:.2%} | Corr: {val_corr:.4f} | Var: {val_var:.2e}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience = 0
            else:
                patience += 1

            if val_var < 1e-6 and patience > 5:
                logger.warning("Variance collapse - warm restart")
                self._apply_noise_perturbation()
                patience = 0

            if patience >= early_stopping:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        if best_state:
            self.model.load_state_dict(best_state)

        final = {'loss': val_loss/len(val_loader), 'dir_acc': val_dir_acc,
                 'correlation': val_corr, 'pred_variance': val_var}

        logger.info("=" * 60)
        logger.info(f"TRAINING COMPLETE | DirAcc: {final['dir_acc']:.2%} | Corr: {final['correlation']:.4f}")
        logger.info("=" * 60)

        return {'history': self.history, 'final_metrics': final, 'best_val_loss': best_val_loss}


def train_enhanced_lstm(X_train, y_train, X_val, y_val, input_size, hidden_size=128,
                       num_layers=2, epochs=100, batch_size=32, learning_rate=0.001, device=None):
    """Convenience function to train enhanced LSTM"""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = EnhancedLSTMNetwork(input_size, hidden_size, num_layers)
    trainer = EnhancedTrainer(model, learning_rate, device=device)
    results = trainer.train(X_train, y_train, X_val, y_val, epochs, batch_size)

    return model, results
