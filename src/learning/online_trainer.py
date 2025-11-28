"""
Online Trainer for Continuous Model Learning.
Fine-tunes models based on live trading outcomes with safety features.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
import copy

from .experience_buffer import ExperienceBuffer


class OnlineTrainer:
    """
    Online trainer with safety features to prevent catastrophic forgetting.

    Features:
    - Elastic Weight Consolidation (EWC) to preserve important weights
    - Validation-based rollback if performance degrades
    - Small learning rate for stability
    - Limited update steps
    """

    def __init__(
        self,
        learning_rate: float = 1e-5,
        max_epochs: int = 3,
        batch_size: int = 16,
        ewc_lambda: float = 1000.0,
        validation_threshold: float = 0.45,
        device: str = 'cpu'
    ):
        """
        Initialize online trainer.

        Args:
            learning_rate: Very small LR for fine-tuning (default 1e-5)
            max_epochs: Max epochs per update (default 3)
            batch_size: Batch size for updates (default 16)
            ewc_lambda: EWC regularization strength (default 1000)
            validation_threshold: Min directional accuracy to accept update (default 0.45)
            device: 'cpu' or 'cuda'
        """
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.ewc_lambda = ewc_lambda
        self.validation_threshold = validation_threshold
        self.device = device

        # EWC: Store Fisher information matrix and optimal weights
        self.fisher_dict: Dict[str, torch.Tensor] = {}
        self.optimal_weights: Dict[str, torch.Tensor] = {}

        logger.info("OnlineTrainer initialized")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Max epochs: {max_epochs}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  EWC lambda: {ewc_lambda}")
        logger.info(f"  Validation threshold: {validation_threshold}")

    def compute_fisher_information(
        self,
        model: nn.Module,
        experience_buffer: ExperienceBuffer,
        num_samples: int = 100
    ):
        """
        Compute Fisher Information Matrix for EWC.

        This identifies which weights are important and should be
        preserved during fine-tuning.

        Args:
            model: Neural network model
            experience_buffer: Buffer with training data
            num_samples: Number of samples for Fisher estimation
        """
        logger.info("📊 Computing Fisher Information Matrix...")

        model.eval()
        fisher = {}

        # Initialize Fisher dict
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param.data)

        # Get samples from experience buffer
        experiences = experience_buffer.get_recent_experiences(n=num_samples)

        if len(experiences) == 0:
            logger.warning("⚠️  No experiences available for Fisher computation")
            return

        # Compute Fisher approximation
        for exp in experiences:
            model.zero_grad()

            # Forward pass
            state_tensor = torch.FloatTensor(exp.state).unsqueeze(0).to(self.device)
            target_tensor = torch.FloatTensor([exp.actual_return]).to(self.device)

            output = model(state_tensor)
            loss = nn.MSELoss()(output, target_tensor)

            # Backward pass
            loss.backward()

            # Accumulate squared gradients (Fisher approximation)
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data ** 2

        # Average
        for name in fisher:
            fisher[name] /= len(experiences)

        self.fisher_dict = fisher

        # Store current weights as optimal
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.optimal_weights[name] = param.data.clone()

        logger.info(f"✅ Fisher Information computed from {len(experiences)} samples")

    def ewc_loss(self, model: nn.Module) -> torch.Tensor:
        """
        Compute EWC regularization loss.

        This penalizes changes to important weights.

        Args:
            model: Neural network model

        Returns:
            EWC loss term
        """
        if len(self.fisher_dict) == 0:
            return torch.tensor(0.0).to(self.device)

        loss = 0.0
        for name, param in model.named_parameters():
            if name in self.fisher_dict and param.requires_grad:
                fisher = self.fisher_dict[name]
                optimal = self.optimal_weights[name]
                loss += (fisher * (param - optimal) ** 2).sum()

        return self.ewc_lambda * loss

    def fine_tune_model(
        self,
        model: nn.Module,
        experience_buffer: ExperienceBuffer,
        model_name: str,
        criterion: Optional[nn.Module] = None
    ) -> Dict:
        """
        Fine-tune a model on recent experiences.

        Safety features:
        1. Small learning rate
        2. Limited epochs
        3. EWC regularization
        4. Validation check with rollback

        Args:
            model: Model to fine-tune
            experience_buffer: Buffer with training data
            model_name: Name of model
            criterion: Loss function (default MSE)

        Returns:
            Dict with training results
        """
        logger.info(f"🎓 Fine-tuning {model_name}...")

        # Save original state for rollback
        original_state = copy.deepcopy(model.state_dict())

        # Get training data
        experiences = experience_buffer.get_recent_experiences(
            n=200,
            model_name=model_name,
            min_confidence=0.3
        )

        if len(experiences) < 20:
            logger.warning(f"⚠️  Insufficient experiences for {model_name}: {len(experiences)} < 20")
            return {
                'status': 'skipped',
                'reason': 'insufficient_data',
                'samples': len(experiences)
            }

        # Split train/validation
        split_idx = int(len(experiences) * 0.8)
        train_exp = experiences[:split_idx]
        val_exp = experiences[split_idx:]

        logger.info(f"  Train samples: {len(train_exp)}")
        logger.info(f"  Val samples: {len(val_exp)}")

        # Prepare data
        train_states = np.array([e.state for e in train_exp])
        train_targets = np.array([e.actual_return for e in train_exp])
        val_states = np.array([e.state for e in val_exp])
        val_targets = np.array([e.actual_return for e in val_exp])

        # Setup training
        if criterion is None:
            criterion = nn.MSELoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        model.train()

        # Compute Fisher if not already done
        if len(self.fisher_dict) == 0:
            self.compute_fisher_information(model, experience_buffer)

        # Training loop
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []

        for epoch in range(self.max_epochs):
            epoch_loss = 0.0
            num_batches = 0

            # Mini-batch training
            indices = np.random.permutation(len(train_states))
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                batch_states = train_states[batch_indices]
                batch_targets = train_targets[batch_indices]

                # Convert to tensors
                states_tensor = torch.FloatTensor(batch_states).to(self.device)
                targets_tensor = torch.FloatTensor(batch_targets).unsqueeze(1).to(self.device)

                # Forward pass
                optimizer.zero_grad()
                outputs = model(states_tensor)

                # Task loss
                task_loss = criterion(outputs, targets_tensor)

                # EWC regularization
                ewc_penalty = self.ewc_loss(model)

                # Total loss
                total_loss = task_loss + ewc_penalty

                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += task_loss.item()
                num_batches += 1

            avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0
            train_losses.append(avg_train_loss)

            # Validation
            model.eval()
            with torch.no_grad():
                val_states_tensor = torch.FloatTensor(val_states).to(self.device)
                val_targets_tensor = torch.FloatTensor(val_targets).unsqueeze(1).to(self.device)

                val_outputs = model(val_states_tensor)
                val_loss = criterion(val_outputs, val_targets_tensor).item()
                val_losses.append(val_loss)

                # Directional accuracy
                val_predictions = val_outputs.cpu().numpy().flatten()
                val_direction_pred = (val_predictions > 0).astype(int)
                val_direction_true = (val_targets > 0).astype(int)
                val_accuracy = (val_direction_pred == val_direction_true).mean()

            model.train()

            logger.info(f"  Epoch {epoch+1}/{self.max_epochs}: "
                       f"Train Loss={avg_train_loss:.6f}, "
                       f"Val Loss={val_loss:.6f}, "
                       f"Val Acc={val_accuracy:.2%}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss

        # Final validation check
        model.eval()
        with torch.no_grad():
            val_states_tensor = torch.FloatTensor(val_states).to(self.device)
            val_targets_tensor = torch.FloatTensor(val_targets).unsqueeze(1).to(self.device)

            final_outputs = model(val_states_tensor)
            final_predictions = final_outputs.cpu().numpy().flatten()

            final_direction_pred = (final_predictions > 0).astype(int)
            final_direction_true = (val_targets > 0).astype(int)
            final_accuracy = (final_direction_pred == final_direction_true).mean()

        logger.info(f"  Final validation accuracy: {final_accuracy:.2%}")

        # Rollback if performance degraded
        if final_accuracy < self.validation_threshold:
            logger.warning(f"⚠️  Validation accuracy {final_accuracy:.2%} < threshold {self.validation_threshold:.2%}")
            logger.warning(f"🔄 Rolling back to original weights")
            model.load_state_dict(original_state)

            return {
                'status': 'rolled_back',
                'reason': 'low_validation_accuracy',
                'final_accuracy': final_accuracy,
                'threshold': self.validation_threshold,
                'samples': len(experiences)
            }

        # Success - update Fisher for next iteration
        self.compute_fisher_information(model, experience_buffer)

        logger.info(f"✅ Fine-tuning successful for {model_name}")

        return {
            'status': 'success',
            'final_accuracy': final_accuracy,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'samples': len(experiences)
        }


class AdaptiveLearningScheduler:
    """
    Decides when to trigger model fine-tuning.

    Triggers based on:
    - Time since last update
    - Number of new experiences
    - Performance degradation detection
    """

    def __init__(
        self,
        min_time_between_updates: int = 3600,  # 1 hour
        min_experiences_for_update: int = 50,
        performance_check_window: int = 20,
        performance_degradation_threshold: float = 0.10  # 10% drop
    ):
        """
        Initialize learning scheduler.

        Args:
            min_time_between_updates: Minimum seconds between updates
            min_experiences_for_update: Minimum new experiences needed
            performance_check_window: Recent trades to check for degradation
            performance_degradation_threshold: % drop to trigger update
        """
        self.min_time_between_updates = min_time_between_updates
        self.min_experiences_for_update = min_experiences_for_update
        self.performance_check_window = performance_check_window
        self.performance_degradation_threshold = performance_degradation_threshold

        self.last_update_time: Dict[str, datetime] = {}
        self.last_experience_count: Dict[str, int] = {}

        logger.info("AdaptiveLearningScheduler initialized")
        logger.info(f"  Min time between updates: {min_time_between_updates}s")
        logger.info(f"  Min experiences for update: {min_experiences_for_update}")
        logger.info(f"  Performance check window: {performance_check_window}")

    def should_update(
        self,
        model_name: str,
        experience_buffer: ExperienceBuffer
    ) -> Tuple[bool, str]:
        """
        Check if model should be updated.

        Args:
            model_name: Name of model
            experience_buffer: Experience buffer

        Returns:
            Tuple of (should_update: bool, reason: str)
        """
        now = datetime.utcnow()

        # Check 1: Minimum time elapsed
        if model_name in self.last_update_time:
            time_since_update = (now - self.last_update_time[model_name]).total_seconds()
            if time_since_update < self.min_time_between_updates:
                return False, f"Too soon (last update {time_since_update:.0f}s ago)"

        # Check 2: Sufficient new experiences
        current_count = len(experience_buffer.completed_experiences)
        if model_name in self.last_experience_count:
            new_experiences = current_count - self.last_experience_count[model_name]
            if new_experiences < self.min_experiences_for_update:
                return False, f"Insufficient experiences ({new_experiences}/{self.min_experiences_for_update})"

        # Check 3: Performance degradation
        perf = experience_buffer.get_model_performance(model_name)
        if perf['total_trades'] >= self.performance_check_window:
            # Get recent accuracy
            recent_experiences = experience_buffer.get_recent_experiences(
                n=self.performance_check_window,
                model_name=model_name
            )
            if len(recent_experiences) >= self.performance_check_window:
                recent_accuracy = sum(1 for e in recent_experiences if e.directional_correct) / len(recent_experiences)
                overall_accuracy = perf['accuracy']

                degradation = overall_accuracy - recent_accuracy
                if degradation > self.performance_degradation_threshold:
                    logger.warning(f"⚠️  Performance degradation detected for {model_name}!")
                    logger.warning(f"   Overall: {overall_accuracy:.2%} → Recent: {recent_accuracy:.2%}")
                    return True, f"Performance degradation ({degradation:.2%} drop)"

        # All checks passed - update recommended
        return True, "Sufficient experiences and time elapsed"

    def mark_update_completed(self, model_name: str, experience_buffer: ExperienceBuffer):
        """
        Mark that an update was completed.

        Args:
            model_name: Name of model
            experience_buffer: Experience buffer
        """
        self.last_update_time[model_name] = datetime.utcnow()
        self.last_experience_count[model_name] = len(experience_buffer.completed_experiences)

        logger.info(f"✅ Update marked complete for {model_name}")
