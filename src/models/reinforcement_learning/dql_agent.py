"""
Deep Q-Learning (DQL) agent for trading.
Memory-optimized implementation with adaptive buffer sizing.
"""

import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from loguru import logger

from .trading_env import TradingEnvironment


class DQNetwork(nn.Module):
    """Deep Q-Network architecture."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_sizes: List[int] = [256, 256, 128]
    ):
        super().__init__()

        layers = []
        prev_size = state_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, action_size))

        self.network = nn.Sequential(*layers)

        # Dueling DQN: separate value and advantage streams
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_sizes[-1], hidden_sizes[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_sizes[-1] // 2, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_sizes[-1], hidden_sizes[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_sizes[-1] // 2, action_size)
        )

        # Feature extractor (without final layer)
        self.feature_extractor = nn.Sequential(*layers[:-1])

    def forward(self, x: torch.Tensor, dueling: bool = True) -> torch.Tensor:
        if dueling:
            features = self.feature_extractor(x)
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            # Combine value and advantage
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
            return q_values
        else:
            return self.network(x)


class EfficientReplayBuffer:
    """
    Memory-efficient replay buffer using pre-allocated numpy arrays.
    Uses float32 for states and avoids Python list overhead.
    """

    def __init__(self, capacity: int, state_size: int):
        self.capacity = capacity
        self.state_size = state_size
        self.position = 0
        self.size = 0

        # Pre-allocate arrays
        self.states = np.zeros((capacity, state_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_size), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        logger.info(f"EfficientReplayBuffer initialized: capacity={capacity}, "
                   f"state_size={state_size}, "
                   f"memory={capacity * state_size * 4 * 2 / 1024 / 1024:.1f} MB")

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = float(done)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.choice(self.size, batch_size, replace=False)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

    def __len__(self) -> int:
        return self.size


class EfficientPrioritizedReplayBuffer:
    """
    Memory-efficient prioritized experience replay buffer.
    Uses pre-allocated numpy arrays to minimize memory fragmentation.
    """

    def __init__(self, capacity: int, state_size: int, alpha: float = 0.6):
        self.capacity = capacity
        self.state_size = state_size
        self.alpha = alpha
        self.position = 0
        self.size = 0

        # Pre-allocate arrays
        self.states = np.zeros((capacity, state_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_size), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.priorities = np.ones(capacity, dtype=np.float32)

        # Calculate memory usage
        memory_mb = (capacity * state_size * 4 * 2 + capacity * 4 * 3) / 1024 / 1024
        logger.info(f"EfficientPrioritizedReplayBuffer initialized: capacity={capacity}, "
                   f"state_size={state_size}, memory={memory_mb:.1f} MB")

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        max_priority = self.priorities[:self.size].max() if self.size > 0 else 1.0

        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = float(done)
        self.priorities[self.position] = max_priority

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float = 0.4) -> Optional[Tuple]:
        if self.size < batch_size:
            return None

        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(self.size, batch_size, p=probabilities, replace=False)

        weights = (self.size * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            indices,
            weights.astype(np.float32)
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        self.priorities[indices] = priorities

    def __len__(self) -> int:
        return self.size


def calculate_optimal_buffer_size(state_size: int, max_memory_mb: int = 256) -> int:
    """
    Calculate optimal buffer size based on state size and available memory.

    Args:
        state_size: Size of each state vector
        max_memory_mb: Maximum memory to use in MB (reduced to 256MB)

    Returns:
        Optimal buffer capacity
    """
    # Memory per transition: state + next_state + action + reward + done + priority
    # = 2 * state_size * 4 (float32) + 8 (int64) + 4 (float32) + 4 (float32) + 4 (float32)
    bytes_per_transition = 2 * state_size * 4 + 8 + 4 + 4 + 4

    max_memory_bytes = max_memory_mb * 1024 * 1024
    optimal_capacity = max_memory_bytes // bytes_per_transition

    # More conservative bounds for large state sizes
    if state_size > 4000:
        optimal_capacity = min(optimal_capacity, 5000)  # Cap at 5K for large states
    else:
        optimal_capacity = min(optimal_capacity, 10000)  # Cap at 10K otherwise

    optimal_capacity = max(1000, optimal_capacity)  # Minimum 1000

    logger.info(f"Calculated buffer size: {optimal_capacity} "
               f"(state_size={state_size}, max_memory={max_memory_mb}MB)")

    return int(optimal_capacity)


class DQLTradingAgent:
    """
    Deep Q-Learning agent for trading with self-learning capabilities.
    Memory-optimized implementation.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int = 3,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 20000,
        buffer_size: Optional[int] = None,
        max_buffer_memory_mb: int = 128,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        use_double_dqn: bool = True,
        use_dueling: bool = True,
        use_per: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize DQL trading agent.

        Args:
            state_size: Size of state space
            action_size: Size of action space
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon_start: Initial epsilon for exploration (1.0)
            epsilon_end: Final epsilon (0.05)
            epsilon_decay_steps: Steps for linear epsilon decay (20000)
            buffer_size: Replay buffer size (auto-calculated if None)
            max_buffer_memory_mb: Maximum memory for replay buffer in MB (128MB)
            batch_size: Training batch size
            target_update_freq: Target network update frequency
            use_double_dqn: Use Double DQN
            use_dueling: Use Dueling DQN
            use_per: Use Prioritized Experience Replay
            device: Device to use
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_double_dqn = use_double_dqn
        self.use_dueling = use_dueling

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Networks
        self.policy_net = DQNetwork(state_size, action_size).to(self.device)
        self.target_net = DQNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10000, gamma=0.9)

        # Calculate optimal buffer size if not provided
        if buffer_size is None:
            buffer_size = calculate_optimal_buffer_size(state_size, max_buffer_memory_mb)

        # Memory-efficient replay buffer
        if use_per:
            self.replay_buffer = EfficientPrioritizedReplayBuffer(buffer_size, state_size)
        else:
            self.replay_buffer = EfficientReplayBuffer(buffer_size, state_size)
        self.use_per = use_per

        # Training tracking
        self.steps = 0
        self.training_losses = []

        logger.info(f"DQLTradingAgent initialized on {self.device}")
        logger.info(f"State size: {state_size}, Buffer size: {buffer_size}")

    def select_action(
        self,
        state: np.ndarray,
        training: bool = True
    ) -> Tuple[int, float]:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state
            training: Whether in training mode

        Returns:
            Tuple of (action, q_value)
        """
        if training and random.random() < self.epsilon:
            action = random.randrange(self.action_size)
            return action, 0.0

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor, dueling=self.use_dueling)
            action = q_values.argmax(1).item()
            confidence = float(q_values[0, action].cpu())

        return action, confidence

    def predict_single(self, state: np.ndarray) -> float:
        """
        Predict continuous value for ensemble integration.

        Converts DQL Q-values to continuous prediction compatible with
        regression models in ensemble.

        Args:
            state: Input state array (can be 2D or 3D)

        Returns:
            Float prediction in range ~-0.01 to +0.01 (buy to sell signal)
        """
        # Reshape state to match expected input
        if len(state.shape) == 3:
            # (1, seq_len, features) -> flatten to (1, seq_len * features)
            batch_size = state.shape[0]
            state = state.reshape(batch_size, -1)
        elif len(state.shape) == 2:
            # (seq_len, features) -> (1, seq_len * features)
            state = state.reshape(1, -1)

        # Ensure correct state size
        if state.shape[1] != self.state_size:
            # Take last state_size elements or pad with zeros
            if state.shape[1] > self.state_size:
                state = state[:, -self.state_size:]
            else:
                # Pad with zeros at the beginning
                padding = np.zeros((state.shape[0], self.state_size - state.shape[1]))
                state = np.concatenate([padding, state], axis=1)

        # Get Q-values from policy network
        state_tensor = torch.FloatTensor(state).to(self.device)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor, dueling=self.use_dueling)

        # Convert Q-values to continuous signal
        # Q-values shape: [batch_size, action_size]
        # Actions: [0=hold, 1=buy, 2=sell]
        q_np = q_values.cpu().numpy()[0]

        # Extract buy and sell Q-values
        buy_q = q_np[1] if len(q_np) > 1 else 0
        sell_q = q_np[2] if len(q_np) > 2 else 0

        # Calculate directional signal: positive = buy, negative = sell
        # Normalize to -1 to 1 range
        signal = (buy_q - sell_q) / (abs(buy_q) + abs(sell_q) + 1e-8)

        # Scale to match other models' prediction range (~0.003 for 0.3% return)
        # DQL signal is -1 to 1, scale to approximately -0.01 to 0.01
        prediction = float(signal * 0.01)

        return prediction

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Store transition in replay buffer."""
        self.replay_buffer.push(
            state.flatten().astype(np.float32),
            action,
            reward,
            next_state.flatten().astype(np.float32),
            done
        )

    def train_step(self) -> float:
        """Perform one training step."""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        # Sample from replay buffer
        if self.use_per:
            sample = self.replay_buffer.sample(self.batch_size)
            if sample is None:
                return 0.0
            states, actions, rewards, next_states, dones, indices, weights = sample
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
            weights = torch.ones(self.batch_size).to(self.device)

        # Convert to tensors (data is already numpy arrays from efficient buffer)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        current_q = self.policy_net(states, dueling=self.use_dueling).gather(1, actions.unsqueeze(1))

        # Target Q values
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: use policy net to select action, target net to evaluate
                next_actions = self.policy_net(next_states, dueling=self.use_dueling).argmax(1, keepdim=True)
                next_q = self.target_net(next_states, dueling=self.use_dueling).gather(1, next_actions)
            else:
                next_q = self.target_net(next_states, dueling=self.use_dueling).max(1, keepdim=True)[0]

            target_q = rewards.unsqueeze(1) + self.gamma * next_q * (1 - dones.unsqueeze(1))

        # Compute loss
        td_errors = torch.abs(current_q - target_q).detach().cpu().numpy()
        loss = (weights.unsqueeze(1) * nn.functional.smooth_l1_loss(current_q, target_q, reduction='none')).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()
        self.scheduler.step()

        # Update priorities for PER
        if self.use_per:
            self.replay_buffer.update_priorities(indices, td_errors.flatten() + 1e-6)

        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Linear epsilon decay
        if self.steps < self.epsilon_decay_steps:
            self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (self.steps / self.epsilon_decay_steps)
        else:
            self.epsilon = self.epsilon_end

        return loss.item()

    def train(
        self,
        env: TradingEnvironment,
        episodes: int = 1000,
        max_steps: int = 1000,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the agent.

        Args:
            env: Trading environment
            episodes: Number of episodes
            max_steps: Max steps per episode
            verbose: Print progress

        Returns:
            Training metrics
        """
        episode_rewards = []
        episode_losses = []

        for episode in range(episodes):
            state, _ = env.reset()
            episode_reward = 0
            losses = []

            for step in range(max_steps):
                action, _ = self.select_action(state, training=True)
                next_state, reward, terminated, truncated, info = env.step(action)

                self.store_transition(state, action, reward, next_state, terminated or truncated)

                loss = self.train_step()
                if loss > 0:
                    losses.append(loss)

                episode_reward += reward
                state = next_state

                if terminated or truncated:
                    break

            episode_rewards.append(episode_reward)
            episode_losses.append(np.mean(losses) if losses else 0)

            # Periodic garbage collection to prevent memory fragmentation
            if (episode + 1) % 10 == 0:
                gc.collect()

            if verbose and (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                logger.info(f"Episode {episode + 1}/{episodes} - Avg Reward: {avg_reward:.4f} - Epsilon: {self.epsilon:.4f}")

        return {
            'episode_rewards': episode_rewards,
            'episode_losses': episode_losses
        }

    def evaluate(
        self,
        env: TradingEnvironment,
        n_episodes: int = 10
    ) -> Dict[str, float]:
        """Evaluate agent performance."""
        total_rewards = []
        metrics_list = []

        for _ in range(n_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False

            while not done:
                action, _ = self.select_action(state, training=False)
                state, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated

            total_rewards.append(episode_reward)
            metrics_list.append(env.get_performance_metrics())

        return {
            'mean_reward': float(np.mean(total_rewards)),
            'mean_return': float(np.mean([m['total_return'] for m in metrics_list])),
            'mean_win_rate': float(np.mean([m['win_rate'] for m in metrics_list])),
            'mean_sharpe': float(np.mean([m['sharpe_ratio'] for m in metrics_list]))
        }

    def save(self, filepath: Union[str, Path]):
        """Save agent to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'config': {
                'state_size': self.state_size,
                'action_size': self.action_size,
                'gamma': self.gamma
            }
        }, filepath)
        logger.info(f"DQL agent saved to {filepath}")

    def load(self, filepath: Union[str, Path]):
        """Load agent from file."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        logger.info(f"DQL agent loaded from {filepath}")
