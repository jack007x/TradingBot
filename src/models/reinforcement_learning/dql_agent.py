"""
Deep Q-Learning (DQL) agent for trading.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
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


class ReplayBuffer:
    """Experience replay buffer."""

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self) -> int:
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer."""

    def __init__(self, capacity: int = 100000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        max_priority = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple:
        if len(self.buffer) < batch_size:
            return None

        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)

        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            indices,
            weights
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self) -> int:
        return len(self.buffer)


class DQLTradingAgent:
    """
    Deep Q-Learning agent for trading with self-learning capabilities.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int = 3,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100000,
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
            epsilon_start: Initial epsilon for exploration
            epsilon_end: Final epsilon
            epsilon_decay: Epsilon decay rate
            buffer_size: Replay buffer size
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
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
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

        # Replay buffer
        if use_per:
            self.replay_buffer = PrioritizedReplayBuffer(buffer_size)
        else:
            self.replay_buffer = ReplayBuffer(buffer_size)
        self.use_per = use_per

        # Training tracking
        self.steps = 0
        self.training_losses = []

        logger.info(f"DQLTradingAgent initialized on {self.device}")

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
            state.flatten(),
            action,
            reward,
            next_state.flatten(),
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

        # Convert to tensors
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

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

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

            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
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
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        logger.info(f"DQL agent loaded from {filepath}")
