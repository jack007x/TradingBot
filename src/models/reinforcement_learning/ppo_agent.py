"""
PPO (Proximal Policy Optimization) agent for trading.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from loguru import logger

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    STABLE_BASELINES_AVAILABLE = False
    logger.warning("stable-baselines3 not available, using custom PPO")

from .trading_env import TradingEnvironment


class TradingCallback(BaseCallback):
    """Custom callback for monitoring training progress."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        if self.locals.get('dones', [False])[0]:
            info = self.locals.get('infos', [{}])[0]
            self.episode_rewards.append(info.get('total_pnl', 0))
        return True


class PPOTradingAgent:
    """
    PPO-based trading agent with self-learning capabilities.
    """

    def __init__(
        self,
        env: Optional[TradingEnvironment] = None,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        device: str = 'auto'
    ):
        """
        Initialize PPO trading agent.

        Args:
            env: Trading environment
            learning_rate: Learning rate
            n_steps: Steps per update
            batch_size: Minibatch size
            n_epochs: Epochs per update
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_range: PPO clip range
            ent_coef: Entropy coefficient
            device: Device to use
        """
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.device = device

        self.env = env
        self.model = None
        self.training_history = []

        if env and STABLE_BASELINES_AVAILABLE:
            self._init_model()

        logger.info("PPOTradingAgent initialized")

    def _init_model(self):
        """Initialize PPO model."""
        vec_env = DummyVecEnv([lambda: self.env])

        self.model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_range,
            ent_coef=self.ent_coef,
            verbose=1,
            device=self.device,
            policy_kwargs={
                'net_arch': [dict(pi=[256, 256], vf=[256, 256])],
                'activation_fn': nn.ReLU
            }
        )

    def set_environment(self, env: TradingEnvironment):
        """Set or update the trading environment."""
        self.env = env
        self._init_model()

    def train(
        self,
        total_timesteps: int = 100000,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        callback: Optional[BaseCallback] = None
    ) -> Dict[str, List[float]]:
        """
        Train the PPO agent.

        Args:
            total_timesteps: Total training timesteps
            eval_freq: Evaluation frequency
            n_eval_episodes: Episodes per evaluation
            callback: Custom callback

        Returns:
            Training metrics
        """
        if self.model is None:
            raise ValueError("Model not initialized. Set environment first.")

        trading_callback = TradingCallback()
        callbacks = [trading_callback]
        if callback:
            callbacks.append(callback)

        logger.info(f"Starting PPO training for {total_timesteps} timesteps")

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )

        self.training_history = trading_callback.episode_rewards

        return {
            'episode_rewards': trading_callback.episode_rewards,
            'episode_lengths': trading_callback.episode_lengths
        }

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[int, float]:
        """
        Predict action for given observation.

        Args:
            observation: Current observation
            deterministic: Use deterministic policy

        Returns:
            Tuple of (action, confidence)
        """
        if self.model is None:
            raise ValueError("Model not initialized")

        action, _ = self.model.predict(observation, deterministic=deterministic)

        # Get action probabilities for confidence
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
        with torch.no_grad():
            distribution = self.model.policy.get_distribution(obs_tensor)
            probs = distribution.distribution.probs.numpy()[0]

        confidence = float(probs[action])
        return int(action), confidence

    def get_action_probabilities(self, observation: np.ndarray) -> np.ndarray:
        """Get probabilities for all actions."""
        if self.model is None:
            raise ValueError("Model not initialized")

        obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
        with torch.no_grad():
            distribution = self.model.policy.get_distribution(obs_tensor)
            return distribution.distribution.probs.numpy()[0]

    def evaluate(
        self,
        env: Optional[TradingEnvironment] = None,
        n_episodes: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate agent performance.

        Args:
            env: Environment to evaluate on
            n_episodes: Number of episodes

        Returns:
            Evaluation metrics
        """
        eval_env = env or self.env
        if eval_env is None:
            raise ValueError("No environment provided")

        total_rewards = []
        total_trades = []
        win_rates = []
        sharpe_ratios = []

        for _ in range(n_episodes):
            obs, _ = eval_env.reset()
            done = False
            episode_reward = 0

            while not done:
                action, _ = self.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                episode_reward += reward
                done = terminated or truncated

            total_rewards.append(episode_reward)
            metrics = eval_env.get_performance_metrics()
            total_trades.append(metrics['total_trades'])
            win_rates.append(metrics['win_rate'])
            sharpe_ratios.append(metrics['sharpe_ratio'])

        return {
            'mean_reward': float(np.mean(total_rewards)),
            'std_reward': float(np.std(total_rewards)),
            'mean_trades': float(np.mean(total_trades)),
            'mean_win_rate': float(np.mean(win_rates)),
            'mean_sharpe': float(np.mean(sharpe_ratios))
        }

    def self_learn(
        self,
        new_data: np.ndarray,
        feature_columns: List[str],
        fine_tune_steps: int = 10000
    ) -> Dict[str, float]:
        """
        Self-learning: adapt to new market data.

        Args:
            new_data: New market data
            feature_columns: Feature column names
            fine_tune_steps: Fine-tuning timesteps

        Returns:
            Performance metrics after adaptation
        """
        # Create new environment with updated data
        new_env = TradingEnvironment(
            df=new_data,
            feature_columns=feature_columns,
            initial_balance=self.env.initial_balance if self.env else 10000,
            transaction_cost=self.env.transaction_cost if self.env else 0.001
        )

        # Evaluate before fine-tuning
        pre_metrics = self.evaluate(new_env, n_episodes=5)

        # Fine-tune on new data with lower learning rate
        old_lr = self.learning_rate
        self.model.learning_rate = old_lr * 0.1

        self.model.set_env(DummyVecEnv([lambda: new_env]))
        self.model.learn(total_timesteps=fine_tune_steps, progress_bar=False)

        self.model.learning_rate = old_lr

        # Evaluate after fine-tuning
        post_metrics = self.evaluate(new_env, n_episodes=5)

        logger.info(f"Self-learning: Sharpe {pre_metrics['mean_sharpe']:.3f} -> {post_metrics['mean_sharpe']:.3f}")

        return {
            'pre_sharpe': pre_metrics['mean_sharpe'],
            'post_sharpe': post_metrics['mean_sharpe'],
            'improvement': post_metrics['mean_sharpe'] - pre_metrics['mean_sharpe']
        }

    def save(self, filepath: Union[str, Path]):
        """Save agent to file."""
        if self.model is None:
            raise ValueError("Model not initialized")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(filepath)
        logger.info(f"PPO agent saved to {filepath}")

    def load(self, filepath: Union[str, Path]):
        """Load agent from file."""
        filepath = Path(filepath)
        self.model = PPO.load(filepath)
        logger.info(f"PPO agent loaded from {filepath}")
