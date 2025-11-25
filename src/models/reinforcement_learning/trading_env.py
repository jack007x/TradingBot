"""
Custom Trading Environment for Reinforcement Learning.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger


class TradingEnvironment(gym.Env):
    """
    Custom Gym environment for trading with RL agents.
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(
        self,
        df: np.ndarray,
        feature_columns: List[str],
        initial_balance: float = 10000.0,
        max_position_size: float = 1.0,
        transaction_cost: float = 0.001,
        reward_scaling: float = 1.0,
        window_size: int = 60,
        max_steps: Optional[int] = None
    ):
        """
        Initialize trading environment.

        Args:
            df: OHLCV data with features as numpy array
            feature_columns: Names of feature columns
            initial_balance: Starting balance
            max_position_size: Maximum position size (0-1)
            transaction_cost: Transaction cost as fraction
            reward_scaling: Scaling factor for rewards
            window_size: Observation window size
            max_steps: Maximum steps per episode
        """
        super().__init__()

        self.data = df
        self.feature_columns = feature_columns
        self.initial_balance = initial_balance
        self.max_position_size = max_position_size
        self.transaction_cost = transaction_cost
        self.reward_scaling = reward_scaling
        self.window_size = window_size
        self.max_steps = max_steps or (len(df) - window_size - 1)

        # Feature indices
        self.close_idx = feature_columns.index('close') if 'close' in feature_columns else 0
        self.num_features = len(feature_columns)

        # Action space: 0=hold, 1=buy, 2=sell (discrete)
        # Or continuous: position size from -1 to 1
        self.action_space = spaces.Discrete(3)

        # Observation space: [market_features, position, balance, unrealized_pnl]
        obs_shape = (window_size, self.num_features + 3)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=np.float32
        )

        # State variables
        self.reset()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        self.balance = self.initial_balance
        self.position = 0.0  # -1 to 1
        self.position_price = 0.0
        self.current_step = self.window_size

        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.max_balance = self.initial_balance
        self.trade_history = []

        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        # Market features
        start_idx = self.current_step - self.window_size
        end_idx = self.current_step
        market_obs = self.data[start_idx:end_idx].copy()

        # Normalize market features
        market_obs = (market_obs - market_obs.mean(axis=0)) / (market_obs.std(axis=0) + 1e-8)

        # Account features (repeated for each timestep)
        account_features = np.array([
            self.position,
            self.balance / self.initial_balance,
            self._get_unrealized_pnl() / self.initial_balance
        ])
        account_obs = np.tile(account_features, (self.window_size, 1))

        # Combine
        obs = np.concatenate([market_obs, account_obs], axis=1)
        return obs.astype(np.float32)

    def _get_current_price(self) -> float:
        """Get current close price."""
        return float(self.data[self.current_step, self.close_idx])

    def _get_unrealized_pnl(self) -> float:
        """Calculate unrealized PnL."""
        if self.position == 0:
            return 0.0

        current_price = self._get_current_price()
        price_change = (current_price - self.position_price) / self.position_price

        if self.position > 0:  # Long
            return self.balance * abs(self.position) * price_change
        else:  # Short
            return self.balance * abs(self.position) * (-price_change)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment with equity-based reward.

        Args:
            action: 0=hold, 1=buy, 2=sell

        Returns:
            observation, reward, terminated, truncated, info
        """
        prev_equity = self.balance + self._get_unrealized_pnl()
        current_price = self._get_current_price()

        # Execute action
        trade_cost = 0.0
        closed_trade_pnl = 0.0

        if action == 1 and self.position <= 0:  # Buy
            if self.position < 0:  # Close short first
                closed_trade_pnl = self._close_position(current_price)
                trade_cost += self.transaction_cost

            # Open long
            self._open_position(current_price, self.max_position_size)
            trade_cost += self.transaction_cost

        elif action == 2 and self.position >= 0:  # Sell
            if self.position > 0:  # Close long first
                closed_trade_pnl = self._close_position(current_price)
                trade_cost += self.transaction_cost

            # Open short
            self._open_position(current_price, -self.max_position_size)
            trade_cost += self.transaction_cost

        # Move to next step
        self.current_step += 1

        # Calculate new equity
        new_equity = self.balance + self._get_unrealized_pnl()

        # ===== EQUITY-BASED REWARD =====
        # 1. Equity change (main signal)
        equity_change = new_equity - prev_equity

        # 2. Transaction costs
        fee_penalty = trade_cost * self.initial_balance

        # 3. Risk penalty (discourage over-leveraging)
        risk_penalty = 0.0
        if abs(self.position) > 0:
            # CRITICAL FIX: Use abs() for unrealized_pnl to ensure penalty is always positive
            risk_penalty = 0.01 * (abs(self.position) * abs(self._get_unrealized_pnl())) / self.initial_balance

        # Final reward
        reward = (equity_change / self.initial_balance) * 1000  # Scale to reasonable range
        # CRITICAL FIX: Reduce fee penalty multiplier from 10 to 2 (was discouraging ALL trading)
        reward -= fee_penalty * 2  # Moderate fee penalty
        reward -= risk_penalty  # Penalize risk

        # Update max equity
        self.max_balance = max(self.max_balance, new_equity)

        # Check termination
        terminated = False
        truncated = False

        # Terminate if significant loss
        if new_equity < self.initial_balance * 0.3:  # 70% drawdown
            terminated = True
            reward -= 100  # Large penalty

        # Truncate if max steps reached
        if self.current_step >= len(self.data) - 1 or \
           self.current_step - self.window_size >= self.max_steps:
            truncated = True
            # Close position at end
            if self.position != 0:
                final_pnl = self._close_position(self._get_current_price())
                reward += (final_pnl / self.initial_balance) * 1000

        # Info dict
        info = {
            'equity': new_equity,
            'equity_change': equity_change,
            'position': self.position,
            'balance': self.balance,
            'total_trades': self.total_trades,
            'win_rate': self.winning_trades / max(1, self.total_trades),
            'total_pnl': self.total_pnl,
            'max_drawdown': (self.max_balance - new_equity) / self.max_balance,
            'closed_trade_pnl': closed_trade_pnl,
            'trade_cost': fee_penalty,
            'risk_penalty': risk_penalty
        }

        return self._get_observation(), reward, terminated, truncated, info

    def _open_position(self, price: float, size: float) -> None:
        """Open a new position."""
        cost = self.balance * abs(size) * self.transaction_cost
        self.balance -= cost
        self.position = size
        self.position_price = price

    def _close_position(self, price: float) -> float:
        """Close current position and return PnL."""
        if self.position == 0:
            return 0.0

        # Calculate PnL
        price_change = (price - self.position_price) / self.position_price
        if self.position > 0:
            pnl = self.balance * abs(self.position) * price_change
        else:
            pnl = self.balance * abs(self.position) * (-price_change)

        # Apply transaction cost
        cost = self.balance * abs(self.position) * self.transaction_cost
        net_pnl = pnl - cost

        # Update state
        self.balance += net_pnl
        self.total_pnl += net_pnl
        self.total_trades += 1
        if net_pnl > 0:
            self.winning_trades += 1

        # Record trade
        self.trade_history.append({
            'entry_price': self.position_price,
            'exit_price': price,
            'position': self.position,
            'pnl': net_pnl
        })

        # Reset position
        self.position = 0.0
        self.position_price = 0.0

        return net_pnl

    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render the environment."""
        if mode == 'human':
            print(f"Step: {self.current_step} | "
                  f"Balance: ${self.balance:.2f} | "
                  f"Position: {self.position:.2f} | "
                  f"Portfolio: ${self.balance + self._get_unrealized_pnl():.2f}")
        return None

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get comprehensive performance metrics."""
        portfolio_value = self.balance + self._get_unrealized_pnl()

        return {
            'total_return': (portfolio_value - self.initial_balance) / self.initial_balance,
            'total_trades': self.total_trades,
            'win_rate': self.winning_trades / max(1, self.total_trades),
            'profit_factor': self._calculate_profit_factor(),
            'max_drawdown': (self.max_balance - portfolio_value) / self.max_balance,
            'sharpe_ratio': self._calculate_sharpe_ratio()
        }

    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor from trade history."""
        if not self.trade_history:
            return 0.0

        gross_profit = sum(t['pnl'] for t in self.trade_history if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in self.trade_history if t['pnl'] < 0))

        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from trade history."""
        if len(self.trade_history) < 2:
            return 0.0

        returns = [t['pnl'] / self.initial_balance for t in self.trade_history]
        if np.std(returns) == 0:
            return 0.0

        return np.mean(returns) / np.std(returns) * np.sqrt(252)
