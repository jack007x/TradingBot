"""
AI Trading Bot - Self-Learning Trading System
=============================================

A cutting-edge artificial intelligence trading system featuring:
- Deep Learning (LSTM, GRU, CNN)
- Reinforcement Learning (PPO, DQL)
- Natural Language Processing for sentiment analysis
- Genetic Algorithms for optimization
- Explainable AI for transparency
- Self-learning capabilities

Author: AI Trading Bot Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "AI Trading Bot Team"

from .utils.logger import setup_logger
from .utils.config import Config

__all__ = [
    "setup_logger",
    "Config",
    "__version__",
    "__author__",
]
