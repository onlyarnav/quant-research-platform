"""
Backtesting engine and portfolio simulation layer for quantitative trading.

Provides deterministic, cost-aware trade simulation from model signals,
capital-constrained portfolio simulation across multi-asset universes,
and position management.
"""

from __future__ import annotations

from src.backtesting.backtesting_engine import BacktestingEngine, BacktestResult, Trade
from src.backtesting.portfolio_simulator import (
    PortfolioSimulationResult,
    PortfolioSimulator,
    PortfolioSnapshot,
)
from src.backtesting.position_manager import OpenPosition, PositionManager

__all__ = [
    "BacktestResult",
    "BacktestingEngine",
    "OpenPosition",
    "PortfolioSimulationResult",
    "PortfolioSimulator",
    "PortfolioSnapshot",
    "PositionManager",
    "Trade",
]
