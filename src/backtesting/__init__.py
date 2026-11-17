"""
Backtesting engine and portfolio simulation layer for quantitative trading.

Provides deterministic, cost-aware trade simulation from model signals,
capital-constrained portfolio simulation across multi-asset universes,
and position management.
"""

from __future__ import annotations

from src.backtesting.allocation_interface import (
    AllocationContext,
    AllocationResult,
    Allocator,
)
from src.backtesting.backtest_config import (
    AllocationMethod,
    BacktestConfig,
    RebalanceFrequency,
)
from src.backtesting.backtest_store import BacktestStore, RunRecord
from src.backtesting.backtesting_engine import BacktestingEngine, BacktestResult, Trade
from src.backtesting.portfolio_simulator import (
    PortfolioSimulationResult,
    PortfolioSimulator,
    PortfolioSnapshot,
)
from src.backtesting.position_book import BookPosition, PositionBook
from src.backtesting.position_manager import OpenPosition, PositionManager
from src.backtesting.price_loader import PriceLoader
from src.backtesting.sensitivity import run_sensitivity
from src.backtesting.signal_loader import SignalLoader
from src.backtesting.weighted_simulator import (
    SimulationOutput,
    WeightedPortfolioSimulator,
)

__all__ = [
    "AllocationContext",
    "AllocationMethod",
    "AllocationResult",
    "Allocator",
    "BacktestConfig",
    "BacktestResult",
    "BacktestStore",
    "BacktestingEngine",
    "BookPosition",
    "OpenPosition",
    "PortfolioSimulationResult",
    "PortfolioSimulator",
    "PortfolioSnapshot",
    "PositionBook",
    "PositionManager",
    "PriceLoader",
    "RebalanceFrequency",
    "RunRecord",
    "SignalLoader",
    "SimulationOutput",
    "Trade",
    "WeightedPortfolioSimulator",
    "run_sensitivity",
]
