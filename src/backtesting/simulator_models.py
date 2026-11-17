"""
Data structures and results containers for weighted portfolio simulation.

Defines the SimulationOutput dataclass, TradeRecord helpers, and daily state.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.backtesting.backtesting_engine import Trade


@dataclass(slots=True)
class SimulationOutput:
    """Complete results, trade log, and exposure histories from a backtest run."""

    trades: list[Trade]
    trades_df: pd.DataFrame
    trade_details_df: pd.DataFrame
    portfolio_history_df: pd.DataFrame
    daily_details_df: pd.DataFrame
    positions_history_df: pd.DataFrame
    summary: dict[str, float]


@dataclass(slots=True)
class SimulationDailyState:
    """Mutable tracking state for the simulator's daily stepping loop."""

    cash: float
    running_max: float
    prev_equity: float
    prev_date: pd.Timestamp | None = None
    realized_pnl_cum: float = 0.0
    fees_cum: float = 0.0
    slippage_cum: float = 0.0
