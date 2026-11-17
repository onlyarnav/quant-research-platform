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


@dataclass(slots=True)
class ExecutionFill:
    """Calculated fill price and leg transaction costs."""

    reference_price: float
    executed_price: float
    fee: float
    slippage_cost: float


def compute_fill(
    price: float,
    quantity: float,
    side: int,
    is_entry: bool,
    transaction_cost: float,
    slippage: float,
) -> ExecutionFill:
    """Compute execution price and leg transaction costs."""
    is_buy = (side == 1 and is_entry) or (side == -1 and not is_entry)
    if is_buy:
        exec_price = price * (1.0 + slippage)
        slip_cost = (exec_price - price) * quantity
    else:
        exec_price = price * (1.0 - slippage)
        slip_cost = (price - exec_price) * quantity

    fee = transaction_cost * exec_price * quantity
    return ExecutionFill(
        reference_price=price,
        executed_price=exec_price,
        fee=fee,
        slippage_cost=slip_cost,
    )
