"""
Execution fill math, cost accounting, and schema helpers for weighted simulation.

Provides pure calculation helpers for slippage, fees, PnL, snapshots, and schemas.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from src.backtesting.backtesting_engine import Trade
from src.backtesting.position_book import PositionBook
from src.backtesting.simulator_models import SimulationOutput
from src.utils.logger import get_logger

logger = get_logger(__name__)

CANONICAL_TRADE_COLS = [
    "trade_id",
    "symbol",
    "entry_date",
    "exit_date",
    "entry_price",
    "exit_price",
    "position_size",
    "fees",
    "slippage_cost",
    "pnl",
    "return_pct",
]

CANONICAL_HISTORY_COLS = [
    "date",
    "cash",
    "invested_capital",
    "portfolio_value",
    "daily_return",
    "cumulative_return",
    "drawdown",
]


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


def empty_trades_df() -> pd.DataFrame:
    """Return an empty trades DataFrame with canonical 11 columns and dtypes."""
    return pd.DataFrame(
        {
            "trade_id": pd.Series(dtype="int64"),
            "symbol": pd.Series(dtype="str"),
            "entry_date": pd.Series(dtype="datetime64[ns]"),
            "exit_date": pd.Series(dtype="datetime64[ns]"),
            "entry_price": pd.Series(dtype="float64"),
            "exit_price": pd.Series(dtype="float64"),
            "position_size": pd.Series(dtype="float64"),
            "fees": pd.Series(dtype="float64"),
            "slippage_cost": pd.Series(dtype="float64"),
            "pnl": pd.Series(dtype="float64"),
            "return_pct": pd.Series(dtype="float64"),
        }
    )


def empty_history_df() -> pd.DataFrame:
    """Return an empty history DataFrame with canonical 7 columns and dtypes."""
    return pd.DataFrame(
        {
            "date": pd.Series(dtype="datetime64[ns]"),
            "cash": pd.Series(dtype="float64"),
            "invested_capital": pd.Series(dtype="float64"),
            "portfolio_value": pd.Series(dtype="float64"),
            "daily_return": pd.Series(dtype="float64"),
            "cumulative_return": pd.Series(dtype="float64"),
            "drawdown": pd.Series(dtype="float64"),
        }
    )


def record_snapshots(
    current_date: pd.Timestamp,
    prev_date: pd.Timestamp | None,
    prev_equity: float,
    equity: float,
    running_max: float,
    book: PositionBook,
    last_prices: dict[str, float],
    traded_notional: float,
    trades: list[Trade],
    positions_history: list[dict[str, Any]],
    history_snapshots: list[dict[str, Any]],
    daily_details: list[dict[str, Any]],
    initial_capital: float,
) -> None:
    """Capture position details and portfolio equity snapshot for current date."""
    for sym in book.symbols():
        p = book.get(sym)
        if p:
            px = last_prices.get(sym, p.avg_entry_price)
            unreal = (
                (px - p.avg_entry_price) * p.quantity
                if p.side == 1
                else (p.avg_entry_price - px) * p.quantity
            )
            positions_history.append(
                {
                    "date": current_date,
                    "symbol": sym,
                    "side": p.side,
                    "quantity": p.quantity,
                    "price": px,
                    "market_value": p.quantity * px,
                    "weight": (p.quantity * px) / equity if equity > 0 else 0.0,
                    "unrealized_pnl": unreal,
                }
            )

    inv_cap = book.market_value(last_prices)
    daily_ret = (equity / prev_equity - 1.0) if prev_date is not None else 0.0
    cum_ret = (equity / initial_capital) - 1.0
    new_max = max(running_max, equity)
    dd = (equity - new_max) / new_max if new_max > 0 else 0.0

    history_snapshots.append(
        {
            "date": current_date,
            "cash": equity - inv_cap,
            "invested_capital": inv_cap,
            "portfolio_value": equity,
            "daily_return": daily_ret,
            "cumulative_return": cum_ret,
            "drawdown": dd,
        }
    )
    exp = book.exposures(last_prices, equity)
    daily_details.append(
        {
            "date": current_date,
            "gross_exposure": exp["gross"],
            "net_exposure": exp["net"],
            "long_exposure": exp["long"],
            "short_exposure": exp["short"],
            "n_positions": len(book.symbols()),
            "turnover": traded_notional / equity if equity > 0 else 0.0,
            "realized_pnl_cum": sum(t.pnl for t in trades),
            "unrealized_pnl": book.unrealized_pnl(last_prices),
            "fees_cum": sum(t.fees for t in trades),
            "slippage_cum": sum(t.slippage_cost for t in trades),
        }
    )


def assemble_simulation_output(
    trades: list[Trade],
    trade_details: list[dict[str, Any]],
    history_snapshots: list[dict[str, Any]],
    daily_details: list[dict[str, Any]],
    positions_history: list[dict[str, Any]],
    initial_capital: float,
) -> SimulationOutput:
    """Package simulation arrays into standard SimulationOutput DataFrames."""
    if not trades:
        logger.warning("Simulation finished with zero trades.")
        trades_df = empty_trades_df()
        details_df = pd.DataFrame(
            columns=["trade_id", "side", "exit_reason", "gross_pnl", "holding_days"]
        )
    else:
        trades_df = pd.DataFrame([asdict(t) for t in trades])[CANONICAL_TRADE_COLS]
        details_df = pd.DataFrame(trade_details)

    hist_df = (
        pd.DataFrame(history_snapshots)[CANONICAL_HISTORY_COLS]
        if history_snapshots
        else empty_history_df()
    )
    daily_df = pd.DataFrame(daily_details)
    pos_df = pd.DataFrame(positions_history)
    fin_eq = (
        hist_df["portfolio_value"].iloc[-1] if not hist_df.empty else initial_capital
    )
    avg_turn = (
        float(daily_df["turnover"].mean() * 252)
        if not daily_df.empty and "turnover" in daily_df
        else 0.0
    )

    summary = {
        "n_trades": float(len(trades)),
        "total_fees": float(trades_df["fees"].sum()) if not trades_df.empty else 0.0,
        "total_slippage": (
            float(trades_df["slippage_cost"].sum()) if not trades_df.empty else 0.0
        ),
        "avg_turnover_annualised": avg_turn,
        "final_equity": float(fin_eq),
    }
    return SimulationOutput(
        trades=trades,
        trades_df=trades_df,
        trade_details_df=details_df,
        portfolio_history_df=hist_df,
        daily_details_df=daily_df,
        positions_history_df=pos_df,
        summary=summary,
    )
