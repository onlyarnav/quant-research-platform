"""
Rebalancing and order generation helpers for weighted simulation.

Handles allocator invocation, deterministic target sorting, priority truncation,
and order execution math.
"""

from __future__ import annotations

import itertools
from typing import Any

import pandas as pd

from src.backtesting.allocation_interface import (
    AllocationContext,
    Allocator,
)
from src.backtesting.backtest_config import (
    AllocationMethod,
    BacktestConfig,
)
from src.backtesting.backtesting_engine import Trade
from src.backtesting.position_book import PositionBook
from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_target_quantities(
    config: BacktestConfig,
    allocator: Allocator | None,
    longs: list[str],
    shorts: list[str],
    prices: dict[str, float],
    equity: float,
    date: pd.Timestamp,
    day_sigs: list[dict[str, Any]],
    prices_df: pd.DataFrame,
    book: PositionBook,
) -> dict[str, tuple[int, float]]:
    """Calculate target quantities per symbol based on configuration allocation method."""
    targets: dict[str, tuple[int, float]] = {}

    if config.allocation_method == AllocationMethod.FIXED_NOTIONAL:
        notional = config.initial_capital * config.max_position_size
        for s in longs:
            pos = book.get(s)
            if pos is not None and pos.side == 1:
                targets[s] = (1, pos.quantity)
            elif s in prices and prices[s] > 0:
                exec_px = prices[s] * (1.0 + config.slippage)
                targets[s] = (1, notional / exec_px)

        for s in shorts:
            pos = book.get(s)
            if pos is not None and pos.side == -1:
                targets[s] = (-1, pos.quantity)
            elif s in prices and prices[s] > 0:
                exec_px = prices[s] * (1.0 - config.slippage)
                targets[s] = (-1, notional / exec_px)
        return targets

    if allocator and longs:
        cand_rows = [s for s in day_sigs if s["symbol"] in longs]
        sig_df = (
            pd.DataFrame(cand_rows)
            if cand_rows
            else pd.DataFrame({"symbol": longs, "signal": [1] * len(longs)})
        )
        hist = prices_df[prices_df["date"] <= date]
        ctx = AllocationContext(
            as_of_date=date,
            signals=sig_df,
            prices_history=hist,
            current_weights={},
            portfolio_value=equity,
            side="long",
        )
        res = allocator.allocate(ctx)
        for s, w in res.weights.items():
            if s in prices and prices[s] > 0 and w > 0:
                targets[s] = (1, (w * equity) / prices[s])

    return targets


def sort_and_truncate_targets(
    targets: dict[str, tuple[int, float]],
    day_sigs: list[dict[str, Any]],
    max_open_positions: int | None,
) -> list[tuple[str, int, float]]:
    """Sort targets deterministically by confidence desc, rank asc, symbol asc."""
    sig_map = {s["symbol"]: s for s in day_sigs}
    items = []
    for sym, (side, qty) in targets.items():
        sig = sig_map.get(sym, {})
        conf = sig.get("confidence", None)
        conf_val = float(conf) if conf is not None and not pd.isna(conf) else -1e9
        rank = sig.get("rank", None)
        rank_val = float(rank) if rank is not None and not pd.isna(rank) else 1e9
        items.append((conf_val, rank_val, sym, side, qty))

    items.sort(key=lambda x: (-x[0], x[1], x[2]))
    if max_open_positions is not None:
        items = items[:max_open_positions]

    return [(sym, side, qty) for _, _, sym, side, qty in items]


def record_trade(
    counter: itertools.count[int],
    symbol: str,
    side: int,
    price: float,
    date: pd.Timestamp,
    lot: Any,
    reason: str,
    trades: list[Trade],
    trade_details: list[dict[str, Any]],
    fill: Any,
) -> None:
    """Create and append Trade and trade details records."""
    t_id = next(counter)
    total_fees = lot.entry_fee_share + fill.fee
    total_slippage = lot.entry_slippage_share + fill.slippage_cost
    gross = (
        (fill.executed_price - lot.avg_entry_price) * lot.quantity
        if side == 1
        else (lot.avg_entry_price - fill.executed_price) * lot.quantity
    )
    net_pnl = gross - total_fees
    cost_basis = lot.avg_entry_price * lot.quantity
    ret_pct = net_pnl / cost_basis if cost_basis > 0 else 0.0

    trades.append(
        Trade(
            trade_id=t_id,
            symbol=symbol,
            entry_date=lot.entry_date,
            exit_date=date,
            entry_price=lot.avg_entry_price,
            exit_price=fill.executed_price,
            position_size=lot.quantity,
            fees=total_fees,
            slippage_cost=total_slippage,
            pnl=net_pnl,
            return_pct=ret_pct,
        )
    )
    trade_details.append(
        {
            "trade_id": t_id,
            "side": "long" if side == 1 else "short",
            "exit_reason": reason,
            "gross_pnl": gross,
            "holding_days": (date - lot.entry_date).days,
        }
    )
