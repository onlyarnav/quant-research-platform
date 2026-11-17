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
from src.backtesting.simulator_accounting import record_trade
from src.backtesting.simulator_models import compute_fill
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


def execute_rebalance(
    config: BacktestConfig,
    counter: itertools.count[int],
    allocator: Allocator | None,
    date: pd.Timestamp,
    book: PositionBook,
    last_prices: dict[str, float],
    states: dict[str, int],
    day_sigs: list[dict[str, Any]],
    prices_df: pd.DataFrame,
    trades: list[Trade],
    trade_details: list[dict[str, Any]],
    cash: float,
) -> tuple[float, float]:
    """Rebalance holdings, updating cash and returning (traded_notional, cash)."""
    target_longs = [s for s, state in states.items() if state == 1]
    target_shorts = [s for s, state in states.items() if state == -1]

    for side, targets in [(1, target_longs), (-1, target_shorts)]:
        for sym in list(book.symbols(side=side)):
            if sym not in targets and sym in last_prices:
                p = book.get(sym)
                if p:
                    lot = book.reduce(sym, p.quantity)
                    fill = compute_fill(
                        last_prices[sym],
                        lot.quantity,
                        side,
                        False,
                        config.transaction_cost,
                        config.slippage,
                    )
                    if side == 1:
                        cash += fill.executed_price * lot.quantity - fill.fee
                    else:
                        cash -= fill.executed_price * lot.quantity + fill.fee
                    record_trade(
                        counter,
                        sym,
                        side,
                        last_prices[sym],
                        date,
                        lot,
                        "signal",
                        trades,
                        trade_details,
                        fill,
                    )

    equity = (
        cash
        + book.market_value(last_prices, side=1)
        - book.market_value(last_prices, side=-1)
    )
    raw_targets = compute_target_quantities(
        config,
        allocator,
        target_longs,
        target_shorts,
        last_prices,
        equity,
        date,
        day_sigs,
        prices_df,
        book,
    )

    traded_notional = 0.0
    for sym, (t_side, t_qty) in raw_targets.items():
        pos = book.get(sym)
        if pos and pos.quantity > t_qty and sym in last_prices:
            px = last_prices[sym]
            diff_qty = pos.quantity - t_qty
            if diff_qty * px >= max(
                config.min_trade_value, config.rebalance_tolerance * equity
            ):
                lot = book.reduce(sym, diff_qty)
                fill = compute_fill(
                    px,
                    diff_qty,
                    pos.side,
                    False,
                    config.transaction_cost,
                    config.slippage,
                )
                if pos.side == 1:
                    cash += fill.executed_price * diff_qty - fill.fee
                else:
                    cash -= fill.executed_price * diff_qty + fill.fee
                record_trade(
                    counter,
                    sym,
                    pos.side,
                    px,
                    date,
                    lot,
                    "rebalance_reduce",
                    trades,
                    trade_details,
                    fill,
                )
                traded_notional += diff_qty * px

    sorted_targets = sort_and_truncate_targets(
        raw_targets, day_sigs, config.max_open_positions
    )
    for sym, t_side, t_qty in sorted_targets:
        if sym not in last_prices:
            continue
        pos = book.get(sym)
        cur_qty = pos.quantity if pos else 0.0
        if t_qty > cur_qty:
            px = last_prices[sym]
            needed_qty = t_qty - cur_qty
            if cur_qty > 0 and needed_qty * px < max(
                config.min_trade_value, config.rebalance_tolerance * equity
            ):
                continue

            if t_side == 1:
                cost_per_unit = (
                    px * (1.0 + config.slippage) * (1.0 + config.transaction_cost)
                )
                avail_cash = max(0.0, cash - (config.cash_reserve or 0.0) * equity)
                if avail_cash <= 0 or cost_per_unit <= 0:
                    logger.debug("Skipping buy for %s due to insufficient cash.", sym)
                    continue
                max_affordable = avail_cash / cost_per_unit
                fill_qty = min(needed_qty, max_affordable)
                if fill_qty <= 1e-9:
                    continue
            else:
                fill_qty = needed_qty

            fill = compute_fill(
                px, fill_qty, t_side, True, config.transaction_cost, config.slippage
            )
            if t_side == 1:
                cash -= fill.executed_price * fill_qty + fill.fee
            else:
                cash += fill.executed_price * fill_qty - fill.fee

            book.add(
                sym,
                t_side,
                date,
                fill_qty,
                fill.executed_price,
                fill.fee,
                fill.slippage_cost,
            )
            traded_notional += fill_qty * px

    return traded_notional, cash
