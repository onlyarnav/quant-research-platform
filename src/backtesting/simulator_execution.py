"""
Order execution and event handlers for weighted simulation.

Handles stop loss evaluation, signal application, rebalance orders, and force closure.
"""

from __future__ import annotations

import itertools
from typing import Any

import pandas as pd

from src.backtesting.allocation_interface import Allocator
from src.backtesting.backtest_config import BacktestConfig
from src.backtesting.backtesting_engine import Trade
from src.backtesting.position_book import PositionBook
from src.backtesting.simulator_accounting import compute_fill
from src.backtesting.simulator_rebalance import (
    compute_target_quantities,
    record_trade,
    sort_and_truncate_targets,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def check_stops(
    config: BacktestConfig,
    counter: itertools.count[int],
    book: PositionBook,
    day_prices: dict[str, float],
    date: pd.Timestamp,
    stopped_out: set[str],
    symbol_states: dict[str, int],
    trades: list[Trade],
    trade_details: list[dict[str, Any]],
    cash: float,
) -> float:
    """Evaluate stop-loss and take-profit thresholds for open positions."""
    sl, tp = config.stop_loss_pct, config.take_profit_pct
    if sl is None and tp is None:
        return cash

    for sym in list(book.symbols()):
        if sym not in day_prices:
            continue
        pos = book.get(sym)
        if not pos:
            continue
        px = day_prices[sym]
        ret = (
            (px / pos.avg_entry_price - 1.0)
            if pos.side == 1
            else (1.0 - px / pos.avg_entry_price)
        )
        reason = (
            "stop_loss"
            if sl and ret <= -sl
            else ("take_profit" if tp and ret >= tp else None)
        )
        if reason:
            lot = book.reduce(sym, pos.quantity)
            fill = compute_fill(
                px,
                lot.quantity,
                pos.side,
                False,
                config.transaction_cost,
                config.slippage,
            )
            if pos.side == 1:
                cash += fill.executed_price * lot.quantity - fill.fee
            else:
                cash -= fill.executed_price * lot.quantity + fill.fee
            record_trade(
                counter,
                sym,
                pos.side,
                px,
                date,
                lot,
                reason,
                trades,
                trade_details,
                fill,
            )
            stopped_out.add(sym)
            symbol_states[sym] = 0
    return cash


def apply_signals(
    signals: list[dict[str, Any]],
    symbol_states: dict[str, int],
    stopped_out: set[str],
    allow_short: bool,
) -> None:
    """Update symbol states from effective daily signals."""
    for sig in signals:
        sym, val = sig["symbol"], int(sig["signal"])
        if sym in stopped_out:
            if val == 1 or (val == -1 and allow_short):
                stopped_out.remove(sym)
                symbol_states[sym] = val
        else:
            if val == 1:
                symbol_states[sym] = 1
            elif val == -1:
                symbol_states[sym] = -1 if allow_short else 0


def force_close_all(
    config: BacktestConfig,
    counter: itertools.count[int],
    book: PositionBook,
    prices: dict[str, float],
    date: pd.Timestamp,
    trades: list[Trade],
    trade_details: list[dict[str, Any]],
    cash: float,
) -> float:
    """Force-close all open positions at final simulation date."""
    for sym in list(book.symbols()):
        pos = book.get(sym)
        if pos and sym in prices:
            lot = book.reduce(sym, pos.quantity)
            fill = compute_fill(
                prices[sym],
                lot.quantity,
                pos.side,
                False,
                config.transaction_cost,
                config.slippage,
            )
            if pos.side == 1:
                cash += fill.executed_price * lot.quantity - fill.fee
            else:
                cash -= fill.executed_price * lot.quantity + fill.fee
            record_trade(
                counter,
                sym,
                pos.side,
                prices[sym],
                date,
                lot,
                "forced_close_end",
                trades,
                trade_details,
                fill,
            )
    return cash


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
