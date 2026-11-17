"""
Order execution and event handlers for weighted simulation.

Handles stop loss evaluation, signal application, rebalance orders, and force closure.
"""

from __future__ import annotations

import itertools
from typing import Any

import pandas as pd

from src.backtesting.backtest_config import BacktestConfig
from src.backtesting.backtesting_engine import Trade
from src.backtesting.position_book import PositionBook
from src.backtesting.simulator_accounting import record_trade
from src.backtesting.simulator_models import compute_fill
from src.backtesting.simulator_rebalance import execute_rebalance
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


__all__ = [
    "check_stops",
    "apply_signals",
    "force_close_all",
    "execute_rebalance",
]
