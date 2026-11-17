"""
Weighted portfolio simulator with capital constraints, rebalancing, and risk controls.

Implements WeightedPortfolioSimulator supporting multi-asset weight allocation,
average-cost lot tracking, execution lag, stop-loss / take-profit, and cash reserves.
"""

from __future__ import annotations

import itertools
from typing import Any

import pandas as pd

from src.backtesting.allocation_interface import Allocator
from src.backtesting.backtest_config import (
    AllocationMethod,
    BacktestConfig,
    RebalanceFrequency,
)
from src.backtesting.backtesting_engine import Trade
from src.backtesting.position_book import PositionBook
from src.backtesting.simulator_accounting import (
    assemble_simulation_output,
    record_snapshots,
    validate_simulator_inputs,
)
from src.backtesting.simulator_execution import (
    apply_signals,
    check_stops,
    execute_rebalance,
    force_close_all,
)
from src.backtesting.simulator_models import SimulationOutput
from src.utils.logger import get_logger

logger = get_logger(__name__)


class WeightedPortfolioSimulator:
    """Realistic capital-constrained backtesting simulator supporting dynamic weighting."""

    def __init__(
        self,
        config: BacktestConfig,
        allocator: Allocator | None = None,
    ) -> None:
        """
        Initialize WeightedPortfolioSimulator.

        Args:
            config: Backtesting configuration parameters.
            allocator: Portfolio weight allocator. None only for FIXED_NOTIONAL.

        Raises:
            ValueError: If allocator is None for weight-based allocation methods.
        """
        self.config = config.resolved()
        self.config.validate()

        if (
            self.config.allocation_method != AllocationMethod.FIXED_NOTIONAL
            and allocator is None
        ):
            raise ValueError(
                f"Allocator cannot be None for allocation_method={self.config.allocation_method.value}"
            )
        self.allocator = allocator
        self._trade_id_counter = itertools.count(1)

    def run(
        self, signals_df: pd.DataFrame, prices_df: pd.DataFrame
    ) -> SimulationOutput:
        """Execute backtest simulation across the given signals and prices."""
        validate_simulator_inputs(signals_df, prices_df)

        signals = signals_df.copy()
        prices = prices_df.copy()
        signals["date"] = pd.to_datetime(signals["date"])
        prices["date"] = pd.to_datetime(prices["date"])

        sig_pairs = set(zip(signals["symbol"], signals["date"]))
        px_pairs = set(zip(prices["symbol"], prices["date"]))
        if not sig_pairs.intersection(px_pairs):
            raise ValueError(
                "No overlapping (symbol, date) rows between signals_df and prices_df."
            )

        calendar = sorted(prices["date"].unique())
        date_to_idx = {d: i for i, d in enumerate(calendar)}

        price_lookup: dict[tuple[str, pd.Timestamp], float] = {}
        for row in prices.itertuples(index=False):
            sym, dt = row.symbol, row.date
            ref_px = (
                getattr(row, "adj_close", None)
                if getattr(row, "adj_close", None) is not None
                and not pd.isna(getattr(row, "adj_close", None))
                else row.close
            )
            if ref_px <= 0:
                raise ValueError(
                    f"Execution price for symbol '{sym}' on {dt} must be positive, got {ref_px}"
                )
            price_lookup[(sym, dt)] = float(ref_px)

        effective_signals = self._build_effective_signals(
            signals, calendar, date_to_idx
        )
        return self._run_simulation_loop(
            calendar, price_lookup, effective_signals, prices
        )

    def _build_effective_signals(
        self,
        signals: pd.DataFrame,
        calendar: list[pd.Timestamp],
        date_to_idx: dict[pd.Timestamp, int],
    ) -> dict[pd.Timestamp, list[dict[str, Any]]]:
        """Align signals with calendar considering execution lag."""
        lag = self.config.execution_lag or 0
        effective_signals: dict[pd.Timestamp, list[dict[str, Any]]] = {
            d: [] for d in calendar
        }

        for row in signals.to_dict(orient="records"):
            sig_date = row["date"]
            if sig_date not in date_to_idx:
                continue
            idx = date_to_idx[sig_date]
            exec_idx = idx + lag
            if exec_idx < len(calendar):
                exec_date = calendar[exec_idx]
                row_copy = dict(row)
                row_copy["decision_date"] = sig_date
                effective_signals[exec_date].append(row_copy)
            else:
                logger.info(
                    "Signal on %s for %s lagged beyond calendar; dropped.",
                    sig_date,
                    row["symbol"],
                )
        return effective_signals

    def _should_rebalance(
        self,
        date: pd.Timestamp,
        prev_date: pd.Timestamp | None,
        cur_targets: tuple[set[str], set[str]],
        prev_targets: tuple[set[str], set[str]],
    ) -> bool:
        """Check rebalance frequency or target change condition."""
        if prev_date is None:
            return True
        freq = self.config.rebalance_frequency
        if freq == RebalanceFrequency.DAILY:
            return True
        if freq == RebalanceFrequency.WEEKLY:
            return date.isocalendar()[:2] != prev_date.isocalendar()[:2]
        if freq == RebalanceFrequency.MONTHLY:
            return (date.year, date.month) != (prev_date.year, prev_date.month)
        return cur_targets != prev_targets

    def _run_simulation_loop(
        self,
        calendar: list[pd.Timestamp],
        price_lookup: dict[tuple[str, pd.Timestamp], float],
        effective_signals: dict[pd.Timestamp, list[dict[str, Any]]],
        prices: pd.DataFrame,
    ) -> SimulationOutput:
        """Execute daily state simulation across all calendar dates."""
        book = PositionBook()
        cash = self.config.initial_capital
        last_prices: dict[str, float] = {}
        symbol_states: dict[str, int] = {}
        stopped_out: set[str] = set()

        trades: list[Trade] = []
        trade_details: list[dict[str, Any]] = []
        history_snapshots: list[dict[str, Any]] = []
        daily_details: list[dict[str, Any]] = []
        positions_history: list[dict[str, Any]] = []

        running_max = cash
        prev_equity = cash
        prev_date: pd.Timestamp | None = None
        prev_targets: tuple[set[str], set[str]] = (set(), set())

        for current_date in calendar:
            day_prices = {
                sym: px for (sym, dt), px in price_lookup.items() if dt == current_date
            }
            last_prices.update(day_prices)

            cash = check_stops(
                self.config,
                self._trade_id_counter,
                book,
                day_prices,
                current_date,
                stopped_out,
                symbol_states,
                trades,
                trade_details,
                cash,
            )

            day_sigs = effective_signals.get(current_date, [])
            apply_signals(day_sigs, symbol_states, stopped_out, self.config.allow_short)

            cur_longs = {s for s, state in symbol_states.items() if state == 1}
            cur_shorts = {s for s, state in symbol_states.items() if state == -1}
            cur_targets = (cur_longs, cur_shorts)

            should_rebal = self._should_rebalance(
                current_date, prev_date, cur_targets, prev_targets
            )

            traded_notional = 0.0
            if should_rebal:
                traded_notional, cash = execute_rebalance(
                    self.config,
                    self._trade_id_counter,
                    self.allocator,
                    current_date,
                    book,
                    last_prices,
                    symbol_states,
                    day_sigs,
                    prices,
                    trades,
                    trade_details,
                    cash,
                )
                prev_targets = cur_targets

            if current_date == calendar[-1] and self.config.force_close_at_end:
                logger.warning(
                    "Force-closing %d open positions at final simulation date %s",
                    len(book.symbols()),
                    current_date,
                )
                cash = force_close_all(
                    self.config,
                    self._trade_id_counter,
                    book,
                    last_prices,
                    current_date,
                    trades,
                    trade_details,
                    cash,
                )

            assert cash >= -1e-6, f"Cash went negative: {cash}"
            cash = max(0.0, cash)

            equity = (
                cash
                + book.market_value(last_prices, side=1)
                - book.market_value(last_prices, side=-1)
            )
            record_snapshots(
                current_date,
                prev_date,
                prev_equity,
                equity,
                running_max,
                book,
                last_prices,
                traded_notional,
                trades,
                positions_history,
                history_snapshots,
                daily_details,
                self.config.initial_capital,
            )
            prev_equity = equity
            running_max = max(running_max, equity)
            prev_date = current_date

        return assemble_simulation_output(
            trades,
            trade_details,
            history_snapshots,
            daily_details,
            positions_history,
            self.config.initial_capital,
        )
