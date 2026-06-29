"""
Portfolio simulator for capital-constrained, multi-symbol backtesting.

Walks chronologically across all symbols using a PositionManager to
enforce shared capital constraints — a BUY signal is only acted on if
sufficient cash is available and no position is already open for that
symbol. Produces a realized trade log and a day-by-day portfolio
value history (cash, invested capital, returns, drawdown).

Note: open positions remaining at the end of the data are force-closed
for trade-log completeness. This incurs slippage and fees, so the
final realized cash position may differ slightly from the last daily
snapshot, which valued open positions at unadjusted mark-to-market
prices.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass

import pandas as pd

from config.settings import settings
from src.backtesting.backtesting_engine import Trade
from src.backtesting.position_manager import PositionManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class PortfolioSnapshot:
    """Portfolio state on a single date."""

    date: pd.Timestamp
    cash: float
    invested_capital: float
    portfolio_value: float
    daily_return: float
    cumulative_return: float
    drawdown: float


@dataclass(slots=True)
class PortfolioSimulationResult:
    """Container for trades and portfolio history from a simulation run."""

    trades: list[Trade]
    trades_df: pd.DataFrame
    portfolio_history: list[PortfolioSnapshot]
    portfolio_history_df: pd.DataFrame


class PortfolioSimulator:
    """
    Simulates portfolio-level trade execution with shared capital
    constraints across symbols.
    """

    def __init__(
        self,
        initial_capital: float | None = None,
        max_position_size: float | None = None,
        transaction_cost: float | None = None,
        slippage: float | None = None,
    ) -> None:
        """
        Args:
            initial_capital: Total portfolio capital. Defaults to
                settings.INITIAL_CAPITAL.
            max_position_size: Max fraction of capital per position.
                Defaults to settings.MAX_POSITION_SIZE.
            transaction_cost: Fractional cost per trade leg. Defaults
                to settings.TRANSACTION_COST.
            slippage: Fractional adverse execution slippage. Defaults
                to settings.SLIPPAGE.
        """
        self.initial_capital = (
            initial_capital if initial_capital is not None else settings.INITIAL_CAPITAL
        )
        self.max_position_size = (
            max_position_size if max_position_size is not None else settings.MAX_POSITION_SIZE
        )
        self.transaction_cost = (
            transaction_cost if transaction_cost is not None else settings.TRANSACTION_COST
        )
        self.slippage = slippage if slippage is not None else settings.SLIPPAGE

        self.position_manager = PositionManager(
            initial_capital=self.initial_capital,
            max_position_size=self.max_position_size,
        )

        self._trade_id_counter = itertools.count(1)
        self._last_known_price: dict[str, float] = {}
        self._raw_entry_prices: dict[str, float] = {}

    def run(self, signals_df: pd.DataFrame, prices_df: pd.DataFrame) -> PortfolioSimulationResult:
        """
        Run a capital-constrained portfolio simulation.

        Args:
            signals_df: DataFrame with 'symbol', 'date', 'signal' columns.
            prices_df: DataFrame with 'symbol', 'date', 'close' (and
                optionally 'adj_close') columns.

        Returns:
            PortfolioSimulationResult with realized trades and the
            daily portfolio history.

        Raises:
            ValueError: If inputs are empty, missing required columns,
                or have no overlapping (symbol, date) rows.
        """
        self._validate_inputs(signals_df, prices_df)
        prices = self._add_execution_price(prices_df)

        merged = pd.merge(
            signals_df[["symbol", "date", "signal"]],
            prices[["symbol", "date", "execution_price"]],
            on=["symbol", "date"],
            how="inner",
        )

        if merged.empty:
            raise ValueError(
                "No overlapping (symbol, date) rows between signals_df and prices_df."
            )

        events = merged[merged["signal"] != 0].sort_values(["date", "symbol"])
        prices_sorted = prices.sort_values(["date", "symbol"])

        all_dates = sorted(prices["date"].unique())
        prices_by_date = {k: v for k, v in prices_sorted.groupby("date")}
        events_by_date = {k: v for k, v in events.groupby("date")}

        logger.info(
            "Running portfolio simulation across %d dates, %d symbols.",
            len(all_dates),
            prices["symbol"].nunique(),
        )

        trades: list[Trade] = []
        history: list[PortfolioSnapshot] = []
        running_max = self.initial_capital
        prev_value = self.initial_capital

        for current_date in all_dates:
            if current_date in prices_by_date:
                for _, row in prices_by_date[current_date].iterrows():
                    self._last_known_price[row["symbol"]] = row["execution_price"]

            if current_date in events_by_date:
                day_events = events_by_date[current_date]

                for _, row in day_events[day_events["signal"] == -1].iterrows():
                    symbol = row["symbol"]
                    if symbol in self.position_manager.get_open_positions():
                        trades.append(
                            self._close_and_record(symbol, current_date, row["execution_price"])
                        )

                for _, row in day_events[day_events["signal"] == 1].iterrows():
                    self._open_and_record(row["symbol"], current_date, row["execution_price"])

            invested_capital = sum(
                pos.position_size * self._last_known_price.get(sym, pos.entry_price)
                for sym, pos in self.position_manager.get_open_positions().items()
            )
            portfolio_value = self.position_manager.cash + invested_capital

            daily_return = (portfolio_value / prev_value - 1) if prev_value > 0 else 0.0
            cumulative_return = portfolio_value / self.initial_capital - 1
            running_max = max(running_max, portfolio_value)
            drawdown = (portfolio_value - running_max) / running_max if running_max > 0 else 0.0

            history.append(
                PortfolioSnapshot(
                    date=current_date,
                    cash=self.position_manager.cash,
                    invested_capital=invested_capital,
                    portfolio_value=portfolio_value,
                    daily_return=daily_return,
                    cumulative_return=cumulative_return,
                    drawdown=drawdown,
                )
            )
            prev_value = portfolio_value

        self._force_close_remaining(all_dates[-1], trades)

        if not trades:
            logger.warning("Portfolio simulation produced zero trades.")

        trades_df = pd.DataFrame([asdict(t) for t in trades])
        history_df = pd.DataFrame([asdict(s) for s in history])

        logger.info(
            "Portfolio simulation completed with %d trades over %d days.",
            len(trades),
            len(history),
        )

        return PortfolioSimulationResult(
            trades=trades,
            trades_df=trades_df,
            portfolio_history=history,
            portfolio_history_df=history_df,
        )

    def _open_and_record(self, symbol: str, entry_date: pd.Timestamp, raw_entry_price: float) -> None:
        """Open a position if capital constraints allow; no-op otherwise."""
        if not self.position_manager.can_open_position(symbol):
            logger.debug(
                "Skipping BUY for symbol=%s on %s — cannot open "
                "(already open or insufficient cash).",
                symbol,
                entry_date,
            )
            return

        executed_entry_price = raw_entry_price * (1 + self.slippage)
        self.position_manager.open_position(symbol, entry_date, executed_entry_price)
        self._raw_entry_prices[symbol] = raw_entry_price

    def _close_and_record(
        self, symbol: str, exit_date: pd.Timestamp, raw_exit_price: float
    ) -> Trade:
        """Close an open position, apply costs, and build the resulting Trade."""
        executed_exit_price = raw_exit_price * (1 - self.slippage)
        closed = self.position_manager.close_position(symbol, executed_exit_price)
        raw_entry_price = self._raw_entry_prices.pop(symbol)

        fees = (
            self.transaction_cost
            * (closed.entry_price + executed_exit_price)
            * closed.position_size
        )
        self.position_manager.cash -= fees

        slippage_cost = (
            (closed.entry_price - raw_entry_price) * closed.position_size
            + (raw_exit_price - executed_exit_price) * closed.position_size
        )
        pnl = (executed_exit_price - closed.entry_price) * closed.position_size - fees

        return Trade(
            trade_id=next(self._trade_id_counter),
            symbol=symbol,
            entry_date=closed.entry_date,
            exit_date=exit_date,
            entry_price=closed.entry_price,
            exit_price=executed_exit_price,
            position_size=closed.position_size,
            fees=fees,
            slippage_cost=slippage_cost,
            pnl=pnl,
            return_pct=pnl / (closed.entry_price * closed.position_size),
        )

    def _force_close_remaining(self, last_date: pd.Timestamp, trades: list[Trade]) -> None:
        """Force-close any positions still open at the end of the data."""
        remaining = list(self.position_manager.get_open_positions().keys())

        for symbol in remaining:
            exit_price = self._last_known_price[symbol]
            logger.warning(
                "Symbol=%s has an open position at end of data. "
                "Force-closing at last known price=%s on date=%s.",
                symbol,
                exit_price,
                last_date,
            )
            trades.append(self._close_and_record(symbol, last_date, exit_price))

    @staticmethod
    def _validate_inputs(signals_df: pd.DataFrame, prices_df: pd.DataFrame) -> None:
        """Validate required columns and non-empty inputs."""
        if signals_df.empty:
            raise ValueError("signals_df cannot be empty.")
        if prices_df.empty:
            raise ValueError("prices_df cannot be empty.")

        required_signal_cols = {"symbol", "date", "signal"}
        missing = required_signal_cols - set(signals_df.columns)
        if missing:
            raise ValueError(f"signals_df missing required columns: {missing}")

        required_price_cols = {"symbol", "date", "close"}
        missing = required_price_cols - set(prices_df.columns)
        if missing:
            raise ValueError(f"prices_df missing required columns: {missing}")

    @staticmethod
    def _add_execution_price(prices_df: pd.DataFrame) -> pd.DataFrame:
        """Compute execution_price using adj_close with fallback to close."""
        prices = prices_df.copy()
        if "adj_close" in prices.columns:
            prices["execution_price"] = prices["adj_close"].fillna(prices["close"])
        else:
            prices["execution_price"] = prices["close"]
        return prices