"""
Benchmark builder for creating reference market benchmarks.

Constructs buy-and-hold index histories or equal-weight universe histories
conforming to the canonical 7-column portfolio history format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

CANONICAL_COLUMNS = [
    "date",
    "portfolio_value",
    "cash",
    "invested_capital",
    "daily_return",
    "drawdown",
    "total_return",
]


class BenchmarkBuilder:
    """Builds aligned benchmark histories for strategy performance comparison."""

    def __init__(self, processed_path: Path | str | None = None) -> None:
        """
        Initialize BenchmarkBuilder.

        Args:
            processed_path: Path to processed parquet files. Defaults to settings.DATA_PROCESSED_PATH.
        """
        self.processed_path = (
            Path(settings.DATA_PROCESSED_PATH)
            if processed_path is None
            else Path(processed_path)
        )

    def _assemble_canonical_history(
        self,
        dates: Sequence[pd.Timestamp],
        daily_returns: Sequence[float],
        initial_capital: float,
    ) -> pd.DataFrame:
        """Assemble canonical 7-column history from dates and daily returns."""
        dates_s = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
        rets_s = (
            pd.Series(daily_returns, dtype=float).fillna(0.0).reset_index(drop=True)
        )

        cum_growth = (1.0 + rets_s).cumprod()
        pv = initial_capital * cum_growth
        cummax = pv.cummax()
        dd = (pv / cummax) - 1.0
        tot_ret = (pv / initial_capital) - 1.0

        return pd.DataFrame(
            {
                "date": dates_s,
                "portfolio_value": pv,
                "cash": 0.0,
                "invested_capital": pv,
                "daily_return": rets_s,
                "drawdown": dd,
                "total_return": tot_ret,
            }
        )[CANONICAL_COLUMNS]

    def from_index(
        self,
        symbol: str | None = None,
        asset_class: str = "index",
        timeframe: str = "daily",
        initial_capital: float = 100_000.0,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
    ) -> pd.DataFrame | None:
        """
        Build buy-and-hold benchmark history from a processed index parquet file.

        Args:
            symbol: Index symbol. Defaults to settings.BENCHMARK_SYMBOL.
            asset_class: Asset class folder name. Defaults to "index".
            timeframe: Data timeframe. Defaults to "daily".
            initial_capital: Initial portfolio value.
            start: Optional start date filter.
            end: Optional end date filter.

        Returns:
            Canonical benchmark DataFrame or None if index file does not exist.
        """
        sym = symbol or settings.BENCHMARK_SYMBOL
        file_path = self.processed_path / asset_class / timeframe / f"{sym}.parquet"

        if not file_path.exists():
            logger.warning("Benchmark index file not found: %s", file_path)
            return None

        try:
            df = pd.read_parquet(file_path)
        except Exception as exc:
            logger.warning(
                "Failed to load benchmark index parquet %s: %s", file_path, exc
            )
            return None

        if "date" not in df.columns or "close" not in df.columns:
            logger.warning(
                "Index parquet missing 'date' or 'close' columns: %s", file_path
            )
            return None

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        if start is not None:
            df = df[df["date"] >= pd.to_datetime(start)]
        if end is not None:
            df = df[df["date"] <= pd.to_datetime(end)]

        if df.empty:
            logger.warning("Index DataFrame is empty after date filtering.")
            return None

        first_close = float(df["close"].iloc[0])
        if first_close <= 0.0:
            logger.warning("First close of index is non-positive (%s).", first_close)
            return None

        pv = initial_capital * (df["close"] / first_close)
        daily_ret = pv.pct_change().fillna(0.0)
        cummax = pv.cummax()
        dd = (pv / cummax) - 1.0
        tot_ret = (pv / initial_capital) - 1.0

        res = pd.DataFrame(
            {
                "date": df["date"],
                "portfolio_value": pv,
                "cash": 0.0,
                "invested_capital": pv,
                "daily_return": daily_ret,
                "drawdown": dd,
                "total_return": tot_ret,
            }
        )[CANONICAL_COLUMNS]

        return res.reset_index(drop=True)

    def equal_weight_universe(
        self,
        prices_df: pd.DataFrame,
        initial_capital: float = 100_000.0,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
    ) -> pd.DataFrame:
        """
        Build daily rebalanced equal-weight universe benchmark.

        Args:
            prices_df: Long DataFrame containing at least ['date', 'symbol', 'close'].
            initial_capital: Initial capital.
            start: Optional start date filter.
            end: Optional end date filter.

        Returns:
            Canonical benchmark DataFrame.
        """
        if prices_df is None or prices_df.empty:
            raise ValueError("prices_df cannot be None or empty.")

        req_cols = {"date", "symbol", "close"}
        if not req_cols.issubset(prices_df.columns):
            raise ValueError(
                f"prices_df missing required columns: {req_cols - set(prices_df.columns)}"
            )

        df = prices_df.copy()
        df["date"] = pd.to_datetime(df["date"])

        if start is not None:
            df = df[df["date"] >= pd.to_datetime(start)]
        if end is not None:
            df = df[df["date"] <= pd.to_datetime(end)]

        if df.empty:
            raise ValueError("prices_df is empty after date filtering.")

        pivoted = df.pivot(index="date", columns="symbol", values="close").sort_index()
        daily_rets = pivoted.pct_change()

        # Mean of daily returns across available symbols on that day
        mean_ret = daily_rets.mean(axis=1, skipna=True).fillna(0.0)

        dates = list(pivoted.index)
        rets = list(mean_ret.values)

        return self._assemble_canonical_history(dates, rets, initial_capital)

    def build(
        self,
        prices_df: pd.DataFrame | None = None,
        initial_capital: float = 100_000.0,
        dates: Sequence[pd.Timestamp] | None = None,
        prefer_index: bool = True,
        symbol: str | None = None,
        asset_class: str = "index",
        timeframe: str = "daily",
    ) -> tuple[pd.DataFrame, str]:
        """
        Build and align benchmark against strategy dates.

        Args:
            prices_df: Optional multi-symbol prices DataFrame for universe fallback.
            initial_capital: Initial capital.
            dates: Optional sequence of target strategy dates to align with.
            prefer_index: Whether to prefer index over equal-weight universe.
            symbol: Index symbol.
            asset_class: Index asset class.
            timeframe: Index timeframe.

        Returns:
            Tuple of (aligned_history_df, label).
        """
        bench_df = None
        label = "Benchmark"

        if prefer_index:
            bench_df = self.from_index(
                symbol=symbol,
                asset_class=asset_class,
                timeframe=timeframe,
                initial_capital=initial_capital,
            )
            if bench_df is not None:
                label = symbol or settings.BENCHMARK_SYMBOL

        if bench_df is None:
            if prices_df is None or prices_df.empty:
                raise ValueError(
                    "Cannot build benchmark: index missing and no prices_df provided."
                )
            bench_df = self.equal_weight_universe(
                prices_df, initial_capital=initial_capital
            )
            label = "Equal-Weight Universe"

        if dates is not None and len(dates) > 0:
            target_dates = (
                pd.to_datetime(pd.Series(dates)).sort_values().drop_duplicates()
            )
            bench_df["date"] = pd.to_datetime(bench_df["date"])
            bench_indexed = bench_df.set_index("date").sort_index()

            # Reindex to target dates, forward fill missing <= 5 consecutive
            reindexed = bench_indexed.reindex(target_dates).ffill(limit=5)
            # Drop any remaining unfillable rows
            valid = reindexed.dropna(subset=["portfolio_value"]).reset_index()
            valid = valid.rename(columns={"index": "date"})
            if valid.empty:
                raise ValueError(
                    "No overlapping dates between strategy and benchmark after alignment."
                )
            bench_df = valid[CANONICAL_COLUMNS]

        return bench_df.reset_index(drop=True), label
