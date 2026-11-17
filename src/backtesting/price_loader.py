"""
Price data loader for historical backtesting and simulation.

Loads and aligns processed asset prices from the parquet data lake layer.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from src.data_storage.parquet_writer import DataLayer, ParquetWriter
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _align_bound(
    bound: str | pd.Timestamp | None, target_tz: Any
) -> pd.Timestamp | None:
    """Align timestamp boundary timezone with series timezone."""
    if bound is None:
        return None
    ts = pd.Timestamp(bound)
    if target_tz is not None:
        if ts.tzinfo is None:
            return ts.tz_localize(target_tz)
        return ts.tz_convert(target_tz)
    else:
        if ts.tzinfo is not None:
            return ts.tz_localize(None)
        return ts


class PriceLoader:
    """Discovers and loads historical price datasets from the processed data layer."""

    def __init__(self, writer: ParquetWriter | None = None) -> None:
        """
        Initialize PriceLoader.

        Args:
            writer: ParquetWriter instance for reading parquet files.
        """
        self.writer = writer or ParquetWriter()

    def discover_symbols(
        self, asset_class: str = "equity", timeframe: str = "daily"
    ) -> list[str]:
        """
        Discover available symbols in the processed layer.

        Args:
            asset_class: Asset class directory (e.g. 'equity').
            timeframe: Data frequency timeframe (e.g. 'daily').

        Returns:
            Alphabetically sorted list of symbol strings.
        """
        parent_dir = ParquetWriter.build_file_path(
            DataLayer.PROCESSED, asset_class, "_", timeframe
        ).parent
        if not parent_dir.exists():
            return []

        pattern = f"{asset_class}_*_{timeframe}.parquet"
        symbols: list[str] = []
        for p in parent_dir.glob(pattern):
            stem = p.stem
            # Format is <asset_class>_<symbol>_<timeframe>
            # Use split to handle symbols that may contain underscores
            prefix = f"{asset_class}_"
            suffix = f"_{timeframe}"
            if stem.startswith(prefix) and stem.endswith(suffix):
                sym = stem[len(prefix) : -len(suffix)]
                symbols.append(sym)

        return sorted(set(symbols))

    def load(
        self,
        symbols: Sequence[str] | None = None,
        asset_class: str = "equity",
        timeframe: str = "daily",
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """
        Load historical prices for the requested symbols.

        Args:
            symbols: Sequence of symbols to load. If None, discovers all.
            asset_class: Asset class folder name.
            timeframe: Data frequency.
            start: Optional inclusive start date filter.
            end: Optional inclusive end date filter.

        Returns:
            Concatenated, sorted DataFrame containing required price columns.

        Raises:
            ValueError: If no files found, empty input, or required columns missing.
        """
        target_symbols = (
            list(symbols)
            if symbols is not None
            else self.discover_symbols(asset_class, timeframe)
        )
        if not target_symbols:
            raise ValueError("No prices files found for the requested symbols.")

        dfs: list[pd.DataFrame] = []
        for sym in target_symbols:
            try:
                df = self.writer.read(
                    layer=DataLayer.PROCESSED,
                    asset_class=asset_class,
                    symbol=sym,
                    timeframe=timeframe,
                )
            except Exception as e:
                logger.warning(
                    "Could not load prices for symbol=%s: %s. Skipping.", sym, e
                )
                continue

            if df is None or df.empty:
                logger.warning("Empty price dataframe for symbol=%s. Skipping.", sym)
                continue

            # Ensure symbol column exists
            df_copy = df.copy()
            if "symbol" not in df_copy.columns:
                df_copy["symbol"] = sym

            dfs.append(df_copy)

        if not dfs:
            raise ValueError("No prices files found for the requested symbols.")

        combined = pd.concat(dfs, ignore_index=True)

        # Validate required columns
        required_cols = {"symbol", "date", "close"}
        missing = required_cols - set(combined.columns)
        if missing:
            raise ValueError(f"prices_df missing required columns: {missing}")

        # Keep relevant columns in canonical preference
        keep_cols = ["symbol", "date", "close"]
        if "adj_close" in combined.columns:
            keep_cols.append("adj_close")
        if "volume" in combined.columns:
            keep_cols.append("volume")
        # Keep any other original columns
        other_cols = [c for c in combined.columns if c not in keep_cols]
        combined = combined[keep_cols + other_cols]

        # Date filtering
        combined["date"] = pd.to_datetime(combined["date"])
        tz = combined["date"].dt.tz
        start_ts = _align_bound(start, tz)
        end_ts = _align_bound(end, tz)

        if start_ts is not None:
            combined = combined[combined["date"] >= start_ts]
        if end_ts is not None:
            combined = combined[combined["date"] <= end_ts]

        if combined.empty:
            raise ValueError("No price records remain after applying date filters.")

        combined = combined.sort_values(["symbol", "date"]).reset_index(drop=True)
        return combined

    load_prices = load
