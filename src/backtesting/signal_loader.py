"""
Signal data loader for historical strategy signals.

Loads, validates, and aligns trading signals from the signals parquet layer.
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


class SignalLoader:
    """Discovers, loads, and validates quantitative signals from the signals layer."""

    def __init__(self, writer: ParquetWriter | None = None) -> None:
        """
        Initialize SignalLoader.

        Args:
            writer: ParquetWriter instance for reading parquet files.
        """
        self.writer = writer or ParquetWriter()

    def discover_symbols(
        self, asset_class: str = "equity", timeframe: str = "daily"
    ) -> list[str]:
        """
        Discover available symbols in the signals data layer.

        Args:
            asset_class: Asset class directory.
            timeframe: Data frequency timeframe.

        Returns:
            Sorted list of symbol names present in the signals layer.
        """
        parent_dir = ParquetWriter.build_file_path(
            DataLayer.SIGNALS, asset_class, "_", timeframe
        ).parent
        if not parent_dir.exists():
            return []

        pattern = f"{asset_class}_*_{timeframe}.parquet"
        symbols: list[str] = []
        for p in parent_dir.glob(pattern):
            stem = p.stem
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
        Load and validate signals for the requested symbols.

        Args:
            symbols: Sequence of symbols to load. If None, discovers all.
            asset_class: Asset class folder name.
            timeframe: Data frequency.
            start: Optional inclusive start date filter.
            end: Optional inclusive end date filter.

        Returns:
            Validated, sorted DataFrame of trading signals.

        Raises:
            ValueError: If no files found, missing columns, invalid signal values,
                duplicate (symbol, date) rows, or NaN in key columns.
        """
        target_symbols = (
            list(symbols)
            if symbols is not None
            else self.discover_symbols(asset_class, timeframe)
        )
        if not target_symbols:
            raise ValueError("No signals files found for the requested symbols.")

        dfs: list[pd.DataFrame] = []
        for sym in target_symbols:
            try:
                df = self.writer.read(
                    layer=DataLayer.SIGNALS,
                    asset_class=asset_class,
                    symbol=sym,
                    timeframe=timeframe,
                )
            except Exception as e:
                logger.warning(
                    "Could not load signals for symbol=%s: %s. Skipping.", sym, e
                )
                continue

            if df is None or df.empty:
                logger.warning("Empty signal dataframe for symbol=%s. Skipping.", sym)
                continue

            df_copy = df.copy()
            if "symbol" not in df_copy.columns:
                df_copy["symbol"] = sym

            dfs.append(df_copy)

        if not dfs:
            raise ValueError("No signals files found for the requested symbols.")

        combined = pd.concat(dfs, ignore_index=True)

        # Validate required columns
        required_cols = {"symbol", "date", "predicted_return", "signal"}
        missing = required_cols - set(combined.columns)
        if missing:
            raise ValueError(f"signals_df missing required columns: {missing}")

        # Check for NaNs in critical columns
        for col in ["symbol", "date", "signal"]:
            if combined[col].isna().any():
                raise ValueError(f"NaN values found in required signal column: {col}")

        # Validate signal values {-1, 0, 1}
        unique_signals = set(combined["signal"].unique())
        if not unique_signals.issubset({-1, 0, 1}):
            raise ValueError(
                f"signal must be in {{-1, 0, 1}}, found invalid values: {unique_signals - {-1, 0, 1}}"
            )

        # Check for duplicate (symbol, date)
        if combined.duplicated(subset=["symbol", "date"]).any():
            raise ValueError("Duplicate (symbol, date) rows in signals")

        # Date filtering with tz alignment
        combined["date"] = pd.to_datetime(combined["date"])
        tz = combined["date"].dt.tz
        start_ts = _align_bound(start, tz)
        end_ts = _align_bound(end, tz)

        if start_ts is not None:
            combined = combined[combined["date"] >= start_ts]
        if end_ts is not None:
            combined = combined[combined["date"] <= end_ts]

        if combined.empty:
            raise ValueError("No signal records remain after applying date filters.")

        combined = combined.sort_values(["symbol", "date"]).reset_index(drop=True)
        return combined

    load_signals = load
