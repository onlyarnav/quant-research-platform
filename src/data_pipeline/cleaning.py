"""
Data cleaning pipeline.

Responsibilities:
- deduplicate rows
- forward fill missing OHLCV values where appropriate
- remove corrupted rows
- sort chronologically
- compute returns and log_returns
- produce processed dataset conforming to processed schema

Input schema:  raw market data (symbol, asset_class, exchange, currency,
               date, open, high, low, close, adj_close, volume, source)

Output schema: processed market data (adds returns, log_returns)

Usage:
    from src.data_pipeline.cleaning import DataCleaner
    cleaner = DataCleaner()
    processed_df = cleaner.clean(raw_df)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


PROCESSED_COLUMNS: tuple[str, ...] = (
    "symbol",
    "asset_class",
    "exchange",
    "currency",
    "date",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "returns",
    "log_returns",
)


class DataCleaningError(Exception):
    """Raised when cleaning pipeline encounters an unrecoverable error."""


class DataCleaner:
    """
    Cleans raw market data and produces processed datasets.

    Cleaning steps applied in order:
    1. Deduplicate (symbol, date) rows
    2. Sort chronologically by (symbol, date)
    3. Remove rows with null required OHLC values
    4. Forward fill missing adj_close and volume within each symbol
    5. Compute simple returns and log returns per symbol
    6. Select and return processed schema columns
    """

    REQUIRED_OHLC_COLUMNS: tuple[str, ...] = (
        "open",
        "high",
        "low",
        "close",
    )

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the full cleaning pipeline on a raw market DataFrame.

        Args:
            df: Raw validated market data DataFrame.

        Returns:
            Cleaned processed DataFrame conforming to processed schema.

        Raises:
            DataCleaningError: If cleaning produces an empty DataFrame.
        """
        logger.info("Starting data cleaning. input_rows=%s", len(df))

        df = self._deduplicate(df)
        df = self._sort_chronologically(df)
        df = self._remove_null_ohlc_rows(df)
        df = self._forward_fill_optional_columns(df)
        df = self._compute_returns(df)
        df = self._select_processed_columns(df)

        if df.empty:
            raise DataCleaningError(
                "Cleaning pipeline produced an empty DataFrame. "
                "All rows were removed during cleaning."
            )

        logger.info("Cleaning complete. output_rows=%s", len(df))

        return df

    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate (symbol, date) rows keeping the last occurrence.
        """
        before = len(df)
        df = df.drop_duplicates(subset=["symbol", "date"], keep="last")
        removed = before - len(df)

        if removed > 0:
            logger.warning("Removed %s duplicate rows.", removed)

        return df

    def _sort_chronologically(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sort data by (symbol, date) ascending.
        """
        df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
        return df

    def _remove_null_ohlc_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop rows with null values in required OHLC columns.
        """
        before = len(df)
        df = df.dropna(subset=list(self.REQUIRED_OHLC_COLUMNS))
        removed = before - len(df)

        if removed > 0:
            logger.warning(
                "Removed %s rows with null OHLC values.", removed
            )

        return df

    def _forward_fill_optional_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Forward fill adj_close and volume within each symbol group.

        Forward fill is appropriate for these columns because missing values
        are typically due to exchange holidays or data gaps, not real zeros.
        """
        optional_columns = ["adj_close", "volume"]
        existing_optional = [
            col for col in optional_columns if col in df.columns
        ]

        if not existing_optional:
            return df

        df[existing_optional] = (
            df.groupby("symbol", group_keys=False)[existing_optional]
            .apply(lambda group: group.ffill())
        )

        return df

    def _compute_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute simple returns and log returns per symbol.

        Uses adj_close if available, otherwise falls back to close.

        Formulas:
            returns     = price(t) / price(t-1) - 1
            log_returns = log(price(t) / price(t-1))

        The first row per symbol will have NaN returns and is dropped.
        """
        price_column = "adj_close" if "adj_close" in df.columns else "close"

        df["returns"] = (
            df.groupby("symbol")[price_column]
            .pct_change()
        )

        df["log_returns"] = (
            df.groupby("symbol")[price_column]
            .transform(lambda x: np.log(x / x.shift(1)))
        )

        before = len(df)
        df = df.dropna(subset=["returns", "log_returns"])
        removed = before - len(df)

        if removed > 0:
            logger.debug(
                "Dropped %s rows with NaN returns (first row per symbol).",
                removed,
            )

        return df

    def _select_processed_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select and order columns to match the processed schema.

        Columns not in the processed schema are dropped.
        Missing optional columns are filled with NaN.
        """
        for col in PROCESSED_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan

        df = df[list(PROCESSED_COLUMNS)]
        df = df.reset_index(drop=True)

        return df