"""
Unit tests for DataCleaner.

Tests cover:
- valid data is cleaned and returned correctly
- deduplication removes duplicate rows
- null OHLC rows are removed
- forward fill applied to adj_close and volume
- returns and log_returns are computed correctly
- first row per symbol dropped after returns computation
- output conforms to processed schema columns
- empty result after cleaning raises DataCleaningError
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline.cleaning import DataCleaner, DataCleaningError, PROCESSED_COLUMNS


# ── Fixtures ──────────────────────────────────────────────────


def make_valid_df() -> pd.DataFrame:
    """Return a minimal valid raw market DataFrame."""
    return pd.DataFrame(
        {
            "symbol": ["RELIANCE.NS"] * 5,
            "asset_class": ["equity"] * 5,
            "exchange": ["NSE"] * 5,
            "currency": ["INR"] * 5,
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                ],
                utc=True,
            ),
            "open": [100.0, 102.0, 104.0, 106.0, 108.0],
            "high": [105.0, 107.0, 109.0, 111.0, 113.0],
            "low": [99.0, 101.0, 103.0, 105.0, 107.0],
            "close": [103.0, 105.0, 107.0, 109.0, 111.0],
            "adj_close": [103.0, 105.0, 107.0, 109.0, 111.0],
            "volume": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
            "source": ["yfinance"] * 5,
        }
    )


# ── Tests ─────────────────────────────────────────────────────


class TestDataCleanerOutput:
    def test_clean_returns_dataframe(self) -> None:
        cleaner = DataCleaner()
        result = cleaner.clean(make_valid_df())
        assert isinstance(result, pd.DataFrame)

    def test_output_columns_match_processed_schema(self) -> None:
        cleaner = DataCleaner()
        result = cleaner.clean(make_valid_df())
        assert list(result.columns) == list(PROCESSED_COLUMNS)

    def test_output_is_not_empty(self) -> None:
        cleaner = DataCleaner()
        result = cleaner.clean(make_valid_df())
        assert len(result) > 0

    def test_index_is_reset(self) -> None:
        cleaner = DataCleaner()
        result = cleaner.clean(make_valid_df())
        assert list(result.index) == list(range(len(result)))


class TestDataCleanerDeduplication:
    def test_duplicates_are_removed(self) -> None:
        cleaner = DataCleaner()
        df = make_valid_df()
        df = pd.concat([df, df.iloc[[1]]], ignore_index=True)
        result = cleaner.clean(df)
        assert result.duplicated(subset=["symbol", "date"]).sum() == 0

    def test_no_duplicates_unchanged_row_count(self) -> None:
        cleaner = DataCleaner()
        df = make_valid_df()
        result = cleaner.clean(df)
        # first row dropped due to NaN returns
        assert len(result) == len(df) - 1


class TestDataCleanerNullOHLC:
    def test_null_close_row_removed(self) -> None:
        cleaner = DataCleaner()
        df = make_valid_df()
        df.loc[2, "close"] = None
        result = cleaner.clean(df)
        assert result["close"].isna().sum() == 0

    def test_null_open_row_removed(self) -> None:
        cleaner = DataCleaner()
        df = make_valid_df()
        df.loc[1, "open"] = None
        result = cleaner.clean(df)
        assert result["open"].isna().sum() == 0

    def test_all_null_ohlc_raises(self) -> None:
        cleaner = DataCleaner()
        df = make_valid_df()
        df[["open", "high", "low", "close"]] = None
        with pytest.raises(DataCleaningError):
            cleaner.clean(df)


class TestDataCleanerForwardFill:
    def test_null_volume_forward_filled(self) -> None:
        cleaner = DataCleaner()
        df = make_valid_df()
        df.loc[2, "volume"] = None
        result = cleaner.clean(df)
        assert result["volume"].isna().sum() == 0

    def test_null_adj_close_forward_filled(self) -> None:
        cleaner = DataCleaner()
        df = make_valid_df()
        df.loc[2, "adj_close"] = None
        result = cleaner.clean(df)
        assert result["adj_close"].isna().sum() == 0

    def test_first_row_volume_null_not_filled(self) -> None:
        cleaner = DataCleaner()
        df = make_valid_df()
        df.loc[0, "volume"] = None
        result = cleaner.clean(df)
        # first row dropped due to NaN returns so this should not matter
        assert len(result) > 0


class TestDataCleanerReturns:
    def test_returns_column_exists(self) -> None:
        cleaner = DataCleaner()
        result = cleaner.clean(make_valid_df())
        assert "returns" in result.columns

    def test_log_returns_column_exists(self) -> None:
        cleaner = DataCleaner()
        result = cleaner.clean(make_valid_df())
        assert "log_returns" in result.columns

    def test_no_null_returns(self) -> None:
        cleaner = DataCleaner()
        result = cleaner.clean(make_valid_df())
        assert result["returns"].isna().sum() == 0
        assert result["log_returns"].isna().sum() == 0

    def test_returns_computed_correctly(self) -> None:
        cleaner = DataCleaner()
        df = make_valid_df()
        result = cleaner.clean(df)
        expected_return = (105.0 / 103.0) - 1
        assert abs(result.iloc[0]["returns"] - expected_return) < 1e-6

    def test_log_returns_computed_correctly(self) -> None:
        cleaner = DataCleaner()
        df = make_valid_df()
        result = cleaner.clean(df)
        expected_log_return = np.log(105.0 / 103.0)
        assert abs(result.iloc[0]["log_returns"] - expected_log_return) < 1e-6

    def test_first_row_per_symbol_dropped(self) -> None:
        cleaner = DataCleaner()
        df = make_valid_df()
        result = cleaner.clean(df)
        assert len(result) == len(df) - 1


class TestDataCleanerMultipleSymbols:
    def test_returns_computed_per_symbol(self) -> None:
        cleaner = DataCleaner()
        df1 = make_valid_df()
        df2 = make_valid_df().copy()
        df2["symbol"] = "TCS.NS"
        df = pd.concat([df1, df2], ignore_index=True)
        result = cleaner.clean(df)
        symbols = result["symbol"].unique()
        assert len(symbols) == 2

    def test_first_row_dropped_per_symbol(self) -> None:
        cleaner = DataCleaner()
        df1 = make_valid_df()
        df2 = make_valid_df().copy()
        df2["symbol"] = "TCS.NS"
        df = pd.concat([df1, df2], ignore_index=True)
        result = cleaner.clean(df)
        # 5 rows per symbol minus 1 first row each = 4 per symbol = 8 total
        assert len(result) == 8


class TestDataCleanerChronologicalSort:
    def test_output_is_sorted_chronologically(self) -> None:
        cleaner = DataCleaner()
        df = make_valid_df().iloc[::-1].reset_index(drop=True)
        result = cleaner.clean(df)
        dates = pd.to_datetime(result["date"])
        assert dates.is_monotonic_increasing


class TestDataCleanerEmptyResult:
    def test_all_rows_removed_raises(self) -> None:
        cleaner = DataCleaner()
        df = make_valid_df()
        df[["open", "high", "low", "close"]] = None
        with pytest.raises(DataCleaningError):
            cleaner.clean(df)