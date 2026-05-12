"""
Unit tests for DataValidator.

Tests cover:
- valid data passes all checks
- empty DataFrame detection
- missing required columns
- null values in required columns
- duplicate (symbol, date) rows
- OHLC relationship violations
- negative volume detection
- non-chronological order warning
- validate_or_raise behaviour
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_pipeline.validation import (
    DataValidationError,
    DataValidator,
    ValidationResult,
)


# ── Fixtures ──────────────────────────────────────────────────


def make_valid_df() -> pd.DataFrame:
    """Return a minimal valid raw market DataFrame."""
    return pd.DataFrame(
        {
            "symbol": ["RELIANCE.NS", "RELIANCE.NS", "RELIANCE.NS"],
            "asset_class": ["equity", "equity", "equity"],
            "exchange": ["NSE", "NSE", "NSE"],
            "currency": ["INR", "INR", "INR"],
            "date": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03"], utc=True
            ),
            "open": [100.0, 101.0, 102.0],
            "high": [105.0, 106.0, 107.0],
            "low": [99.0, 100.0, 101.0],
            "close": [103.0, 104.0, 105.0],
            "adj_close": [103.0, 104.0, 105.0],
            "volume": [1000.0, 2000.0, 3000.0],
            "source": ["yfinance", "yfinance", "yfinance"],
        }
    )


# ── Tests ─────────────────────────────────────────────────────


class TestDataValidatorValidData:
    def test_valid_dataframe_passes(self) -> None:
        validator = DataValidator()
        result = validator.validate(make_valid_df())
        assert result.is_valid is True
        assert result.errors == []

    def test_valid_dataframe_correct_row_count(self) -> None:
        validator = DataValidator()
        df = make_valid_df()
        result = validator.validate(df)
        assert result.rows_total == len(df)


class TestDataValidatorEmptyDataFrame:
    def test_empty_dataframe_fails(self) -> None:
        validator = DataValidator()
        result = validator.validate(pd.DataFrame())
        assert result.is_valid is False
        assert any("empty" in e.lower() for e in result.errors)


class TestDataValidatorRequiredColumns:
    def test_missing_single_column_fails(self) -> None:
        validator = DataValidator()
        df = make_valid_df().drop(columns=["close"])
        result = validator.validate(df)
        assert result.is_valid is False
        assert any("close" in e for e in result.errors)

    def test_missing_multiple_columns_fails(self) -> None:
        validator = DataValidator()
        df = make_valid_df().drop(columns=["open", "high"])
        result = validator.validate(df)
        assert result.is_valid is False

    def test_all_required_columns_present_passes(self) -> None:
        validator = DataValidator()
        result = validator.validate(make_valid_df())
        assert result.is_valid is True


class TestDataValidatorNullValues:
    def test_null_in_close_fails(self) -> None:
        validator = DataValidator()
        df = make_valid_df()
        df.loc[0, "close"] = None
        result = validator.validate(df)
        assert result.is_valid is False
        assert any("close" in e for e in result.errors)

    def test_null_in_symbol_fails(self) -> None:
        validator = DataValidator()
        df = make_valid_df()
        df.loc[0, "symbol"] = None
        result = validator.validate(df)
        assert result.is_valid is False

    def test_null_in_optional_volume_passes(self) -> None:
        validator = DataValidator()
        df = make_valid_df()
        df["volume"] = None
        result = validator.validate(df)
        assert result.is_valid is True


class TestDataValidatorDuplicates:
    def test_duplicate_symbol_date_fails(self) -> None:
        validator = DataValidator()
        df = make_valid_df()
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        result = validator.validate(df)
        assert result.is_valid is False
        assert any("duplicate" in e.lower() for e in result.errors)

    def test_no_duplicates_passes(self) -> None:
        validator = DataValidator()
        result = validator.validate(make_valid_df())
        assert result.is_valid is True


class TestDataValidatorOHLC:
    def test_high_less_than_low_fails(self) -> None:
        validator = DataValidator()
        df = make_valid_df()
        df.loc[0, "high"] = 90.0
        df.loc[0, "low"] = 95.0
        result = validator.validate(df)
        assert result.is_valid is False
        assert any("high < low" in e for e in result.errors)

    def test_high_less_than_open_fails(self) -> None:
        validator = DataValidator()
        df = make_valid_df()
        df.loc[0, "high"] = 90.0
        df.loc[0, "open"] = 95.0
        result = validator.validate(df)
        assert result.is_valid is False

    def test_high_less_than_close_fails(self) -> None:
        validator = DataValidator()
        df = make_valid_df()
        df.loc[0, "high"] = 90.0
        df.loc[0, "close"] = 95.0
        result = validator.validate(df)
        assert result.is_valid is False

    def test_low_greater_than_open_fails(self) -> None:
        validator = DataValidator()
        df = make_valid_df()
        df.loc[0, "low"] = 110.0
        result = validator.validate(df)
        assert result.is_valid is False

    def test_low_greater_than_close_fails(self) -> None:
        validator = DataValidator()
        df = make_valid_df()
        df.loc[0, "low"] = 110.0
        result = validator.validate(df)
        assert result.is_valid is False

    def test_valid_ohlc_passes(self) -> None:
        validator = DataValidator()
        result = validator.validate(make_valid_df())
        assert result.is_valid is True


class TestDataValidatorVolume:
    def test_negative_volume_fails(self) -> None:
        validator = DataValidator()
        df = make_valid_df()
        df.loc[0, "volume"] = -500.0
        result = validator.validate(df)
        assert result.is_valid is False
        assert any("negative volume" in e.lower() for e in result.errors)

    def test_zero_volume_passes(self) -> None:
        validator = DataValidator()
        df = make_valid_df()
        df.loc[0, "volume"] = 0.0
        result = validator.validate(df)
        assert result.is_valid is True


class TestDataValidatorChronologicalOrder:
    def test_non_chronological_adds_warning(self) -> None:
        validator = DataValidator()
        df = make_valid_df().iloc[::-1].reset_index(drop=True)
        result = validator.validate(df)
        assert len(result.warnings) > 0

    def test_chronological_no_warning(self) -> None:
        validator = DataValidator()
        result = validator.validate(make_valid_df())
        assert result.warnings == []


class TestDataValidatorValidateOrRaise:
    def test_valid_df_does_not_raise(self) -> None:
        validator = DataValidator()
        validator.validate_or_raise(make_valid_df())

    def test_invalid_df_raises(self) -> None:
        validator = DataValidator()
        with pytest.raises(DataValidationError):
            validator.validate_or_raise(pd.DataFrame())