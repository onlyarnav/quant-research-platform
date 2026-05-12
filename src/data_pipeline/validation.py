"""
Data validation pipeline.

Validates raw market data against schema rules defined in Data_Schema.md
before promotion to the processed data layer.

Validation rules:
- no duplicate (symbol, date) rows
- no null values in required columns
- no negative volume
- high >= low
- high >= open and high >= close
- low <= open and low <= close
- chronological sort enforced

Usage:
    from src.data_pipeline.validation import DataValidator
    validator = DataValidator()
    result = validator.validate(df)
    if result.is_valid:
        ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


REQUIRED_COLUMNS: tuple[str, ...] = (
    "symbol",
    "asset_class",
    "exchange",
    "currency",
    "date",
    "open",
    "high",
    "low",
    "close",
    "source",
)


@dataclass
class ValidationResult:
    """
    Result of a validation run against a DataFrame.

    Attributes:
        is_valid: True if no validation errors were found.
        errors: List of validation error messages.
        warnings: List of non-fatal warnings.
        rows_total: Total rows in the DataFrame.
        rows_invalid: Number of rows that failed validation.
    """

    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    rows_total: int = 0
    rows_invalid: int = 0

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)


class DataValidationError(Exception):
    """Raised when validation fails and pipeline must stop."""


class DataValidator:
    """
    Validates raw market DataFrames against platform schema rules.

    Each check is independent. All checks run before returning the result
    so the caller gets a full picture of all violations at once.
    """

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """
        Run all validation checks against a DataFrame.

        Args:
            df: Raw market data DataFrame to validate.

        Returns:
            ValidationResult with is_valid flag and full error list.
        """
        result = ValidationResult(rows_total=len(df))

        logger.info("Running validation on DataFrame with %s rows.", len(df))

        self._check_not_empty(df, result)

        if not result.is_valid:
            return result

        self._check_required_columns(df, result)

        if not result.is_valid:
            return result

        self._check_null_required_columns(df, result)
        self._check_duplicate_index(df, result)
        self._check_ohlc_relationships(df, result)
        self._check_negative_volume(df, result)
        self._check_chronological_order(df, result)

        if result.is_valid:
            logger.info("Validation passed. rows=%s", len(df))
        else:
            logger.warning(
                "Validation failed. errors=%s rows_invalid=%s",
                len(result.errors),
                result.rows_invalid,
            )

        return result

    def validate_or_raise(self, df: pd.DataFrame) -> None:
        """
        Validate a DataFrame and raise if validation fails.

        Args:
            df: Raw market data DataFrame to validate.

        Raises:
            DataValidationError: If any validation check fails.
        """
        result = self.validate(df)
        if not result.is_valid:
            error_summary = " | ".join(result.errors)
            raise DataValidationError(
                f"Validation failed with {len(result.errors)} error(s): {error_summary}"
            )

    def _check_not_empty(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
    ) -> None:
        if df.empty:
            result.add_error("DataFrame is empty.")

    def _check_required_columns(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
    ) -> None:
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            result.add_error(f"Missing required columns: {missing}")

    def _check_null_required_columns(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
    ) -> None:
        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                continue
            null_count = df[col].isna().sum()
            if null_count > 0:
                result.add_error(
                    f"Column '{col}' has {null_count} null value(s)."
                )
                result.rows_invalid += int(null_count)

    def _check_duplicate_index(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
    ) -> None:
        if "symbol" not in df.columns or "date" not in df.columns:
            return
        duplicate_count = df.duplicated(subset=["symbol", "date"]).sum()
        if duplicate_count > 0:
            result.add_error(
                f"Found {duplicate_count} duplicate (symbol, date) row(s)."
            )
            result.rows_invalid += int(duplicate_count)

    def _check_ohlc_relationships(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
    ) -> None:
        required = {"high", "low", "open", "close"}
        if not required.issubset(df.columns):
            return

        high_lt_low = (df["high"] < df["low"]).sum()
        if high_lt_low > 0:
            result.add_error(
                f"Found {high_lt_low} row(s) where high < low."
            )
            result.rows_invalid += int(high_lt_low)

        high_lt_open = (df["high"] < df["open"]).sum()
        if high_lt_open > 0:
            result.add_error(
                f"Found {high_lt_open} row(s) where high < open."
            )
            result.rows_invalid += int(high_lt_open)

        high_lt_close = (df["high"] < df["close"]).sum()
        if high_lt_close > 0:
            result.add_error(
                f"Found {high_lt_close} row(s) where high < close."
            )
            result.rows_invalid += int(high_lt_close)

        low_gt_open = (df["low"] > df["open"]).sum()
        if low_gt_open > 0:
            result.add_error(
                f"Found {low_gt_open} row(s) where low > open."
            )
            result.rows_invalid += int(low_gt_open)

        low_gt_close = (df["low"] > df["close"]).sum()
        if low_gt_close > 0:
            result.add_error(
                f"Found {low_gt_close} row(s) where low > close."
            )
            result.rows_invalid += int(low_gt_close)

    def _check_negative_volume(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
    ) -> None:
        if "volume" not in df.columns:
            return
        negative_count = (
            pd.to_numeric(df["volume"], errors="coerce").fillna(0) < 0
        ).sum()
        if negative_count > 0:
            result.add_error(
                f"Found {negative_count} row(s) with negative volume."
            )
            result.rows_invalid += int(negative_count)

    def _check_chronological_order(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
    ) -> None:
        if "date" not in df.columns:
            return
        dates = pd.to_datetime(df["date"])
        if not dates.is_monotonic_increasing:
            result.add_warning(
                "Data is not sorted chronologically. "
                "Pipeline will sort before writing."
            )