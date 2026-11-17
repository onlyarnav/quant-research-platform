"""
Helper functions for normalization and validation of Yahoo Finance market data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from config.constants import (
    ASSET_CLASS_EQUITY,
    SOURCE_YFINANCE,
    TIMEFRAME_DAILY,
)


class YFinanceConnectorError(Exception):
    """Base exception for Yahoo Finance connector errors."""


class YFinanceDataValidationError(YFinanceConnectorError):
    """Raised when fetched data does not match the required schema."""


@dataclass(frozen=True)
class YFinanceRequest:
    """Request object for fetching Yahoo Finance data."""

    symbol: str
    asset_class: str = ASSET_CLASS_EQUITY
    exchange: str = "NSE"
    currency: str = "INR"
    start_date: str | None = None
    end_date: str | None = None
    timeframe: str = TIMEFRAME_DAILY


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

YFINANCE_COLUMN_MAP: dict[str, str] = {
    "Date": "date",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}


def normalize_yfinance_dataframe(
    df: pd.DataFrame,
    request: Any,
) -> pd.DataFrame:
    """Normalize raw yfinance output into project schema."""
    normalized_df = df.copy().reset_index()
    normalized_df = normalized_df.rename(columns=YFINANCE_COLUMN_MAP)

    normalized_df["symbol"] = request.symbol
    normalized_df["asset_class"] = request.asset_class
    normalized_df["exchange"] = request.exchange
    normalized_df["currency"] = request.currency
    normalized_df["source"] = SOURCE_YFINANCE

    if "adj_close" not in normalized_df.columns:
        normalized_df["adj_close"] = pd.NA

    if "volume" not in normalized_df.columns:
        normalized_df["volume"] = pd.NA

    normalized_df["date"] = pd.to_datetime(normalized_df["date"], utc=True)

    selected_columns = [
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
        "source",
    ]

    normalized_df = normalized_df[selected_columns]
    normalized_df = normalized_df.sort_values(["symbol", "date"]).reset_index(drop=True)
    return normalized_df


def validate_yfinance_dataframe(df: pd.DataFrame) -> None:
    """Validate normalized market data against required schema rules."""
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise YFinanceDataValidationError(
            f"Missing required columns: {missing_columns}"
        )

    null_required_columns = [col for col in REQUIRED_COLUMNS if df[col].isna().any()]
    if null_required_columns:
        raise YFinanceDataValidationError(
            f"Null values found in required columns: {null_required_columns}"
        )

    duplicate_count = int(df.duplicated(subset=["symbol", "date"]).sum())
    if duplicate_count > 0:
        raise YFinanceDataValidationError(
            f"Found duplicate rows for primary index (symbol, date): {duplicate_count}"
        )

    invalid_high_low_count = int((df["high"] < df["low"]).sum())
    if invalid_high_low_count > 0:
        raise YFinanceDataValidationError(
            f"Found rows where high < low: {invalid_high_low_count}"
        )

    invalid_high_open_close_count = int(
        ((df["high"] < df["open"]) | (df["high"] < df["close"])).sum()
    )
    if invalid_high_open_close_count > 0:
        raise YFinanceDataValidationError(
            "Found rows where high is less than open or close: "
            f"{invalid_high_open_close_count}"
        )

    invalid_low_open_close_count = int(
        ((df["low"] > df["open"]) | (df["low"] > df["close"])).sum()
    )
    if invalid_low_open_close_count > 0:
        raise YFinanceDataValidationError(
            "Found rows where low is greater than open or close: "
            f"{invalid_low_open_close_count}"
        )

    if "volume" in df.columns:
        negative_volume_count = int((df["volume"].fillna(0) < 0).sum())
        if negative_volume_count > 0:
            raise YFinanceDataValidationError(
                f"Found negative volume rows: {negative_volume_count}"
            )
