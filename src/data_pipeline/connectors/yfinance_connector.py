"""
Yahoo Finance connector.

This module centralizes all yfinance-based market data fetching for the
Quant AI Research Platform.

Responsibilities:
- fetch OHLCV data from Yahoo Finance
- normalize column names to project schema
- add required metadata columns
- validate required market data fields
- retry transient failures
- rate-limit requests to avoid aggressive polling

Output schema aligns with the platform raw market data contract:
symbol, asset_class, exchange, currency, date, open, high, low, close,
adj_close, volume, source.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import yfinance as yf

from config.constants import (
    ASSET_CLASS_EQUITY,
    SOURCE_YFINANCE,
    TIMEFRAME_DAILY,
)


logger = logging.getLogger(__name__)


class YFinanceConnectorError(Exception):
    """Base exception for Yahoo Finance connector errors."""


class YFinanceDataValidationError(YFinanceConnectorError):
    """Raised when fetched data does not match the required schema."""


@dataclass(frozen=True)
class YFinanceRequest:
    """
    Request object for fetching Yahoo Finance data.

    Attributes:
        symbol: Yahoo Finance ticker symbol.
        asset_class: Platform asset class, for example equity, index, etf.
        exchange: Exchange or venue, for example NSE, BSE, NASDAQ.
        currency: Trading currency, for example INR, USD.
        start_date: Inclusive start date in YYYY-MM-DD format.
        end_date: Exclusive end date in YYYY-MM-DD format.
        timeframe: Data interval. Currently defaults to daily.
    """

    symbol: str
    asset_class: str = ASSET_CLASS_EQUITY
    exchange: str = "NSE"
    currency: str = "INR"
    start_date: str | None = None
    end_date: str | None = None
    timeframe: str = TIMEFRAME_DAILY


class YFinanceConnector:
    """
    Connector for fetching market data from Yahoo Finance.

    This class should be used by ingestion jobs instead of calling yfinance
    directly. Keeping yfinance usage centralized makes retries, validation,
    schema normalization, and rate limiting consistent across the platform.
    """

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

    OPTIONAL_COLUMNS: tuple[str, ...] = (
        "adj_close",
        "volume",
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

    INTERVAL_MAP: dict[str, str] = {
        "daily": "1d",
        "weekly": "1wk",
        "monthly": "1mo",
        "hourly": "1h",
    }

    def __init__(
        self,
        *,
        max_retries: int = 3,
        retry_backoff_seconds: float = 2.0,
        rate_limit_seconds: float = 1.0,
        timeout_seconds: int = 30,
    ) -> None:
        """
        Initialize the Yahoo Finance connector.

        Args:
            max_retries: Number of retry attempts for transient failures.
            retry_backoff_seconds: Base delay used for exponential backoff.
            rate_limit_seconds: Minimum delay between outbound requests.
            timeout_seconds: yfinance request timeout in seconds.
        """

        if max_retries < 1:
            raise ValueError("max_retries must be at least 1")

        if retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds cannot be negative")

        if rate_limit_seconds < 0:
            raise ValueError("rate_limit_seconds cannot be negative")

        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.rate_limit_seconds = rate_limit_seconds
        self.timeout_seconds = timeout_seconds

        self._last_request_at: float | None = None

    def fetch_history(self, request: YFinanceRequest) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for one symbol.

        Args:
            request: YFinanceRequest containing symbol and metadata.

        Returns:
            Normalized pandas DataFrame following the raw market data schema.

        Raises:
            YFinanceConnectorError: If data cannot be fetched.
            YFinanceDataValidationError: If returned data is invalid.
        """

        logger.info(
            "Fetching yfinance data for symbol=%s timeframe=%s start=%s end=%s",
            request.symbol,
            request.timeframe,
            request.start_date,
            request.end_date,
        )

        interval = self._resolve_interval(request.timeframe)

        raw_df = self._fetch_with_retry(
            symbol=request.symbol,
            start_date=request.start_date,
            end_date=request.end_date,
            interval=interval,
        )

        normalized_df = self._normalize_dataframe(raw_df, request)
        self._validate_dataframe(normalized_df)

        logger.info(
            "Fetched %s rows from yfinance for symbol=%s",
            len(normalized_df),
            request.symbol,
        )

        return normalized_df

    def fetch_many(self, requests: Iterable[YFinanceRequest]) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for multiple symbols.

        Failed symbols are logged and skipped. If all symbols fail, an exception
        is raised.

        Args:
            requests: Iterable of YFinanceRequest objects.

        Returns:
            Combined normalized DataFrame for all successfully fetched symbols.
        """

        frames: list[pd.DataFrame] = []
        failed_symbols: list[str] = []

        for request in requests:
            try:
                frames.append(self.fetch_history(request))
            except YFinanceConnectorError:
                logger.exception(
                    "Failed to fetch yfinance data for symbol=%s",
                    request.symbol,
                )
                failed_symbols.append(request.symbol)

        if not frames:
            raise YFinanceConnectorError(
                f"Failed to fetch data for all symbols: {failed_symbols}"
            )

        combined_df = pd.concat(frames, ignore_index=True)

        logger.info(
            "Fetched yfinance data for %s symbols. Failed symbols=%s",
            len(frames),
            failed_symbols,
        )

        return combined_df

    def _fetch_with_retry(
        self,
        *,
        symbol: str,
        start_date: str | None,
        end_date: str | None,
        interval: str,
    ) -> pd.DataFrame:
        """
        Fetch data from yfinance with retry and exponential backoff.
        """

        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                self._apply_rate_limit()

                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    auto_adjust=False,
                    actions=False,
                    timeout=self.timeout_seconds,
                )

                if df.empty:
                    raise YFinanceConnectorError(
                        f"No data returned from yfinance for symbol={symbol}"
                    )

                return df

            except Exception as exc:
                last_error = exc

                logger.warning(
                    "yfinance fetch failed for symbol=%s attempt=%s/%s error=%s",
                    symbol,
                    attempt,
                    self.max_retries,
                    exc,
                )

                if attempt < self.max_retries:
                    sleep_seconds = self.retry_backoff_seconds * (2 ** (attempt - 1))
                    time.sleep(sleep_seconds)

        raise YFinanceConnectorError(
            f"Failed to fetch yfinance data for symbol={symbol} "
            f"after {self.max_retries} attempts"
        ) from last_error
    
    def _apply_rate_limit(self) -> None:
        """
        Enforce minimum delay between yfinance requests.
        """

        now = time.monotonic()
        if self._last_request_at is not None:
            elapsed = now - self._last_request_at
            remaining = self.rate_limit_seconds - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self._last_request_at = time.monotonic()

    def _normalize_dataframe(
        self,
        df: pd.DataFrame,
        request: YFinanceRequest,
    ) -> pd.DataFrame:
        """
        Normalize raw yfinance output into project schema.
        """

        normalized_df = df.copy()

        normalized_df = normalized_df.reset_index()
        normalized_df = normalized_df.rename(columns=self.YFINANCE_COLUMN_MAP)

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
        normalized_df = normalized_df.sort_values(["symbol", "date"])
        normalized_df = normalized_df.reset_index(drop=True)

        return normalized_df

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """
        Validate normalized market data against required schema rules.
        """

        missing_columns = [
            column for column in self.REQUIRED_COLUMNS if column not in df.columns
        ]

        if missing_columns:
            raise YFinanceDataValidationError(
                f"Missing required columns: {missing_columns}"
            )

        null_required_columns = [
            column for column in self.REQUIRED_COLUMNS if df[column].isna().any()
        ]

        if null_required_columns:
            raise YFinanceDataValidationError(
                f"Null values found in required columns: {null_required_columns}"
            )

        duplicate_count = df.duplicated(subset=["symbol", "date"]).sum()
        if duplicate_count > 0:
            raise YFinanceDataValidationError(
                f"Found duplicate rows for primary index (symbol, date): "
                f"{duplicate_count}"
            )

        invalid_high_low_count = (df["high"] < df["low"]).sum()
        if invalid_high_low_count > 0:
            raise YFinanceDataValidationError(
                f"Found rows where high < low: {invalid_high_low_count}"
            )

        invalid_high_open_close_count = (
            (df["high"] < df["open"]) | (df["high"] < df["close"])
        ).sum()
        if invalid_high_open_close_count > 0:
            raise YFinanceDataValidationError(
                "Found rows where high is less than open or close: "
                f"{invalid_high_open_close_count}"
            )

        invalid_low_open_close_count = (
            (df["low"] > df["open"]) | (df["low"] > df["close"])
        ).sum()
        if invalid_low_open_close_count > 0:
            raise YFinanceDataValidationError(
                "Found rows where low is greater than open or close: "
                f"{invalid_low_open_close_count}"
            )

        if "volume" in df.columns:
            negative_volume_count = (df["volume"].fillna(0) < 0).sum()
            if negative_volume_count > 0:
                raise YFinanceDataValidationError(
                    f"Found negative volume rows: {negative_volume_count}"
                )

    def _resolve_interval(self, timeframe: str) -> str:
        """
        Convert platform timeframe into yfinance interval.
        """

        try:
            return self.INTERVAL_MAP[timeframe]
        except KeyError as exc:
            supported = tuple(self.INTERVAL_MAP.keys())
            raise ValueError(
                f"Unsupported timeframe={timeframe!r}. "
                f"Supported timeframes: {supported}"
            ) from exc