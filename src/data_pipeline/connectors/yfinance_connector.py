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

import time
from typing import Iterable

import pandas as pd
import yfinance as yf

from src.data_pipeline.connectors.yfinance_helpers import (
    REQUIRED_COLUMNS,
    YFINANCE_COLUMN_MAP,
    YFinanceConnectorError,
    YFinanceDataValidationError,
    YFinanceRequest,
    normalize_yfinance_dataframe,
    validate_yfinance_dataframe,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = [
    "YFinanceConnector",
    "YFinanceRequest",
    "YFinanceConnectorError",
    "YFinanceDataValidationError",
]

logger = get_logger(__name__)


class YFinanceConnector:
    """
    Connector for fetching market data from Yahoo Finance.

    This class should be used by ingestion jobs instead of calling yfinance
    directly. Keeping yfinance usage centralized makes retries, validation,
    schema normalization, and rate limiting consistent across the platform.
    """

    REQUIRED_COLUMNS = REQUIRED_COLUMNS
    OPTIONAL_COLUMNS: tuple[str, ...] = ("adj_close", "volume")
    YFINANCE_COLUMN_MAP = YFINANCE_COLUMN_MAP

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
        """Normalize raw yfinance output into project schema."""
        return normalize_yfinance_dataframe(df, request)

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate normalized market data against required schema rules."""
        validate_yfinance_dataframe(df)

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
