"""
Data models and report structures for the ingestion pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config.constants import TIMEFRAME_DAILY


@dataclass(frozen=True)
class AssetConfig:
    """
    Configuration for a single asset to be ingested.

    Attributes:
        symbol: Yahoo Finance ticker symbol.
        asset_class: Platform asset class.
        exchange: Trading exchange or venue.
        currency: Trading currency.
        timeframe: Data timeframe. Defaults to daily.
    """

    symbol: str
    asset_class: str
    exchange: str
    currency: str
    timeframe: str = TIMEFRAME_DAILY


@dataclass
class IngestionResult:
    """
    Summary result of a single asset ingestion attempt.

    Attributes:
        symbol: Asset symbol.
        success: Whether ingestion succeeded.
        rows_written: Number of rows written to storage.
        error: Error message if ingestion failed.
    """

    symbol: str
    success: bool
    rows_written: int = 0
    error: str | None = None


@dataclass
class IngestionReport:
    """
    Aggregate report for a full ingestion pipeline run.

    Attributes:
        total: Total number of assets attempted.
        succeeded: Number of successfully ingested assets.
        failed: Number of failed assets.
        results: Per-asset ingestion results.
    """

    total: int = 0
    succeeded: int = 0
    failed: int = 0
    results: list[IngestionResult] = field(default_factory=list)

    def add(self, result: IngestionResult) -> None:
        self.results.append(result)
        self.total += 1
        if result.success:
            self.succeeded += 1
        else:
            self.failed += 1


class IngestionPipelineError(Exception):
    """Raised when the entire ingestion pipeline fails."""
