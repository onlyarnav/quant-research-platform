"""
Historical data ingestion pipeline.

Responsibilities:
- accept a universe of assets to ingest
- fetch historical OHLCV data via YFinanceConnector
- write raw data to parquet data lake via ParquetWriter
- support full and incremental update modes
- log ingestion results and failures

Usage:
    from src.data_pipeline.ingestion import IngestionPipeline, AssetConfig
    
    pipeline = IngestionPipeline()
    pipeline.run([
        AssetConfig(symbol="RELIANCE.NS", asset_class="equity", exchange="NSE", currency="INR"),
        AssetConfig(symbol="^NSEI", asset_class="index", exchange="NSE", currency="INR"),
    ])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

import pandas as pd

from config.constants import TIMEFRAME_DAILY
from config.settings import settings
from src.data_pipeline.connectors.yfinance_connector import (
    YFinanceConnector,
    YFinanceConnectorError,
    YFinanceRequest,
)
from src.data_storage.parquet_writer import DataLayer, ParquetWriter, ParquetWriterError

logger = logging.getLogger(__name__)


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


class IngestionPipeline:
    """
    Historical and incremental data ingestion pipeline.

    Fetches OHLCV market data from Yahoo Finance and writes
    raw parquet files to the data lake.

    Supports two modes controlled by settings.DATA_UPDATE_MODE:
        full        — download full history from DATA_START_DATE
        incremental — append only data newer than last stored date
    """

    def __init__(
        self,
        connector: YFinanceConnector | None = None,
        writer: ParquetWriter | None = None,
    ) -> None:
        """
        Initialize the ingestion pipeline.

        Args:
            connector: YFinanceConnector instance. Created with defaults if not provided.
            writer: ParquetWriter instance. Created with defaults if not provided.
        """
        self.connector = connector or YFinanceConnector()
        self.writer = writer or ParquetWriter()
        self.mode = settings.DATA_UPDATE_MODE
        self.start_date = settings.DATA_START_DATE

    def run(self, assets: list[AssetConfig]) -> IngestionReport:
        """
        Run ingestion for a list of assets.

        Args:
            assets: List of AssetConfig objects to ingest.

        Returns:
            IngestionReport summarizing success and failure counts.

        Raises:
            IngestionPipelineError: If assets list is empty.
        """
        if not assets:
            raise IngestionPipelineError("Asset list is empty. Nothing to ingest.")

        logger.info(
            "Starting ingestion pipeline. mode=%s assets=%s",
            self.mode,
            len(assets),
        )

        report = IngestionReport()

        for asset in assets:
            result = self._ingest_asset(asset)
            report.add(result)

        logger.info(
            "Ingestion pipeline complete. total=%s succeeded=%s failed=%s",
            report.total,
            report.succeeded,
            report.failed,
        )

        if report.failed > 0:
            failed_symbols = [
                r.symbol for r in report.results if not r.success
            ]
            logger.warning("Failed symbols: %s", failed_symbols)

        return report

    def _ingest_asset(self, asset: AssetConfig) -> IngestionResult:
        """
        Ingest a single asset.

        Args:
            asset: AssetConfig for the asset to ingest.

        Returns:
            IngestionResult for this asset.
        """
        logger.info(
            "Ingesting symbol=%s asset_class=%s mode=%s",
            asset.symbol,
            asset.asset_class,
            self.mode,
        )

        try:
            start_date, end_date = self._resolve_date_range(asset)

            if start_date is None:
                logger.info(
                    "symbol=%s is already up to date. Skipping.",
                    asset.symbol,
                )
                return IngestionResult(symbol=asset.symbol, success=True, rows_written=0)

            request = YFinanceRequest(
                symbol=asset.symbol,
                asset_class=asset.asset_class,
                exchange=asset.exchange,
                currency=asset.currency,
                start_date=start_date,
                end_date=end_date,
                timeframe=asset.timeframe,
            )

            df = self.connector.fetch_history(request)

            write_mode = "overwrite" if self.mode == "full" else "append"

            self.writer.write(
                df,
                layer=DataLayer.RAW,
                asset_class=asset.asset_class,
                symbol=asset.symbol,
                timeframe=asset.timeframe,
                mode=write_mode,
            )

            logger.info(
                "Ingested symbol=%s rows=%s",
                asset.symbol,
                len(df),
            )

            return IngestionResult(
                symbol=asset.symbol,
                success=True,
                rows_written=len(df),
            )

        except (YFinanceConnectorError, ParquetWriterError) as exc:
            logger.exception(
                "Ingestion failed for symbol=%s error=%s",
                asset.symbol,
                exc,
            )
            return IngestionResult(
                symbol=asset.symbol,
                success=False,
                error=str(exc),
            )

    def _resolve_date_range(
        self,
        asset: AssetConfig,
    ) -> tuple[str | None, str | None]:
        """
        Resolve start and end dates for the fetch request.

        In full mode: always fetch from DATA_START_DATE to today.
        In incremental mode: fetch from last stored date + 1 day to today.
        If data is already current in incremental mode, returns (None, None).

        Args:
            asset: AssetConfig for the asset.

        Returns:
            Tuple of (start_date, end_date) strings in YYYY-MM-DD format,
            or (None, None) if no fetch is needed.
        """
        today = date.today().isoformat()

        if self.mode == "full":
            return self.start_date, today

        # incremental mode
        if not self.writer.exists(
            layer=DataLayer.RAW,
            asset_class=asset.asset_class,
            symbol=asset.symbol,
            timeframe=asset.timeframe,
        ):
            logger.info(
                "No existing data for symbol=%s. Running full fetch.",
                asset.symbol,
            )
            return self.start_date, today

        existing_df = self.writer.read(
            layer=DataLayer.RAW,
            asset_class=asset.asset_class,
            symbol=asset.symbol,
            timeframe=asset.timeframe,
        )

        if existing_df.empty:
            return self.start_date, today

        last_date = pd.to_datetime(existing_df["date"]).max().date()

        if last_date >= date.today():
            return None, None

        incremental_start = (
            pd.Timestamp(last_date) + pd.Timedelta(days=1)
        ).date().isoformat()

        logger.info(
            "Incremental fetch for symbol=%s from=%s to=%s",
            asset.symbol,
            incremental_start,
            today,
        )

        return incremental_start, today