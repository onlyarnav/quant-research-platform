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

from datetime import date, timedelta

import pandas as pd

from config.settings import settings
from src.data_pipeline.connectors.yfinance_connector import (
    YFinanceConnector,
    YFinanceConnectorError,
    YFinanceRequest,
)
from src.data_pipeline.ingestion_models import (
    AssetConfig,
    IngestionPipelineError,
    IngestionReport,
    IngestionResult,
)
from src.data_storage.parquet_writer import DataLayer, ParquetWriter, ParquetWriterError
from src.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = [
    "IngestionPipeline",
    "AssetConfig",
    "IngestionResult",
    "IngestionReport",
    "IngestionPipelineError",
]


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
            failed_symbols = [r.symbol for r in report.results if not r.success]
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
                return IngestionResult(
                    symbol=asset.symbol, success=True, rows_written=0
                )

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
        end_date = (date.today() + timedelta(days=1)).isoformat()

        if self.mode == "full":
            return self.start_date, end_date

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
            return self.start_date, end_date

        existing_df = self.writer.read(
            layer=DataLayer.RAW,
            asset_class=asset.asset_class,
            symbol=asset.symbol,
            timeframe=asset.timeframe,
        )

        if existing_df.empty:
            return self.start_date, end_date

        last_date = pd.to_datetime(existing_df["date"]).max().date()

        if last_date >= date.today():
            return None, None

        incremental_start = (
            (pd.Timestamp(last_date) + pd.Timedelta(days=1)).date().isoformat()
        )

        logger.info(
            "Incremental fetch for symbol=%s from=%s to=%s",
            asset.symbol,
            incremental_start,
            end_date,
        )

        return incremental_start, end_date
