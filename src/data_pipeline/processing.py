"""
Data processing pipeline.

Orchestrates the full raw → processed data promotion workflow:
1. Read raw parquet data from data lake
2. Validate against schema rules
3. Clean and compute derived columns
4. Write processed parquet to processed data layer

This module is the single entry point for promoting raw data
to the processed layer. Downstream feature engineering reads
exclusively from the processed layer.

Usage:
    from src.data_pipeline.processing import ProcessingPipeline, ProcessingConfig

    pipeline = ProcessingPipeline()
    pipeline.run([
        ProcessingConfig(symbol="RELIANCE.NS", asset_class="equity"),
    ])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from config.constants import TIMEFRAME_DAILY
from src.data_pipeline.cleaning import DataCleaner, DataCleaningError
from src.data_pipeline.validation import DataValidator, DataValidationError
from src.data_storage.parquet_writer import DataLayer, ParquetWriter, ParquetWriterError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProcessingConfig:
    """
    Configuration for processing a single asset.

    Attributes:
        symbol: Trading symbol.
        asset_class: Platform asset class.
        timeframe: Data timeframe. Defaults to daily.
    """

    symbol: str
    asset_class: str
    timeframe: str = TIMEFRAME_DAILY


@dataclass
class ProcessingResult:
    """
    Result of a single asset processing attempt.

    Attributes:
        symbol: Asset symbol.
        success: Whether processing succeeded.
        rows_written: Number of rows written to processed layer.
        error: Error message if processing failed.
    """

    symbol: str
    success: bool
    rows_written: int = 0
    error: str | None = None


@dataclass
class ProcessingReport:
    """
    Aggregate report for a full processing pipeline run.

    Attributes:
        total: Total assets attempted.
        succeeded: Successfully processed assets.
        failed: Failed assets.
        results: Per-asset processing results.
    """

    total: int = 0
    succeeded: int = 0
    failed: int = 0
    results: list[ProcessingResult] = field(default_factory=list)

    def add(self, result: ProcessingResult) -> None:
        self.results.append(result)
        self.total += 1
        if result.success:
            self.succeeded += 1
        else:
            self.failed += 1


class ProcessingPipelineError(Exception):
    """Raised when the entire processing pipeline fails."""


class ProcessingPipeline:
    """
    Orchestrates raw → processed data promotion.

    For each asset:
    - reads raw parquet from data lake
    - validates schema and data quality
    - cleans and computes returns
    - writes processed parquet to processed layer

    If validation fails, the asset is skipped and error is recorded.
    Pipeline does not stop on individual asset failures.
    """

    def __init__(
        self,
        validator: DataValidator | None = None,
        cleaner: DataCleaner | None = None,
        writer: ParquetWriter | None = None,
    ) -> None:
        """
        Initialize the processing pipeline.

        Args:
            validator: DataValidator instance.
            cleaner: DataCleaner instance.
            writer: ParquetWriter instance.
        """
        self.validator = validator or DataValidator()
        self.cleaner = cleaner or DataCleaner()
        self.writer = writer or ParquetWriter()

    def run(self, configs: list[ProcessingConfig]) -> ProcessingReport:
        """
        Run processing for a list of asset configs.

        Args:
            configs: List of ProcessingConfig objects.

        Returns:
            ProcessingReport summarizing results.

        Raises:
            ProcessingPipelineError: If configs list is empty.
        """
        if not configs:
            raise ProcessingPipelineError(
                "Config list is empty. Nothing to process."
            )

        logger.info(
            "Starting processing pipeline. assets=%s",
            len(configs),
        )

        report = ProcessingReport()

        for config in configs:
            result = self._process_asset(config)
            report.add(result)

        logger.info(
            "Processing pipeline complete. total=%s succeeded=%s failed=%s",
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

    def _process_asset(self, config: ProcessingConfig) -> ProcessingResult:
        """
        Process a single asset through validation, cleaning, and storage.

        Args:
            config: ProcessingConfig for the asset.

        Returns:
            ProcessingResult for this asset.
        """
        logger.info(
            "Processing symbol=%s asset_class=%s",
            config.symbol,
            config.asset_class,
        )

        try:
            if not self.writer.exists(
                layer=DataLayer.RAW,
                asset_class=config.asset_class,
                symbol=config.symbol,
                timeframe=config.timeframe,
            ):
                raise ProcessingPipelineError(
                    f"Raw data not found for symbol={config.symbol}. "
                    "Run ingestion pipeline first."
                )

            raw_df = self.writer.read(
                layer=DataLayer.RAW,
                asset_class=config.asset_class,
                symbol=config.symbol,
                timeframe=config.timeframe,
            )

            logger.info(
                "Read %s raw rows for symbol=%s",
                len(raw_df),
                config.symbol,
            )

            validation_result = self.validator.validate(raw_df)

            if not validation_result.is_valid:
                error_summary = " | ".join(validation_result.errors)
                raise DataValidationError(
                    f"Validation failed for symbol={config.symbol}: "
                    f"{error_summary}"
                )

            if validation_result.warnings:
                for warning in validation_result.warnings:
                    logger.warning(
                        "symbol=%s validation warning: %s",
                        config.symbol,
                        warning,
                    )

            processed_df = self.cleaner.clean(raw_df)

            self.writer.write(
                processed_df,
                layer=DataLayer.PROCESSED,
                asset_class=config.asset_class,
                symbol=config.symbol,
                timeframe=config.timeframe,
                mode="overwrite",
            )

            logger.info(
                "Processed symbol=%s rows_written=%s",
                config.symbol,
                len(processed_df),
            )

            return ProcessingResult(
                symbol=config.symbol,
                success=True,
                rows_written=len(processed_df),
            )

        except (
            DataValidationError,
            DataCleaningError,
            ParquetWriterError,
            ProcessingPipelineError,
        ) as exc:
            logger.exception(
                "Processing failed for symbol=%s error=%s",
                config.symbol,
                exc,
            )
            return ProcessingResult(
                symbol=config.symbol,
                success=False,
                error=str(exc),
            )