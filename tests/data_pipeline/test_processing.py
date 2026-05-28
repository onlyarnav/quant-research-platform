"""
Unit tests for ProcessingPipeline.

Tests cover:
- successful single asset processing
- successful multi-asset processing
- missing raw data is recorded as failure
- validation failure is recorded and skipped
- cleaning failure is recorded and skipped
- empty config list raises ProcessingPipelineError
- processing report counts are correct
- processed data is written to processed layer
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data_pipeline.processing import (
    ProcessingConfig,
    ProcessingPipeline,
    ProcessingPipelineError,
    ProcessingReport,
    ProcessingResult,
)
from src.data_pipeline.validation import ValidationResult
from src.data_pipeline.cleaning import DataCleaningError


# ── Fixtures ──────────────────────────────────────────────────


def make_config() -> ProcessingConfig:
    return ProcessingConfig(
        symbol="RELIANCE.NS",
        asset_class="equity",
    )


def make_raw_df() -> pd.DataFrame:
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


def make_processed_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["RELIANCE.NS"] * 4,
            "asset_class": ["equity"] * 4,
            "exchange": ["NSE"] * 4,
            "currency": ["INR"] * 4,
            "date": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
                utc=True,
            ),
            "open": [102.0, 104.0, 106.0, 108.0],
            "high": [107.0, 109.0, 111.0, 113.0],
            "low": [101.0, 103.0, 105.0, 107.0],
            "close": [105.0, 107.0, 109.0, 111.0],
            "adj_close": [105.0, 107.0, 109.0, 111.0],
            "volume": [2000.0, 3000.0, 4000.0, 5000.0],
            "returns": [0.019, 0.019, 0.019, 0.019],
            "log_returns": [0.018, 0.018, 0.018, 0.018],
        }
    )


def make_valid_validation_result() -> ValidationResult:
    result = ValidationResult()
    result.rows_total = 5
    return result


def make_invalid_validation_result() -> ValidationResult:
    result = ValidationResult()
    result.add_error("high < low found in 1 row(s).")
    return result


# ── Tests ─────────────────────────────────────────────────────


class TestProcessingPipelineEmptyConfig:
    def test_empty_config_list_raises(self) -> None:
        pipeline = ProcessingPipeline()
        with pytest.raises(ProcessingPipelineError):
            pipeline.run([])


class TestProcessingPipelineSuccessful:
    def test_single_asset_success(self) -> None:
        mock_validator = MagicMock()
        mock_cleaner = MagicMock()
        mock_writer = MagicMock()

        mock_writer.exists.return_value = True
        mock_writer.read.return_value = make_raw_df()
        mock_validator.validate.return_value = make_valid_validation_result()
        mock_cleaner.clean.return_value = make_processed_df()
        mock_writer.write.return_value = None

        pipeline = ProcessingPipeline(
            validator=mock_validator,
            cleaner=mock_cleaner,
            writer=mock_writer,
        )

        report = pipeline.run([make_config()])

        assert report.total == 1
        assert report.succeeded == 1
        assert report.failed == 0

    def test_single_asset_rows_written_correct(self) -> None:
        mock_validator = MagicMock()
        mock_cleaner = MagicMock()
        mock_writer = MagicMock()

        processed = make_processed_df()
        mock_writer.exists.return_value = True
        mock_writer.read.return_value = make_raw_df()
        mock_validator.validate.return_value = make_valid_validation_result()
        mock_cleaner.clean.return_value = processed

        pipeline = ProcessingPipeline(
            validator=mock_validator,
            cleaner=mock_cleaner,
            writer=mock_writer,
        )

        report = pipeline.run([make_config()])
        assert report.results[0].rows_written == len(processed)

    def test_multiple_assets_all_succeed(self) -> None:
        mock_validator = MagicMock()
        mock_cleaner = MagicMock()
        mock_writer = MagicMock()

        mock_writer.exists.return_value = True
        mock_writer.read.return_value = make_raw_df()
        mock_validator.validate.return_value = make_valid_validation_result()
        mock_cleaner.clean.return_value = make_processed_df()

        pipeline = ProcessingPipeline(
            validator=mock_validator,
            cleaner=mock_cleaner,
            writer=mock_writer,
        )

        configs = [
            make_config(),
            ProcessingConfig(symbol="TCS.NS", asset_class="equity"),
        ]

        report = pipeline.run(configs)
        assert report.total == 2
        assert report.succeeded == 2
        assert report.failed == 0


class TestProcessingPipelineMissingRawData:
    def test_missing_raw_data_recorded_as_failure(self) -> None:
        mock_validator = MagicMock()
        mock_cleaner = MagicMock()
        mock_writer = MagicMock()

        mock_writer.exists.return_value = False

        pipeline = ProcessingPipeline(
            validator=mock_validator,
            cleaner=mock_cleaner,
            writer=mock_writer,
        )

        report = pipeline.run([make_config()])

        assert report.failed == 1
        assert report.results[0].success is False
        assert report.results[0].error is not None


class TestProcessingPipelineValidationFailure:
    def test_validation_failure_recorded_as_failure(self) -> None:
        mock_validator = MagicMock()
        mock_cleaner = MagicMock()
        mock_writer = MagicMock()

        mock_writer.exists.return_value = True
        mock_writer.read.return_value = make_raw_df()
        mock_validator.validate.return_value = make_invalid_validation_result()

        pipeline = ProcessingPipeline(
            validator=mock_validator,
            cleaner=mock_cleaner,
            writer=mock_writer,
        )

        report = pipeline.run([make_config()])

        assert report.failed == 1
        assert report.results[0].success is False
        mock_cleaner.clean.assert_not_called()

    def test_validation_failure_does_not_stop_other_assets(self) -> None:
        mock_validator = MagicMock()
        mock_cleaner = MagicMock()
        mock_writer = MagicMock()

        mock_writer.exists.return_value = True
        mock_writer.read.return_value = make_raw_df()
        mock_validator.validate.side_effect = [
            make_invalid_validation_result(),
            make_valid_validation_result(),
        ]
        mock_cleaner.clean.return_value = make_processed_df()

        pipeline = ProcessingPipeline(
            validator=mock_validator,
            cleaner=mock_cleaner,
            writer=mock_writer,
        )

        configs = [
            make_config(),
            ProcessingConfig(symbol="TCS.NS", asset_class="equity"),
        ]

        report = pipeline.run(configs)
        assert report.failed == 1
        assert report.succeeded == 1


class TestProcessingPipelineCleaningFailure:
    def test_cleaning_failure_recorded_as_failure(self) -> None:
        mock_validator = MagicMock()
        mock_cleaner = MagicMock()
        mock_writer = MagicMock()

        mock_writer.exists.return_value = True
        mock_writer.read.return_value = make_raw_df()
        mock_validator.validate.return_value = make_valid_validation_result()
        mock_cleaner.clean.side_effect = DataCleaningError(
            "All rows removed during cleaning."
        )

        pipeline = ProcessingPipeline(
            validator=mock_validator,
            cleaner=mock_cleaner,
            writer=mock_writer,
        )

        report = pipeline.run([make_config()])

        assert report.failed == 1
        assert report.results[0].success is False


class TestProcessingPipelineWritesProcessedLayer:
    def test_write_called_with_processed_layer(self) -> None:
        mock_validator = MagicMock()
        mock_cleaner = MagicMock()
        mock_writer = MagicMock()

        mock_writer.exists.return_value = True
        mock_writer.read.return_value = make_raw_df()
        mock_validator.validate.return_value = make_valid_validation_result()
        mock_cleaner.clean.return_value = make_processed_df()

        pipeline = ProcessingPipeline(
            validator=mock_validator,
            cleaner=mock_cleaner,
            writer=mock_writer,
        )

        pipeline.run([make_config()])

        call_kwargs = mock_writer.write.call_args.kwargs
        from src.data_storage.parquet_writer import DataLayer
        assert call_kwargs["layer"] == DataLayer.PROCESSED

    def test_write_called_with_overwrite_mode(self) -> None:
        mock_validator = MagicMock()
        mock_cleaner = MagicMock()
        mock_writer = MagicMock()

        mock_writer.exists.return_value = True
        mock_writer.read.return_value = make_raw_df()
        mock_validator.validate.return_value = make_valid_validation_result()
        mock_cleaner.clean.return_value = make_processed_df()

        pipeline = ProcessingPipeline(
            validator=mock_validator,
            cleaner=mock_cleaner,
            writer=mock_writer,
        )

        pipeline.run([make_config()])

        call_kwargs = mock_writer.write.call_args.kwargs
        assert call_kwargs["mode"] == "overwrite"


class TestProcessingReport:
    def test_report_add_success(self) -> None:
        report = ProcessingReport()
        report.add(
            ProcessingResult(symbol="A.NS", success=True, rows_written=100)
        )
        assert report.total == 1
        assert report.succeeded == 1
        assert report.failed == 0

    def test_report_add_failure(self) -> None:
        report = ProcessingReport()
        report.add(
            ProcessingResult(symbol="A.NS", success=False, error="failed")
        )
        assert report.total == 1
        assert report.succeeded == 0
        assert report.failed == 1