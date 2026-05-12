"""
Unit tests for IngestionPipeline.

Tests cover:
- successful single asset ingestion
- successful multi-asset ingestion
- failed fetch is recorded and skipped
- all assets failing raises no exception but report reflects failures
- empty asset list raises IngestionPipelineError
- incremental mode skips up-to-date assets
- full mode always fetches from start date
- ingestion report counts are correct
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data_pipeline.ingestion import (
    AssetConfig,
    IngestionPipeline,
    IngestionPipelineError,
    IngestionReport,
    IngestionResult,
)
from src.data_pipeline.connectors.yfinance_connector import YFinanceConnectorError
from src.data_storage.parquet_writer import DataLayer


# ── Fixtures ──────────────────────────────────────────────────


def make_asset() -> AssetConfig:
    return AssetConfig(
        symbol="RELIANCE.NS",
        asset_class="equity",
        exchange="NSE",
        currency="INR",
    )


def make_raw_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["RELIANCE.NS", "RELIANCE.NS"],
            "asset_class": ["equity", "equity"],
            "exchange": ["NSE", "NSE"],
            "currency": ["INR", "INR"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
            "open": [100.0, 101.0],
            "high": [105.0, 106.0],
            "low": [99.0, 100.0],
            "close": [103.0, 104.0],
            "adj_close": [103.0, 104.0],
            "volume": [1000.0, 2000.0],
            "source": ["yfinance", "yfinance"],
        }
    )


# ── Tests ─────────────────────────────────────────────────────


class TestIngestionPipelineEmptyUniverse:
    def test_empty_asset_list_raises(self) -> None:
        pipeline = IngestionPipeline()
        with pytest.raises(IngestionPipelineError):
            pipeline.run([])


class TestIngestionPipelineSuccessful:
    def test_single_asset_success(self) -> None:
        mock_connector = MagicMock()
        mock_writer = MagicMock()

        mock_connector.fetch_history.return_value = make_raw_df()
        mock_writer.exists.return_value = False
        mock_writer.write.return_value = None

        pipeline = IngestionPipeline(
            connector=mock_connector,
            writer=mock_writer,
        )
        pipeline.mode = "full"

        report = pipeline.run([make_asset()])

        assert report.total == 1
        assert report.succeeded == 1
        assert report.failed == 0

    def test_single_asset_result_has_rows_written(self) -> None:
        mock_connector = MagicMock()
        mock_writer = MagicMock()

        mock_connector.fetch_history.return_value = make_raw_df()
        mock_writer.exists.return_value = False
        mock_writer.write.return_value = None

        pipeline = IngestionPipeline(
            connector=mock_connector,
            writer=mock_writer,
        )
        pipeline.mode = "full"

        report = pipeline.run([make_asset()])
        assert report.results[0].rows_written == len(make_raw_df())

    def test_multiple_assets_all_succeed(self) -> None:
        mock_connector = MagicMock()
        mock_writer = MagicMock()

        mock_connector.fetch_history.return_value = make_raw_df()
        mock_writer.exists.return_value = False

        pipeline = IngestionPipeline(
            connector=mock_connector,
            writer=mock_writer,
        )
        pipeline.mode = "full"

        assets = [
            make_asset(),
            AssetConfig(
                symbol="TCS.NS",
                asset_class="equity",
                exchange="NSE",
                currency="INR",
            ),
        ]

        report = pipeline.run(assets)
        assert report.total == 2
        assert report.succeeded == 2
        assert report.failed == 0


class TestIngestionPipelineFailures:
    def test_single_asset_fetch_failure_recorded(self) -> None:
        mock_connector = MagicMock()
        mock_writer = MagicMock()

        mock_connector.fetch_history.side_effect = YFinanceConnectorError(
            "API timeout"
        )
        mock_writer.exists.return_value = False

        pipeline = IngestionPipeline(
            connector=mock_connector,
            writer=mock_writer,
        )
        pipeline.mode = "full"

        report = pipeline.run([make_asset()])

        assert report.total == 1
        assert report.succeeded == 0
        assert report.failed == 1
        assert report.results[0].success is False
        assert report.results[0].error is not None

    def test_partial_failure_counts_correct(self) -> None:
        mock_connector = MagicMock()
        mock_writer = MagicMock()

        mock_connector.fetch_history.side_effect = [
            make_raw_df(),
            YFinanceConnectorError("failed"),
        ]
        mock_writer.exists.return_value = False

        pipeline = IngestionPipeline(
            connector=mock_connector,
            writer=mock_writer,
        )
        pipeline.mode = "full"

        assets = [
            make_asset(),
            AssetConfig(
                symbol="TCS.NS",
                asset_class="equity",
                exchange="NSE",
                currency="INR",
            ),
        ]

        report = pipeline.run(assets)
        assert report.succeeded == 1
        assert report.failed == 1


class TestIngestionPipelineIncrementalMode:
    def test_incremental_skips_up_to_date_asset(self) -> None:
        mock_connector = MagicMock()
        mock_writer = MagicMock()

        existing_df = make_raw_df().copy()
        existing_df["date"] = pd.to_datetime(
            [date.today().isoformat(), date.today().isoformat()], utc=True
        )

        mock_writer.exists.return_value = True
        mock_writer.read.return_value = existing_df

        pipeline = IngestionPipeline(
            connector=mock_connector,
            writer=mock_writer,
        )
        pipeline.mode = "incremental"

        report = pipeline.run([make_asset()])

        mock_connector.fetch_history.assert_not_called()
        assert report.succeeded == 1
        assert report.results[0].rows_written == 0

    def test_incremental_fetches_when_no_existing_data(self) -> None:
        mock_connector = MagicMock()
        mock_writer = MagicMock()

        mock_connector.fetch_history.return_value = make_raw_df()
        mock_writer.exists.return_value = False

        pipeline = IngestionPipeline(
            connector=mock_connector,
            writer=mock_writer,
        )
        pipeline.mode = "incremental"

        report = pipeline.run([make_asset()])

        mock_connector.fetch_history.assert_called_once()
        assert report.succeeded == 1


class TestIngestionPipelineFullMode:
    def test_full_mode_always_fetches(self) -> None:
        mock_connector = MagicMock()
        mock_writer = MagicMock()

        mock_connector.fetch_history.return_value = make_raw_df()
        mock_writer.exists.return_value = True

        pipeline = IngestionPipeline(
            connector=mock_connector,
            writer=mock_writer,
        )
        pipeline.mode = "full"

        pipeline.run([make_asset()])

        mock_connector.fetch_history.assert_called_once()

    def test_full_mode_uses_overwrite(self) -> None:
        mock_connector = MagicMock()
        mock_writer = MagicMock()

        mock_connector.fetch_history.return_value = make_raw_df()
        mock_writer.exists.return_value = False

        pipeline = IngestionPipeline(
            connector=mock_connector,
            writer=mock_writer,
        )
        pipeline.mode = "full"

        pipeline.run([make_asset()])

        call_kwargs = mock_writer.write.call_args.kwargs
        assert call_kwargs["mode"] == "overwrite"


class TestIngestionReport:
    def test_report_add_success(self) -> None:
        report = IngestionReport()
        report.add(IngestionResult(symbol="A.NS", success=True, rows_written=10))
        assert report.total == 1
        assert report.succeeded == 1
        assert report.failed == 0

    def test_report_add_failure(self) -> None:
        report = IngestionReport()
        report.add(
            IngestionResult(symbol="A.NS", success=False, error="timeout")
        )
        assert report.total == 1
        assert report.succeeded == 0
        assert report.failed == 1