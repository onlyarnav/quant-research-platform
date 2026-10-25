"""
Tests for the FeaturePipeline orchestration.

Tests processing of market data files into feature parquet files,
error handling for corrupt files, and pipeline summary reporting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import settings
from src.data_storage.parquet_writer import LAYER_PATH_MAP, DataLayer
from src.feature_engineering.feature_pipeline import FeaturePipeline


@pytest.fixture
def patched_layers(tmp_path, monkeypatch):
    """Patch processed and features directories to temporary paths."""
    proc_dir = tmp_path / "data" / "processed"
    feat_dir = tmp_path / "data" / "features"
    proc_dir.mkdir(parents=True, exist_ok=True)
    feat_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(settings, "DATA_PROCESSED_PATH", proc_dir)
    monkeypatch.setattr(settings, "DATA_FEATURES_PATH", feat_dir)
    monkeypatch.setitem(LAYER_PATH_MAP, DataLayer.PROCESSED, proc_dir)
    monkeypatch.setitem(LAYER_PATH_MAP, DataLayer.FEATURES, feat_dir)

    return proc_dir, feat_dir


def _create_valid_processed_file(file_path, symbol: str, n_rows: int = 40) -> None:
    """Helper to write a valid processed market parquet file."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=n_rows, freq="B")
    base = 100.0 + np.cumsum(rng.standard_normal(n_rows))
    high = base + 1.0
    low = base - 1.0
    close = base
    volume = np.full(n_rows, 10_000.0)

    df = pd.DataFrame(
        {
            "symbol": symbol,
            "date": dates,
            "open": base,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,
            "volume": volume,
            "returns": np.zeros(n_rows),
            "log_returns": np.zeros(n_rows),
        }
    )
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(file_path, index=False)


# =====================================================================
# FeaturePipeline Tests
# =====================================================================


def test_feature_pipeline_empty_directory(patched_layers) -> None:
    """Assert empty processed directory returns a zeroed report."""
    pipeline = FeaturePipeline()
    report = pipeline.run()
    assert report.total == 0
    assert report.succeeded == 0
    assert report.failed == 0


def test_feature_pipeline_processes_files(patched_layers) -> None:
    """Assert valid processed files are engineered and written to features layer."""
    proc_dir, feat_dir = patched_layers

    # Write two valid files under equity
    f1 = proc_dir / "equity" / "equity_AAPL_daily.parquet"
    f2 = proc_dir / "equity" / "equity_MSFT_daily.parquet"
    _create_valid_processed_file(f1, "AAPL", 40)
    _create_valid_processed_file(f2, "MSFT", 40)

    pipeline = FeaturePipeline()
    report = pipeline.run()

    assert report.total == 2
    assert report.succeeded == 2
    assert report.failed == 0
    assert report.failed_files == []

    # Check output files exist in features layer
    out1 = feat_dir / "equity" / "equity_AAPL_daily.parquet"
    out2 = feat_dir / "equity" / "equity_MSFT_daily.parquet"
    assert out1.exists()
    assert out2.exists()

    df_out = pd.read_parquet(out1)
    assert "sma_5" in df_out.columns
    assert "future_return_1d" in df_out.columns
    assert len(df_out) == 40


def test_feature_pipeline_handles_corrupt_file(patched_layers) -> None:
    """Assert a corrupt or empty file increments failed count without stopping other files."""
    proc_dir, feat_dir = patched_layers

    # One valid file, one corrupt/empty file
    f1 = proc_dir / "equity" / "equity_GOOD_daily.parquet"
    _create_valid_processed_file(f1, "GOOD", 40)

    corrupt_file = proc_dir / "equity" / "equity_BAD_daily.parquet"
    corrupt_file.parent.mkdir(parents=True, exist_ok=True)
    # Write invalid / empty dataframe
    pd.DataFrame().to_parquet(corrupt_file, index=False)

    pipeline = FeaturePipeline()
    report = pipeline.run()

    assert report.total == 2
    assert report.succeeded == 1
    assert report.failed == 1
    assert "equity_BAD_daily.parquet" in report.failed_files

    # The good file was still generated
    assert (feat_dir / "equity" / "equity_GOOD_daily.parquet").exists()
