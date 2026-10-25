"""
Extended tests for PredictionPipeline: timezone preservation and append write mode.
"""

from __future__ import annotations

import pandas as pd
import pytest

from config.settings import settings
from src.data_storage.parquet_writer import LAYER_PATH_MAP, DataLayer, ParquetWriter
from src.models.linear_model import LinearModel
from src.models.prediction_pipeline import PredictionPipeline


@pytest.fixture
def patched_predictions_layer(tmp_path, monkeypatch):
    """Patch predictions directory to tmp_path."""
    pred_dir = tmp_path / "data" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "DATA_PREDICTIONS_PATH", pred_dir)
    monkeypatch.setitem(LAYER_PATH_MAP, DataLayer.PREDICTIONS, pred_dir)
    return pred_dir


def test_timezone_aware_dates_preserved_exactly(patched_predictions_layer) -> None:
    """Assert timezone-aware dates retain exact tz dtype without shift in output and parquet."""
    tz = "Asia/Kolkata"
    dates = pd.date_range("2023-01-01", periods=5, freq="D", tz=tz)
    df = pd.DataFrame(
        {
            "symbol": ["AAPL"] * 5,
            "date": dates,
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    model = LinearModel()
    model.fit(df[["f1"]], pd.Series([0.1, 0.2, 0.3, 0.4, 0.5]))

    pipeline = PredictionPipeline()
    res = pipeline.run(model, df, ["f1"], model_version="v1", asset_class="equity")

    # In-memory output Series has same dtype and timezone
    assert res["date"].dtype == df["date"].dtype
    assert str(res["date"].dt.tz) == tz
    pd.testing.assert_series_equal(res["date"], df["date"])

    # Written file has same dates
    writer = ParquetWriter()
    read_back = writer.read(
        layer=DataLayer.PREDICTIONS,
        asset_class="equity",
        symbol="AAPL",
        timeframe="daily",
    )
    assert read_back["date"].dtype == df["date"].dtype
    pd.testing.assert_series_equal(read_back["date"], df["date"])


def test_write_mode_append(patched_predictions_layer) -> None:
    """Assert write_mode='append' retains earlier prediction rows for the symbol."""
    dates1 = pd.date_range("2023-01-01", periods=3, freq="D")
    df1 = pd.DataFrame({"symbol": ["AAPL"] * 3, "date": dates1, "f1": [1.0, 2.0, 3.0]})

    dates2 = pd.date_range("2023-01-04", periods=2, freq="D")
    df2 = pd.DataFrame({"symbol": ["AAPL"] * 2, "date": dates2, "f1": [4.0, 5.0]})

    model = LinearModel()
    model.fit(df1[["f1"]], pd.Series([0.1, 0.2, 0.3]))

    pipeline = PredictionPipeline()
    # First write
    pipeline.run(
        model,
        df1,
        ["f1"],
        model_version="v1",
        asset_class="equity",
        write_mode="overwrite",
    )

    # Second write with append
    pipeline.run(
        model,
        df2,
        ["f1"],
        model_version="v1",
        asset_class="equity",
        write_mode="append",
    )

    writer = ParquetWriter()
    combined = writer.read(
        layer=DataLayer.PREDICTIONS,
        asset_class="equity",
        symbol="AAPL",
        timeframe="daily",
    )
    assert len(combined) == 5
