"""Unit tests for SignalPipeline extensions (discover_symbols and write_mode)."""

from __future__ import annotations

import pandas as pd

from config.settings import settings
from src.data_storage.parquet_writer import LAYER_PATH_MAP, DataLayer, ParquetWriter
from src.signals.signal_pipeline import SignalPipeline
from src.signals.topn_signal import TopNLongShortSignalGenerator

# ==============================================================================
# discover_symbols()
# ==============================================================================


def test_discover_symbols_empty_when_directory_missing(tmp_path, monkeypatch) -> None:
    """Assert discover_symbols returns empty list when prediction directory does not exist."""
    missing_dir = tmp_path / "nonexistent_predictions"
    monkeypatch.setattr(settings, "DATA_PREDICTIONS_PATH", missing_dir)
    monkeypatch.setitem(LAYER_PATH_MAP, DataLayer.PREDICTIONS, missing_dir)

    pipeline = SignalPipeline()
    symbols = pipeline.discover_symbols(asset_class="equity")
    assert symbols == []


def test_discover_symbols_success(tmp_path, monkeypatch) -> None:
    """Assert discover_symbols extracts sorted unique symbols from parquet files."""
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "DATA_PREDICTIONS_PATH", pred_dir)
    monkeypatch.setitem(LAYER_PATH_MAP, DataLayer.PREDICTIONS, pred_dir)

    writer = ParquetWriter()
    df = pd.DataFrame(
        {
            "symbol": ["INFY"],
            "date": [pd.Timestamp("2023-01-01")],
            "predicted_return": [0.05],
            "model_name": ["model"],
            "model_version": ["v1"],
        }
    )

    writer.write(
        df,
        layer=DataLayer.PREDICTIONS,
        asset_class="equity",
        symbol="INFY",
        timeframe="daily",
    )
    writer.write(
        df.assign(symbol="TCS"),
        layer=DataLayer.PREDICTIONS,
        asset_class="equity",
        symbol="TCS",
        timeframe="daily",
    )
    writer.write(
        df.assign(symbol="RELIANCE"),
        layer=DataLayer.PREDICTIONS,
        asset_class="equity",
        symbol="RELIANCE",
        timeframe="daily",
    )

    pipeline = SignalPipeline(writer=writer)
    symbols = pipeline.discover_symbols(asset_class="equity")
    assert symbols == ["INFY", "RELIANCE", "TCS"]


# ==============================================================================
# write_mode in SignalPipeline.run()
# ==============================================================================


def test_signal_pipeline_write_mode(tmp_path, monkeypatch, mocker) -> None:
    """Assert write_mode argument is forwarded to writer.write()."""
    pred_dir = tmp_path / "predictions"
    sig_dir = tmp_path / "signals"
    pred_dir.mkdir(parents=True, exist_ok=True)
    sig_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(settings, "DATA_PREDICTIONS_PATH", pred_dir)
    monkeypatch.setitem(LAYER_PATH_MAP, DataLayer.PREDICTIONS, pred_dir)
    monkeypatch.setattr(settings, "DATA_SIGNALS_PATH", sig_dir)
    monkeypatch.setitem(LAYER_PATH_MAP, DataLayer.SIGNALS, sig_dir)

    writer = ParquetWriter()
    df = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "date": [pd.Timestamp("2023-01-01")],
            "predicted_return": [0.05],
            "model_name": ["model"],
            "model_version": ["v1"],
        }
    )
    writer.write(
        df,
        layer=DataLayer.PREDICTIONS,
        asset_class="equity",
        symbol="AAPL",
        timeframe="daily",
    )

    spy_write = mocker.spy(writer, "write")
    pipeline = SignalPipeline(writer=writer)
    gen = TopNLongShortSignalGenerator(top_n=1)

    pipeline.run(
        generator=gen,
        symbols=["AAPL"],
        asset_class="equity",
        write_mode="append",
    )

    assert spy_write.call_count >= 1
    # Check that mode='append' was passed
    call_kwargs = spy_write.call_args[1]
    assert call_kwargs["mode"] == "append"
