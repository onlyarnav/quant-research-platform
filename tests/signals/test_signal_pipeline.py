"""
Tests for SignalPipeline.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_storage.parquet_writer import DataLayer, ParquetWriter
from src.signals.signal_pipeline import SignalPipeline
from src.signals.threshold_signal import ThresholdSignalGenerator


@pytest.fixture
def patched_writer(monkeypatch, tmp_path):
    """
    Real ParquetWriter pointed at tmp_path for both predictions
    and signals layers, avoiding any real data lake writes.
    """
    from src.data_storage import parquet_writer as pw_module

    predictions_path = tmp_path / "predictions"
    signals_path = tmp_path / "signals"
    predictions_path.mkdir()
    signals_path.mkdir()

    monkeypatch.setattr(pw_module.settings, "DATA_PREDICTIONS_PATH", predictions_path)
    monkeypatch.setattr(pw_module.settings, "DATA_SIGNALS_PATH", signals_path)
    monkeypatch.setitem(pw_module.LAYER_PATH_MAP, pw_module.DataLayer.PREDICTIONS, predictions_path)
    monkeypatch.setitem(pw_module.LAYER_PATH_MAP, pw_module.DataLayer.SIGNALS, signals_path)

    return pw_module.ParquetWriter(), predictions_path, signals_path


def _seed_predictions(writer: ParquetWriter, symbol: str, asset_class: str) -> None:
    """Helper to write a synthetic prediction file for a symbol."""
    df = pd.DataFrame({
        "symbol": [symbol, symbol],
        "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "predicted_return": [0.02, -0.02],
        "model_name": ["dummy_model", "dummy_model"],
        "model_version": ["v1.0", "v1.0"],
    })
    writer.write(
        df,
        layer=DataLayer.PREDICTIONS,
        asset_class=asset_class,
        symbol=symbol,
        timeframe="daily",
        mode="overwrite",
    )


# =============================================================================
# Input validation
# =============================================================================

def test_run_raises_on_empty_symbols_list(patched_writer):
    writer, _, _ = patched_writer
    pipeline = SignalPipeline(writer=writer)
    generator = ThresholdSignalGenerator(threshold=0.01)

    with pytest.raises(ValueError, match="symbols list cannot be empty"):
        pipeline.run(generator=generator, symbols=[], asset_class="equity")


def test_run_raises_when_no_prediction_files_found(patched_writer):
    writer, _, _ = patched_writer
    pipeline = SignalPipeline(writer=writer)
    generator = ThresholdSignalGenerator(threshold=0.01)

    with pytest.raises(ValueError, match="No prediction files found"):
        pipeline.run(generator=generator, symbols=["AAPL"], asset_class="equity")


# =============================================================================
# Core behavior
# =============================================================================

def test_run_reads_and_combines_multiple_symbols(patched_writer):
    writer, _, _ = patched_writer
    _seed_predictions(writer, "AAPL", "equity")
    _seed_predictions(writer, "MSFT", "equity")

    pipeline = SignalPipeline(writer=writer)
    generator = ThresholdSignalGenerator(threshold=0.01)

    signals_df = pipeline.run(generator=generator, symbols=["AAPL", "MSFT"], asset_class="equity")

    assert set(signals_df["symbol"].unique()) == {"AAPL", "MSFT"}
    assert len(signals_df) == 4  # 2 rows per symbol


def test_run_skips_missing_symbol_with_warning(patched_writer, caplog):
    writer, _, _ = patched_writer
    _seed_predictions(writer, "AAPL", "equity")

    pipeline = SignalPipeline(writer=writer)
    generator = ThresholdSignalGenerator(threshold=0.01)

    signals_df = pipeline.run(
        generator=generator,
        symbols=["AAPL", "MISSING"],
        asset_class="equity",
    )

    assert set(signals_df["symbol"].unique()) == {"AAPL"}
    assert any("No prediction file found for symbol=MISSING" in rec.message for rec in caplog.records)


def test_run_applies_generator_correctly(patched_writer):
    writer, _, _ = patched_writer
    _seed_predictions(writer, "AAPL", "equity")

    pipeline = SignalPipeline(writer=writer)
    generator = ThresholdSignalGenerator(threshold=0.01)

    signals_df = pipeline.run(generator=generator, symbols=["AAPL"], asset_class="equity")

    # predicted_return=[0.02, -0.02] with threshold=0.01 -> [BUY, SELL]
    assert list(signals_df["signal"]) == [1, -1]


def test_run_writes_one_signal_file_per_symbol(patched_writer):
    writer, _, signals_path = patched_writer
    _seed_predictions(writer, "AAPL", "equity")
    _seed_predictions(writer, "MSFT", "equity")

    pipeline = SignalPipeline(writer=writer)
    generator = ThresholdSignalGenerator(threshold=0.01)

    pipeline.run(generator=generator, symbols=["AAPL", "MSFT"], asset_class="equity")

    assert (signals_path / "equity" / "equity_AAPL_daily.parquet").exists()
    assert (signals_path / "equity" / "equity_MSFT_daily.parquet").exists()


def test_run_written_signal_file_content_matches_output(patched_writer):
    writer, _, signals_path = patched_writer
    _seed_predictions(writer, "AAPL", "equity")

    pipeline = SignalPipeline(writer=writer)
    generator = ThresholdSignalGenerator(threshold=0.01)

    signals_df = pipeline.run(generator=generator, symbols=["AAPL"], asset_class="equity")

    written_df = pd.read_parquet(signals_path / "equity" / "equity_AAPL_daily.parquet")

    pd.testing.assert_frame_equal(
        written_df.reset_index(drop=True),
        signals_df.reset_index(drop=True),
        check_dtype=False,
    )


def test_run_returns_correct_schema(patched_writer):
    writer, _, _ = patched_writer
    _seed_predictions(writer, "AAPL", "equity")

    pipeline = SignalPipeline(writer=writer)
    generator = ThresholdSignalGenerator(threshold=0.01)

    signals_df = pipeline.run(generator=generator, symbols=["AAPL"], asset_class="equity")

    expected_cols = {"symbol", "date", "predicted_return", "signal", "confidence", "rank"}
    assert set(signals_df.columns) == expected_cols