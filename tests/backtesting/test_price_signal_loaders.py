"""
Unit tests for PriceLoader and SignalLoader components.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from config.settings import settings
from src.backtesting.price_loader import PriceLoader
from src.backtesting.signal_loader import SignalLoader
from src.data_storage.parquet_writer import LAYER_PATH_MAP, DataLayer, ParquetWriter


def test_discovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test symbol discovery including symbols with underscores and missing folders."""
    writer = ParquetWriter()
    proc_dir = tmp_path / "processed"
    sig_dir = tmp_path / "signals"

    monkeypatch.setattr(settings, "DATA_PROCESSED_PATH", proc_dir)
    monkeypatch.setitem(LAYER_PATH_MAP, DataLayer.PROCESSED, proc_dir)
    monkeypatch.setattr(settings, "DATA_SIGNALS_PATH", sig_dir)
    monkeypatch.setitem(LAYER_PATH_MAP, DataLayer.SIGNALS, sig_dir)

    price_loader = PriceLoader(writer)
    signal_loader = SignalLoader(writer)

    # Missing folders return []
    assert price_loader.discover_symbols() == []
    assert signal_loader.discover_symbols() == []

    # Write files: normal, with underscore, and different timeframe
    dummy_px = pd.DataFrame(
        {
            "symbol": ["INFY"],
            "date": [pd.Timestamp("2024-01-01")],
            "close": [1500.0],
        }
    )
    writer.write(
        dummy_px,
        layer=DataLayer.PROCESSED,
        asset_class="equity",
        symbol="INFY",
        timeframe="daily",
    )
    writer.write(
        dummy_px,
        layer=DataLayer.PROCESSED,
        asset_class="equity",
        symbol="NIFTY_BANK",
        timeframe="daily",
    )
    writer.write(
        dummy_px,
        layer=DataLayer.PROCESSED,
        asset_class="equity",
        symbol="HOURLY_SYM",
        timeframe="hourly",
    )

    discovered = price_loader.discover_symbols()
    assert discovered == ["INFY", "NIFTY_BANK"]
    assert "HOURLY_SYM" not in discovered


def test_price_loader_load_and_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test PriceLoader load filters dates with naive and tz-aware bounds."""
    writer = ParquetWriter()
    proc_dir = tmp_path / "processed"
    monkeypatch.setattr(settings, "DATA_PROCESSED_PATH", proc_dir)
    monkeypatch.setitem(LAYER_PATH_MAP, DataLayer.PROCESSED, proc_dir)

    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {
            "symbol": ["RELIANCE"] * 5,
            "date": dates,
            "close": [2500.0, 2510.0, 2490.0, 2520.0, 2530.0],
            "adj_close": [2500.0, 2510.0, 2490.0, 2520.0, 2530.0],
            "volume": [1000] * 5,
        }
    )
    writer.write(
        df,
        layer=DataLayer.PROCESSED,
        asset_class="equity",
        symbol="RELIANCE",
        timeframe="daily",
    )

    loader = PriceLoader(writer)
    loaded = loader.load(symbols=["RELIANCE"], start="2024-01-02", end="2024-01-04")
    assert len(loaded) == 3
    assert loaded["date"].iloc[0] == pd.Timestamp("2024-01-02")
    assert loaded["date"].iloc[-1] == pd.Timestamp("2024-01-04")


def test_signal_loader_validation_and_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test SignalLoader raises on invalid signal values, duplicates, and missing files."""
    writer = ParquetWriter()
    sig_dir = tmp_path / "signals"
    monkeypatch.setattr(settings, "DATA_SIGNALS_PATH", sig_dir)
    monkeypatch.setitem(LAYER_PATH_MAP, DataLayer.SIGNALS, sig_dir)

    loader = SignalLoader(writer)

    # Missing files raise
    with pytest.raises(ValueError, match="No signals files found"):
        loader.load(symbols=["NON_EXISTENT"])

    # Duplicate (symbol, date)
    dup_df = pd.DataFrame(
        {
            "symbol": ["TCS", "TCS"],
            "date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01")],
            "predicted_return": [0.01, 0.02],
            "signal": [1, 1],
        }
    )
    writer.write(
        dup_df,
        layer=DataLayer.SIGNALS,
        asset_class="equity",
        symbol="TCS",
        timeframe="daily",
    )

    with pytest.raises(ValueError, match="Duplicate \\(symbol, date\\) rows"):
        loader.load(symbols=["TCS"])

    # Invalid signal value (e.g. 2)
    inv_df = pd.DataFrame(
        {
            "symbol": ["WIPRO"],
            "date": [pd.Timestamp("2024-01-01")],
            "predicted_return": [0.01],
            "signal": [2],
        }
    )
    writer.write(
        inv_df,
        layer=DataLayer.SIGNALS,
        asset_class="equity",
        symbol="WIPRO",
        timeframe="daily",
    )

    with pytest.raises(ValueError, match="signal must be in {-1, 0, 1}"):
        loader.load(symbols=["WIPRO"])
