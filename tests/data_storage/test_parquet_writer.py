"""
Unit tests for ParquetWriter.

Tests cover:
- build_file_path returns correct path
- write creates file on disk
- write overwrites existing file
- write appends and deduplicates correctly
- write skips empty DataFrame
- write raises on invalid mode
- read returns correct DataFrame
- read raises FileNotFoundError for missing file
- exists returns True for existing file
- exists returns False for missing file
- delete removes file
- delete raises FileNotFoundError for missing file
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data_storage.parquet_writer import DataLayer, ParquetWriter, ParquetWriterError


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture()
def writer(tmp_path: Path, monkeypatch) -> ParquetWriter:
    """Return a ParquetWriter with all data layer paths pointed to tmp_path."""
    import src.data_storage.parquet_writer as pw

    monkeypatch.setitem(
        pw.LAYER_PATH_MAP, DataLayer.RAW, tmp_path / "raw"
    )
    monkeypatch.setitem(
        pw.LAYER_PATH_MAP, DataLayer.PROCESSED, tmp_path / "processed"
    )
    monkeypatch.setitem(
        pw.LAYER_PATH_MAP, DataLayer.FEATURES, tmp_path / "features"
    )
    monkeypatch.setitem(
        pw.LAYER_PATH_MAP, DataLayer.SIGNALS, tmp_path / "signals"
    )
    monkeypatch.setitem(
        pw.LAYER_PATH_MAP, DataLayer.PREDICTIONS, tmp_path / "predictions"
    )

    return ParquetWriter()


def make_df(dates: list[str], symbol: str = "RELIANCE.NS") -> pd.DataFrame:
    n = len(dates)
    return pd.DataFrame(
        {
            "symbol": [symbol] * n,
            "asset_class": ["equity"] * n,
            "exchange": ["NSE"] * n,
            "currency": ["INR"] * n,
            "date": pd.to_datetime(dates, utc=True),
            "open": [100.0] * n,
            "high": [105.0] * n,
            "low": [99.0] * n,
            "close": [103.0] * n,
            "adj_close": [103.0] * n,
            "volume": [1000.0] * n,
            "source": ["yfinance"] * n,
        }
    )


# ── Tests: build_file_path ─────────────────────────────────────


class TestBuildFilePath:
    def test_returns_correct_filename(self, writer: ParquetWriter, tmp_path: Path) -> None:
        path = writer.build_file_path(
            DataLayer.RAW, "equity", "RELIANCE", "daily"
        )
        assert path.name == "equity_RELIANCE_daily.parquet"

    def test_returns_correct_directory(self, writer: ParquetWriter, tmp_path: Path) -> None:
        path = writer.build_file_path(
            DataLayer.RAW, "equity", "RELIANCE", "daily"
        )
        assert path.parent.name == "equity"

    def test_different_layers_return_different_paths(
        self, writer: ParquetWriter
    ) -> None:
        raw_path = writer.build_file_path(DataLayer.RAW, "equity", "RELIANCE", "daily")
        processed_path = writer.build_file_path(
            DataLayer.PROCESSED, "equity", "RELIANCE", "daily"
        )
        assert raw_path != processed_path


# ── Tests: write ──────────────────────────────────────────────


class TestParquetWriterWrite:
    def test_write_creates_file(self, writer: ParquetWriter) -> None:
        df = make_df(["2024-01-01", "2024-01-02"])
        path = writer.write(
            df,
            layer=DataLayer.RAW,
            asset_class="equity",
            symbol="RELIANCE.NS",
            timeframe="daily",
        )
        assert path.exists()

    def test_write_returns_correct_path(self, writer: ParquetWriter) -> None:
        df = make_df(["2024-01-01"])
        path = writer.write(
            df,
            layer=DataLayer.RAW,
            asset_class="equity",
            symbol="RELIANCE.NS",
            timeframe="daily",
        )
        assert path.name == "equity_RELIANCE.NS_daily.parquet"

    def test_write_overwrite_replaces_existing(self, writer: ParquetWriter) -> None:
        df1 = make_df(["2024-01-01", "2024-01-02"])
        df2 = make_df(["2024-01-03", "2024-01-04"])

        writer.write(
            df1,
            layer=DataLayer.RAW,
            asset_class="equity",
            symbol="RELIANCE.NS",
            timeframe="daily",
            mode="overwrite",
        )
        writer.write(
            df2,
            layer=DataLayer.RAW,
            asset_class="equity",
            symbol="RELIANCE.NS",
            timeframe="daily",
            mode="overwrite",
        )

        result = writer.read(
            layer=DataLayer.RAW,
            asset_class="equity",
            symbol="RELIANCE.NS",
            timeframe="daily",
        )
        assert len(result) == 2
        assert pd.to_datetime(result["date"].min()).date().isoformat() == "2024-01-03"

    def test_write_append_merges_data(self, writer: ParquetWriter) -> None:
        df1 = make_df(["2024-01-01", "2024-01-02"])
        df2 = make_df(["2024-01-03", "2024-01-04"])

        writer.write(
            df1,
            layer=DataLayer.RAW,
            asset_class="equity",
            symbol="RELIANCE.NS",
            timeframe="daily",
            mode="overwrite",
        )
        writer.write(
            df2,
            layer=DataLayer.RAW,
            asset_class="equity",
            symbol="RELIANCE.NS",
            timeframe="daily",
            mode="append",
        )

        result = writer.read(
            layer=DataLayer.RAW,
            asset_class="equity",
            symbol="RELIANCE.NS",
            timeframe="daily",
        )
        assert len(result) == 4

    def test_write_append_deduplicates(self, writer: ParquetWriter) -> None:
        df1 = make_df(["2024-01-01", "2024-01-02"])
        df2 = make_df(["2024-01-02", "2024-01-03"])

        writer.write(
            df1,
            layer=DataLayer.RAW,
            asset_class="equity",
            symbol="RELIANCE.NS",
            timeframe="daily",
            mode="overwrite",
        )
        writer.write(
            df2,
            layer=DataLayer.RAW,
            asset_class="equity",
            symbol="RELIANCE.NS",
            timeframe="daily",
            mode="append",
        )

        result = writer.read(
            layer=DataLayer.RAW,
            asset_class="equity",
            symbol="RELIANCE.NS",
            timeframe="daily",
        )
        assert len(result) == 3

    def test_write_empty_df_skips(self, writer: ParquetWriter) -> None:
        df = pd.DataFrame()
        path = writer.write(
            df,
            layer=DataLayer.RAW,
            asset_class="equity",
            symbol="RELIANCE.NS",
            timeframe="daily",
        )
        assert not path.exists()

    def test_write_invalid_mode_raises(self, writer: ParquetWriter) -> None:
        df = make_df(["2024-01-01"])
        with pytest.raises(ParquetWriterError):
            writer.write(
                df,
                layer=DataLayer.RAW,
                asset_class="equity",
                symbol="RELIANCE.NS",
                timeframe="daily",
                mode="invalid",
            )


# ── Tests: read ───────────────────────────────────────────────


class TestParquetWriterRead:
    def test_read_returns_dataframe(self, writer: ParquetWriter) -> None:
        df = make_df(["2024-01-01", "2024-01-02"])
        writer.write(
            df,
            layer=DataLayer.RAW,
            asset_class="equity",
            symbol="RELIANCE.NS",
            timeframe="daily",
        )
        result = writer.read(
            layer=DataLayer.RAW,
            asset_class="equity",
            symbol="RELIANCE.NS",
            timeframe="daily",
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_read_missing_file_raises(self, writer: ParquetWriter) -> None:
        with pytest.raises(FileNotFoundError):
            writer.read(
                layer=DataLayer.RAW,
                asset_class="equity",
                symbol="MISSING.NS",
                timeframe="daily",
            )

    def test_read_columns_preserved(self, writer: ParquetWriter) -> None:
        df = make_df(["2024-01-01"])
        writer.write(
            df,
            layer=DataLayer.RAW,
            asset_class="equity",
            symbol="RELIANCE.NS",
            timeframe="daily",
        )
        result = writer.read(
            layer=DataLayer.RAW,
            asset_class="equity",
            symbol="RELIANCE.NS",
            timeframe="daily",
        )
        for col in df.columns:
            assert col in result.columns


# ── Tests: exists ─────────────────────────────────────────────


class TestParquetWriterExists:
    def test_exists_returns_true_for_written_file(
        self, writer: ParquetWriter
    ) -> None:
        df = make_df(["2024-01-01"])
        writer.write(
            df,
            layer=DataLayer.RAW,
            asset_class="equity",
            symbol="RELIANCE.NS",
            timeframe="daily",
        )
        assert writer.exists(
            layer=DataLayer.RAW,
            asset_class="equity",
            symbol="RELIANCE.NS",
            timeframe="daily",
        )

    def test_exists_returns_false_for_missing_file(
        self, writer: ParquetWriter
    ) -> None:
        assert not writer.exists(
            layer=DataLayer.RAW,
            asset_class="equity",
            symbol="MISSING.NS",
            timeframe="daily",
        )


# ── Tests: delete ─────────────────────────────────────────────


class TestParquetWriterDelete:
    def test_delete_removes_file(self, writer: ParquetWriter) -> None:
        df = make_df(["2024-01-01"])
        writer.write(
            df,
            layer=DataLayer.RAW,
            asset_class="equity",
            symbol="RELIANCE.NS",
            timeframe="daily",
        )
        writer.delete(
            layer=DataLayer.RAW,
            asset_class="equity",
            symbol="RELIANCE.NS",
            timeframe="daily",
        )
        assert not writer.exists(
            layer=DataLayer.RAW,
            asset_class="equity",
            symbol="RELIANCE.NS",
            timeframe="daily",
        )

    def test_delete_missing_file_raises(self, writer: ParquetWriter) -> None:
        with pytest.raises(FileNotFoundError):
            writer.delete(
                layer=DataLayer.RAW,
                asset_class="equity",
                symbol="MISSING.NS",
                timeframe="daily",
            )