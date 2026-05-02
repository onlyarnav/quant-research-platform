"""
Parquet writer and reader for the Quant AI Research Platform data lake.

Responsibilities:
- write DataFrames to parquet following naming conventions
- read parquet files back into DataFrames
- enforce directory structure
- support append and overwrite modes
- validate file paths before read/write

File naming convention:
    <asset_class>_<symbol>_<timeframe>.parquet

Usage:
    from src.data_storage.parquet_writer import ParquetWriter
    writer = ParquetWriter()
    writer.write(df, layer="raw", asset_class="equity", symbol="RELIANCE", timeframe="daily")
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config.settings import settings

logger = logging.getLogger(__name__)


class DataLayer(str, Enum):
    RAW = "raw"
    PROCESSED = "processed"
    FEATURES = "features"
    SIGNALS = "signals"
    PREDICTIONS = "predictions"


LAYER_PATH_MAP: dict[DataLayer, Path] = {
    DataLayer.RAW: settings.DATA_RAW_PATH,
    DataLayer.PROCESSED: settings.DATA_PROCESSED_PATH,
    DataLayer.FEATURES: settings.DATA_FEATURES_PATH,
    DataLayer.SIGNALS: settings.DATA_SIGNALS_PATH,
    DataLayer.PREDICTIONS: settings.DATA_PREDICTIONS_PATH,
}


class ParquetWriterError(Exception):
    """Base exception for parquet storage errors."""


class ParquetWriter:
    """
    Handles all parquet read/write operations for the platform data lake.

    Enforces:
    - standard file naming convention
    - directory creation
    - overwrite and append modes
    - schema consistency via pyarrow
    """

    @staticmethod
    def build_file_path(
        layer: DataLayer,
        asset_class: str,
        symbol: str,
        timeframe: str,
    ) -> Path:
        """
        Build the canonical parquet file path for a dataset.

        Args:
            layer: Data lake layer (raw, processed, features, signals, predictions).
            asset_class: Asset class such as equity, index, crypto.
            symbol: Trading symbol such as RELIANCE, NIFTY50.
            timeframe: Data timeframe such as daily, hourly.

        Returns:
            Absolute Path to the parquet file.
        """
        base_path = LAYER_PATH_MAP[layer]
        filename = f"{asset_class}_{symbol}_{timeframe}.parquet"
        return base_path / asset_class / filename

    def write(
        self,
        df: pd.DataFrame,
        *,
        layer: DataLayer,
        asset_class: str,
        symbol: str,
        timeframe: str,
        mode: str = "overwrite",
    ) -> Path:
        """
        Write a DataFrame to parquet storage.

        Args:
            df: DataFrame to write.
            layer: Target data lake layer.
            asset_class: Asset class of the data.
            symbol: Trading symbol.
            timeframe: Data timeframe.
            mode: Write mode. Either overwrite or append.

        Returns:
            Path where the file was written.

        Raises:
            ParquetWriterError: If write fails or mode is invalid.
        """
        if mode not in {"overwrite", "append"}:
            raise ParquetWriterError(
                f"Invalid write mode={mode!r}. Must be 'overwrite' or 'append'."
            )

        if df.empty:
            logger.warning(
                "Skipping write — empty DataFrame for symbol=%s layer=%s",
                symbol,
                layer.value,
            )
            return self.build_file_path(layer, asset_class, symbol, timeframe)

        file_path = self.build_file_path(layer, asset_class, symbol, timeframe)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if mode == "append" and file_path.exists():
                existing_df = self.read(
                    layer=layer,
                    asset_class=asset_class,
                    symbol=symbol,
                    timeframe=timeframe,
                )
                df = pd.concat([existing_df, df], ignore_index=True)
                df = df.drop_duplicates(subset=["symbol", "date"])
                df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
                logger.debug(
                    "Appended data for symbol=%s. Total rows after merge=%s",
                    symbol,
                    len(df),
                )

            table = pa.Table.from_pandas(df, preserve_index=False)
            pq.write_table(table, file_path, compression="snappy")

            logger.info(
                "Wrote %s rows to %s",
                len(df),
                file_path,
            )

            return file_path

        except Exception as exc:
            raise ParquetWriterError(
                f"Failed to write parquet for symbol={symbol} "
                f"layer={layer.value}: {exc}"
            ) from exc

    def read(
        self,
        *,
        layer: DataLayer,
        asset_class: str,
        symbol: str,
        timeframe: str,
    ) -> pd.DataFrame:
        """
        Read a parquet file from the data lake.

        Args:
            layer: Source data lake layer.
            asset_class: Asset class of the data.
            symbol: Trading symbol.
            timeframe: Data timeframe.

        Returns:
            DataFrame loaded from parquet.

        Raises:
            FileNotFoundError: If the parquet file does not exist.
            ParquetWriterError: If read fails.
        """
        file_path = self.build_file_path(layer, asset_class, symbol, timeframe)

        if not file_path.exists():
            raise FileNotFoundError(
                f"Parquet file not found: {file_path}"
            )

        try:
            table = pq.read_table(file_path)
            df = table.to_pandas()

            logger.info(
                "Read %s rows from %s",
                len(df),
                file_path,
            )

            return df

        except Exception as exc:
            raise ParquetWriterError(
                f"Failed to read parquet for symbol={symbol} "
                f"layer={layer.value}: {exc}"
            ) from exc

    def exists(
        self,
        *,
        layer: DataLayer,
        asset_class: str,
        symbol: str,
        timeframe: str,
    ) -> bool:
        """
        Check whether a parquet file exists in the data lake.

        Args:
            layer: Data lake layer to check.
            asset_class: Asset class of the data.
            symbol: Trading symbol.
            timeframe: Data timeframe.

        Returns:
            True if the file exists, False otherwise.
        """
        file_path = self.build_file_path(layer, asset_class, symbol, timeframe)
        return file_path.exists()

    def delete(
        self,
        *,
        layer: DataLayer,
        asset_class: str,
        symbol: str,
        timeframe: str,
    ) -> None:
        """
        Delete a parquet file from the data lake.

        Args:
            layer: Data lake layer.
            asset_class: Asset class of the data.
            symbol: Trading symbol.
            timeframe: Data timeframe.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        file_path = self.build_file_path(layer, asset_class, symbol, timeframe)

        if not file_path.exists():
            raise FileNotFoundError(
                f"Cannot delete — parquet file not found: {file_path}"
            )

        file_path.unlink()
        logger.info("Deleted parquet file: %s", file_path)