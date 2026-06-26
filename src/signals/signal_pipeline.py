"""
Signal generation orchestration pipeline.

Reads model predictions from the data lake, applies a configurable
signal generation strategy (threshold-based or rank-based), and
persists the resulting BUY/HOLD/SELL signals back to the data lake.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from src.data_storage.parquet_writer import DataLayer, ParquetWriter
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SignalGenerator(Protocol):
    """
    Structural type for any signal generation strategy.

    Both ThresholdSignalGenerator and RankSignalGenerator satisfy
    this protocol without needing a shared base class.
    """

    def generate(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        ...


class SignalPipeline:
    """
    Orchestrates reading predictions, applying a signal strategy,
    and persisting signals — one parquet file per symbol.
    """

    def __init__(self, writer: ParquetWriter | None = None) -> None:
        """
        Args:
            writer: Optional ParquetWriter instance. Defaults to a
                new ParquetWriter if not provided.
        """
        self.writer = writer or ParquetWriter()

    def run(
        self,
        generator: SignalGenerator,
        symbols: list[str],
        asset_class: str,
        timeframe: str = "daily",
    ) -> pd.DataFrame:
        """
        Read predictions for the given symbols, generate signals,
        and write results back to the signals data lake layer.

        Args:
            generator: A signal generation strategy implementing
                .generate(predictions_df) -> DataFrame.
            symbols: List of symbols to process.
            asset_class: Asset class for file path construction (e.g. "equity").
            timeframe: Data timeframe for file naming. Defaults to "daily".

        Returns:
            Combined DataFrame of all generated signals across symbols.

        Raises:
            ValueError: If symbols is empty, or if no prediction files
                were found for any of the requested symbols.
        """
        if not symbols:
            raise ValueError("symbols list cannot be empty.")

        predictions_frames = []

        for symbol in symbols:
            try:
                symbol_predictions = self.writer.read(
                    layer=DataLayer.PREDICTIONS,
                    asset_class=asset_class,
                    symbol=symbol,
                    timeframe=timeframe,
                )
                predictions_frames.append(symbol_predictions)
            except FileNotFoundError:
                logger.warning(
                    "No prediction file found for symbol=%s asset_class=%s. Skipping.",
                    symbol,
                    asset_class,
                )

        if not predictions_frames:
            raise ValueError(
                f"No prediction files found for any of the requested symbols: {symbols}"
            )

        predictions_df = pd.concat(predictions_frames, ignore_index=True)

        signals_df = generator.generate(predictions_df)

        for symbol in signals_df["symbol"].unique():
            symbol_df = signals_df[signals_df["symbol"] == symbol].reset_index(drop=True)
            self.writer.write(
                symbol_df,
                layer=DataLayer.SIGNALS,
                asset_class=asset_class,
                symbol=symbol,
                timeframe=timeframe,
                mode="overwrite",
            )

        logger.info(
            "Generated and wrote signals for %s symbols.",
            signals_df["symbol"].nunique(),
        )

        return signals_df