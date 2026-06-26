"""
Prediction pipeline for generating and persisting model predictions.

Loads a fitted model and a cross-sectional feature dataset, generates
predicted returns for every symbol, and persists results to the
predictions data lake layer — one parquet file per symbol, matching
the platform's file naming convention.
"""
from __future__ import annotations

import pandas as pd

from src.data_storage.parquet_writer import DataLayer, ParquetWriter
from src.models.base_model import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PredictionPipeline:
    def __init__(self, writer: ParquetWriter | None = None) -> None:
        self.writer = writer or ParquetWriter()

    def run(
        self,
        model: BaseModel,
        features_df: pd.DataFrame,
        feature_columns: list[str],
        model_version: str,
        asset_class: str,
        timeframe: str = "daily",
    ) -> pd.DataFrame:
        """
        Generate predictions for all symbols in features_df and persist them.

        Steps:
            1. Validate input.
            2. Predict on features_df[feature_columns].
            3. Build output schema: symbol, date, predicted_return, model_name, model_version.
            4. Write one parquet file per symbol to the predictions layer.
            5. Return the combined predictions DataFrame.

        Args:
            model: A fitted BaseModel instance.
            features_df: Cross-sectional feature DataFrame containing at least
                'symbol', 'date', and all columns in feature_columns.
            feature_columns: Column names to pass into model.predict().
            model_version: Version identifier for this model (e.g. "v1.0").
            asset_class: Asset class for file path construction (e.g. "equity").
            timeframe: Data timeframe for file naming. Defaults to "daily".

        Returns:
            DataFrame with columns: symbol, date, predicted_return, model_name, model_version.

        Raises:
            ValueError: If features_df is empty or missing required columns.
        """
        if features_df.empty:
            raise ValueError("features_df cannot be empty.")

        required_cols = {"symbol", "date", *feature_columns}
        missing = required_cols - set(features_df.columns)
        if missing:
            raise ValueError(f"features_df missing required columns: {missing}")

        predictions = model.predict(features_df[feature_columns])

        output_df = pd.DataFrame({
            "symbol": features_df["symbol"].values,
            "date": features_df["date"].values,
            "predicted_return": predictions,
            "model_name": model.model_name,
            "model_version": model_version,
        })

        for symbol in output_df["symbol"].unique():
            symbol_df = output_df[output_df["symbol"] == symbol].reset_index(drop=True)
            self.writer.write(
                symbol_df,
                layer=DataLayer.PREDICTIONS,
                asset_class=asset_class,
                symbol=symbol,
                timeframe=timeframe,
                mode="overwrite",
            )

        logger.info("Wrote predictions for %s symbols.", output_df["symbol"].nunique())

        return output_df
