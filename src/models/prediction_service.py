"""High-level prediction service orchestrating model loading and inference.

Loads the latest (or specific) trained model from ModelStore, reads features,
filters rows strictly by evaluation scope ('test' out-of-sample by default),
runs inference, and persists results to the predictions lake via PredictionPipeline.
"""

from __future__ import annotations

import pandas as pd

from src.models.dataset_builder import DatasetBuilder
from src.models.model_store import ModelStore
from src.models.prediction_pipeline import PredictionPipeline
from src.models.split_utils import align_timestamp
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PredictionService:
    """
    High-level service for executing model inference on feature datasets.

    Scopes:
        - 'test' (default): Predicts only on dates strictly after val_end_date.
          This is the primary production default for backtesting, as predictions
          on training and validation periods represent in-sample / fit-period data
          and would invalidate backtest realism.
        - 'oos': Predicts on all out-of-sample dates strictly after train_end_date.
        - 'all': Predicts on all available rows without date filtering.
        - 'latest': Predicts only on the most recent available date per symbol,
          appending predictions to storage for live trading support.
    """

    SCOPES: tuple[str, ...] = ("all", "oos", "test", "latest")

    def __init__(
        self,
        model_store: ModelStore | None = None,
        prediction_pipeline: PredictionPipeline | None = None,
        dataset_builder: DatasetBuilder | None = None,
    ) -> None:
        """
        Initialize the PredictionService with storage, pipeline, and dataset builder.

        Args:
            model_store: ModelStore instance for loading models and metadata.
            prediction_pipeline: Pipeline handling inference and parquet storage.
            dataset_builder: DatasetBuilder instance for loading feature files.
        """
        self.model_store = model_store or ModelStore()
        self.prediction_pipeline = prediction_pipeline or PredictionPipeline()
        self.dataset_builder = dataset_builder or DatasetBuilder()

    def run(
        self,
        model_name: str,
        asset_class: str = "equity",
        timeframe: str = "daily",
        model_version: str | None = None,
        scope: str = "test",
    ) -> pd.DataFrame:
        """
        Load model, filter features by scope, generate predictions, and persist.

        Args:
            model_name: Name of the trained model to load.
            asset_class: Asset class partition (e.g. 'equity').
            timeframe: Data timeframe (e.g. 'daily').
            model_version: Specific version string (None = LATEST).
            scope: Prediction date scope ('test', 'oos', 'all', 'latest').

        Returns:
            DataFrame with symbol, date, predicted_return, model_name, model_version.

        Raises:
            ValueError: If scope is invalid or split dates are missing.
            FileNotFoundError: If model cannot be found in the store.
        """
        if scope not in self.SCOPES:
            raise ValueError(f"scope must be one of {self.SCOPES}, got '{scope}'.")

        model, meta = self.model_store.load(model_name, model_version)
        raw_features = self.dataset_builder.load_all_features()

        features_df = raw_features.copy()
        features_df["date"] = pd.to_datetime(features_df["date"])

        missing = sorted(set(meta.feature_columns) - set(features_df.columns))
        if missing:
            raise ValueError(f"features_df missing required columns: {missing}")

        # Drop rows with NaN features only (never drop based on targets)
        before_drop = len(features_df)
        clean_df = features_df.dropna(subset=list(meta.feature_columns)).copy()
        dropped = before_drop - len(clean_df)
        if dropped > 0:
            logger.info(
                "Dropped %s rows containing NaN values in feature columns.",
                dropped,
            )

        # Apply scope filtering
        if scope == "all":
            filtered = clean_df
        elif scope == "oos":
            if meta.train_end_date is None:
                raise ValueError("Model metadata has no split dates; use scope='all'.")
            cutoff = align_timestamp(meta.train_end_date, clean_df["date"])
            filtered = clean_df[clean_df["date"] > cutoff].copy()
        elif scope == "test":
            if meta.val_end_date is None:
                raise ValueError("Model metadata has no split dates; use scope='all'.")
            cutoff = align_timestamp(meta.val_end_date, clean_df["date"])
            filtered = clean_df[clean_df["date"] > cutoff].copy()
        elif scope == "latest":
            # Keep latest date per symbol
            filtered = (
                clean_df.sort_values(["symbol", "date"])
                .groupby("symbol", as_index=False)
                .last()
            )

        if filtered.empty:
            raise ValueError(f"No rows to predict for scope='{scope}'.")

        filtered = filtered.reset_index(drop=True)
        write_mode = "append" if scope == "latest" else "overwrite"

        preds_df = self.prediction_pipeline.run(
            model=model,
            features_df=filtered,
            feature_columns=list(meta.feature_columns),
            model_version=meta.version,
            asset_class=asset_class,
            timeframe=timeframe,
            write_mode=write_mode,
        )

        return preds_df
