"""
Dataset builder for creating chronologically split cross-sectional training sets.

This module ensures that data splits occur along date cutoffs across the entire
universe of symbols to prevent lookahead bias, where future data from one symbol
could leak into the training set of another.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class DatasetSplits:
    """
    Container for split training, validation, and test datasets.
    """
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    feature_columns: list[str]

class DatasetBuilder:
    """
    Handles loading feature files and splitting them into training,
    validation, and test sets using chronological cutoffs.
    """

    ALL_TARGET_COLUMNS = ["future_return_1d", "future_return_5d", "future_return_10d"]

    def __init__(self, features_path: Path | None = None) -> None:
        """
        Initialize the DatasetBuilder.

        Args:
            features_path: Path to the directory containing feature parquet files.
                           Defaults to settings.DATA_FEATURES_PATH.
        """
        self.features_path = features_path or Path(settings.DATA_FEATURES_PATH)

    def load_all_features(self) -> pd.DataFrame:
        """
        Load all parquet feature files from the features directory and concatenate them.

        Returns:
            A single DataFrame containing all symbols and dates.

        Raises:
            ValueError: If no feature files are found in the features_path.
        """
        files = list(self.features_path.rglob("*.parquet"))
        if not files:
            logger.error("No feature files found in %s", self.features_path)
            raise ValueError(f"No feature files found in {self.features_path}")

        logger.info("Loading %d feature files from %s", len(files), self.features_path)
        dfs = [pd.read_parquet(f) for f in files]
        return pd.concat(dfs, ignore_index=True)

    def select_target_column(self, horizon: int) -> str:
        """
        Map a prediction horizon (in days) to the corresponding target column name.

        Args:
            horizon: The prediction horizon (e.g., 1, 5, 10).

        Returns:
            The name of the target column.

        Raises:
            ValueError: If the horizon is not one of the supported values (1, 5, 10).
        """
        mapping = {
            1: "future_return_1d",
            5: "future_return_5d",
            10: "future_return_10d",
        }
        if horizon not in mapping:
            logger.error("Invalid prediction horizon requested: %s", horizon)
            raise ValueError(f"Invalid horizon {horizon}. Supported horizons are 1, 5, 10.")

        return mapping[horizon]

    def build(self, horizon: int | None = None) -> DatasetSplits:
        """
        Build the training, validation, and test splits.

        The process follows these steps:
        1. Load all features into a single DataFrame.
        2. Select the target column based on the horizon.
        3. Drop rows with NaN targets (end-of-series) or NaN features (warm-up period).
        4. Identify feature columns by excluding identifiers (symbol, date) and all targets.
        5. Sort the data chronologically by date.
        6. Compute date cutoffs based on TRAIN_SPLIT_RATIO and VAL_SPLIT_RATIO.
        7. Split the data across the entire universe at these date boundaries.

        Args:
            horizon: The prediction horizon. Defaults to settings.MODEL_PREDICTION_HORIZON.

        Returns:
            A DatasetSplits instance containing the processed and split data.
        """
        target_horizon = horizon or settings.MODEL_PREDICTION_HORIZON
        target_col = self.select_target_column(target_horizon)

        df = self.load_all_features()

        # Drop rows where target is NaN (forward-looking returns not available)
        df = df.dropna(subset=[target_col])

        # Drop rows where any potential feature is NaN (rolling window warm-up)
        # We identify potential features by excluding known non-feature columns
        non_feature_cols = ["symbol", "date"] + self.ALL_TARGET_COLUMNS
        feature_candidates = [c for c in df.columns if c not in non_feature_cols]
        df = df.dropna(subset=feature_candidates)

        # Final feature column identification
        feature_columns = feature_candidates

        # Sort by date ascending to ensure chronological splits
        df = df.sort_values("date").reset_index(drop=True)

        # Calculate chronological date cutoffs across the entire universe
        unique_dates = sorted(df["date"].unique())
        num_dates = len(unique_dates)

        if num_dates < 3:
            raise ValueError("Insufficient unique dates for train/val/test split.")

        train_idx = int(num_dates * settings.TRAIN_SPLIT_RATIO)
        val_idx = int(num_dates * (settings.TRAIN_SPLIT_RATIO + settings.VAL_SPLIT_RATIO))

        train_idx = min(train_idx, num_dates - 2)
        val_idx = min(val_idx, num_dates - 1)

        if train_idx >= val_idx:
            raise ValueError(
                f"Invalid split: train_idx={train_idx} >= val_idx={val_idx}. "
                f"Adjust TRAIN_SPLIT_RATIO/VAL_SPLIT_RATIO or provide more data."
            )

        cutoff1 = unique_dates[train_idx]
        cutoff2 = unique_dates[val_idx]

        logger.info("Splitting dataset chronologically. Cutoffs: %s, %s", cutoff1, cutoff2)

        # Split by date boundaries (Cross-Sectional)
        train_df = df[df["date"] <= cutoff1]
        val_df = df[(df["date"] > cutoff1) & (df["date"] <= cutoff2)]
        test_df = df[df["date"] > cutoff2]

        return DatasetSplits(
            X_train=train_df[feature_columns],
            y_train=train_df[target_col],
            X_val=val_df[feature_columns],
            y_val=val_df[target_col],
            X_test=test_df[feature_columns],
            y_test=test_df[target_col],
            feature_columns=feature_columns,
        )
