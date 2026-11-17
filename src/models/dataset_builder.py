"""
Dataset builder for creating chronologically split cross-sectional training sets.

Ensures that data splits occur along global date cutoffs across the entire universe
of symbols to prevent lookahead bias. Supports optional embargo periods between splits
to eliminate target overlap leakage (critical for multi-day horizons like 5d and 10d),
rolling lookback windows, and cryptographic dataset versioning.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from config.settings import settings
from src.feature_engineering.feature_schema import TARGET_COLUMNS as SCHEMA_TARGETS
from src.feature_engineering.feature_schema import get_feature_columns
from src.models.dataset_metadata import (
    DatasetMetadata,
    build_split_summary,
    compute_data_checksum,
    compute_dataset_version,
)
from src.models.split_utils import compute_split_positions
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetSplits:
    """Container for split training, validation, and test datasets with metadata."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    feature_columns: list[str]
    meta_train: pd.DataFrame | None = None
    meta_val: pd.DataFrame | None = None
    meta_test: pd.DataFrame | None = None
    metadata: DatasetMetadata | None = None


@dataclass(slots=True)
class PreparedFrame:
    """Cleaned and sorted feature matrix ready for splitting."""

    frame: pd.DataFrame
    feature_columns: list[str]
    target_column: str


class DatasetBuilder:
    """
    Handles loading feature files, preparing clean matrices, and splitting them
    into training, validation, and test sets using chronological cutoffs.
    """

    TARGET_COLUMNS: dict[int, str] = {
        1: SCHEMA_TARGETS[0],
        5: SCHEMA_TARGETS[1],
        10: SCHEMA_TARGETS[2],
    }
    ALL_TARGET_COLUMNS: list[str] = list(TARGET_COLUMNS.values())

    def __init__(self, features_path: Path | None = None) -> None:
        """
        Initialize the DatasetBuilder.

        Args:
            features_path: Path to the directory containing feature parquet files.
                Defaults to settings.DATA_FEATURES_PATH.
        """
        self.features_path = features_path or Path(settings.DATA_FEATURES_PATH)

    def load_all_features(self, asset_class: str = "equity") -> pd.DataFrame:
        """
        Load all parquet feature files for asset_class and concatenate them.

        Returns:
            A single DataFrame containing all symbols and dates.

        Raises:
            ValueError: If no feature files are found in features_path.
        """
        sub_dir = self.features_path / asset_class
        target_dir = sub_dir if sub_dir.exists() else self.features_path
        files = [
            f for f in target_dir.glob("*.parquet")
            if not f.name.startswith(f"{asset_class}_^") and not f.name.startswith("index_")
        ] or list(target_dir.glob("*.parquet"))
        if not files:
            logger.error("No feature files found in %s", target_dir)
            raise ValueError(f"No feature files found in {target_dir}")

        logger.info("Loading %s feature files from %s", len(files), target_dir)
        dfs = [pd.read_parquet(f) for f in files]
        return pd.concat(dfs, ignore_index=True)

    def select_target_column(self, horizon: int) -> str:
        """
        Map a prediction horizon (in days) to the corresponding target column name.

        Args:
            horizon: Prediction horizon (1, 5, 10).

        Returns:
            Target column name.

        Raises:
            ValueError: If horizon is not supported.
        """
        if horizon not in self.TARGET_COLUMNS:
            logger.error("Invalid prediction horizon requested: %s", horizon)
            raise ValueError(
                f"Invalid horizon {horizon}. Supported horizons are 1, 5, 10."
            )
        return self.TARGET_COLUMNS[horizon]

    def prepare_frame(self, df: pd.DataFrame, horizon: int) -> PreparedFrame:
        """
        Validate, clean, and chronologically sort a feature DataFrame.

        Drops rows with missing target or missing features, and sorts
        by date then symbol with mergesort for deterministic order.

        Args:
            df: Raw input DataFrame from features storage.
            horizon: Prediction horizon in days.

        Returns:
            PreparedFrame containing cleaned frame and feature columns.

        Raises:
            ValueError: If df is missing symbol, date, or chosen target column.
        """
        target_col = self.select_target_column(horizon)
        required_cols = {"symbol", "date", target_col}
        missing = sorted(required_cols - set(df.columns))
        if missing:
            raise ValueError(f"Feature frame missing required columns: {missing}")

        clean_df = df.copy()
        clean_df["date"] = pd.to_datetime(clean_df["date"])

        # Drop rows where target is NaN
        clean_df = clean_df.dropna(subset=[target_col])

        # Identify feature columns and drop rows with any NaN feature
        feature_columns = get_feature_columns(clean_df.columns)
        clean_df = clean_df.dropna(subset=feature_columns)

        # Sort chronologically, then by symbol deterministically
        clean_df = clean_df.sort_values(
            ["date", "symbol"], kind="mergesort"
        ).reset_index(drop=True)

        return PreparedFrame(
            frame=clean_df,
            feature_columns=feature_columns,
            target_column=target_col,
        )

    def build(
        self,
        horizon: int | None = None,
        *,
        embargo_dates: int = 0,
        lookback_dates: int | None = None,
    ) -> DatasetSplits:
        """
        Build chronological train, validation, and test splits with optional embargo.

        Args:
            horizon: Prediction horizon. Defaults to settings.MODEL_PREDICTION_HORIZON.
            embargo_dates: Number of unique dates purged before validation and test
                starts to avoid forward-looking return label leakage. Defaults to 0.
            lookback_dates: Optional rolling lookback window in unique dates.

        Returns:
            DatasetSplits containing features, targets, metadata, and summary.

        Raises:
            ValueError: On insufficient dates, invalid split, or excessive embargo.
        """
        target_horizon = horizon or settings.MODEL_PREDICTION_HORIZON
        raw_df = self.load_all_features()
        prepared = self.prepare_frame(raw_df, target_horizon)

        df = prepared.frame
        feature_columns = prepared.feature_columns
        target_col = prepared.target_column

        # Apply rolling lookback window if requested
        if lookback_dates is not None:
            if lookback_dates < 3:
                raise ValueError("Insufficient unique dates for train/val/test split.")
            unique_dates_all = sorted(df["date"].unique())
            if len(unique_dates_all) < 3:
                raise ValueError("Insufficient unique dates for train/val/test split.")
            kept_dates = set(unique_dates_all[-lookback_dates:])
            df = df[df["date"].isin(kept_dates)].reset_index(drop=True)

        unique_dates = sorted(df["date"].unique())
        num_dates = len(unique_dates)

        positions = compute_split_positions(
            n_dates=num_dates,
            train_ratio=settings.TRAIN_SPLIT_RATIO,
            val_ratio=settings.VAL_SPLIT_RATIO,
            embargo_dates=embargo_dates,
        )

        k = embargo_dates
        train_cutoff = unique_dates[positions.train_idx - k]
        val_start = unique_dates[positions.train_idx]
        val_cutoff = unique_dates[positions.val_idx - k]
        test_start = unique_dates[positions.val_idx]

        logger.info(
            "Splitting chronologically. train<=%s, val=(%s..%s], test>%s",
            train_cutoff,
            val_start,
            val_cutoff,
            test_start,
        )

        train_df = df[df["date"] <= train_cutoff]
        val_df = df[(df["date"] > val_start) & (df["date"] <= val_cutoff)]
        test_df = df[df["date"] > test_start]

        meta_train = train_df[["symbol", "date"]].copy()
        meta_val = val_df[["symbol", "date"]].copy()
        meta_test = test_df[["symbol", "date"]].copy()

        splits_summary = (
            build_split_summary("train", train_df),
            build_split_summary("val", val_df),
            build_split_summary("test", test_df),
        )

        checksum = compute_data_checksum(
            [
                train_df[feature_columns],
                train_df[target_col],
                val_df[feature_columns],
                val_df[target_col],
                test_df[feature_columns],
                test_df[target_col],
            ]
        )

        dataset_ver = compute_dataset_version(
            horizon=target_horizon,
            feature_columns=feature_columns,
            splits=splits_summary,
            data_checksum=checksum,
        )

        metadata = DatasetMetadata(
            horizon=target_horizon,
            target_column=target_col,
            feature_columns=tuple(feature_columns),
            feature_set_version=settings.FEATURE_SET_VERSION,
            embargo_dates=embargo_dates,
            lookback_dates=lookback_dates,
            splits=splits_summary,
            dataset_version=dataset_ver,
        )

        return DatasetSplits(
            X_train=train_df[feature_columns].copy(),
            y_train=train_df[target_col].copy(),
            X_val=val_df[feature_columns].copy(),
            y_val=val_df[target_col].copy(),
            X_test=test_df[feature_columns].copy(),
            y_test=test_df[target_col].copy(),
            feature_columns=list(feature_columns),
            meta_train=meta_train,
            meta_val=meta_val,
            meta_test=meta_test,
            metadata=metadata,
        )
