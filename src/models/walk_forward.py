"""Walk-forward validation and splitting for time-series equity models.

Provides leakage-safe walk-forward splitting and validation logic that
strictly respects chronological ordering, out-of-sample test windows,
optional embargo / gap periods, and expanding or rolling train windows.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from config.settings import settings
from src.models.dataset_builder import DatasetBuilder
from src.models.model_evaluator import ModelEvaluator
from src.models.model_factory import create_model
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    """Represents a single walk-forward split fold metadata."""

    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train_dates: int
    n_test_dates: int


class WalkForwardSplitter:
    """Splits time-series data chronologically into walk-forward folds."""

    def __init__(
        self,
        n_splits: int | None = None,
        test_window_dates: int | None = None,
        train_window_dates: int | None = None,
        gap_dates: int = 0,
        min_train_dates: int = 60,
    ) -> None:
        """Initialize WalkForwardSplitter.

        Args:
            n_splits: Number of walk-forward folds (defaults to settings).
            test_window_dates: Fixed dates per test fold (defaults to N // (splits+1)).
            train_window_dates: Rolling train window size (None = expanding).
            gap_dates: Embargo gap in dates between train_end and test_start.
            min_train_dates: Minimum required training dates per fold.
        """
        self.n_splits = (
            n_splits if n_splits is not None else settings.WALK_FORWARD_SPLITS
        )
        self.test_window_dates = test_window_dates
        self.train_window_dates = train_window_dates
        self.gap_dates = gap_dates
        self.min_train_dates = min_train_dates

        if self.n_splits < 1:
            raise ValueError(f"n_splits must be at least 1, got {self.n_splits}.")
        if self.gap_dates < 0:
            raise ValueError(f"gap_dates must be non-negative, got {self.gap_dates}.")
        if self.train_window_dates is not None and self.train_window_dates < 1:
            raise ValueError(
                f"train_window_dates must be at least 1, got {self.train_window_dates}."
            )
        if self.min_train_dates < 1:
            raise ValueError(
                f"min_train_dates must be at least 1, got {self.min_train_dates}."
            )

    def split_dates(
        self, unique_dates: Sequence[pd.Timestamp]
    ) -> list[WalkForwardFold]:
        """Generate chronological fold metadata over a sequence of dates.

        Args:
            unique_dates: Sequence of unique timestamps sorted chronologically.

        Returns:
            List of WalkForwardFold definitions.

        Raises:
            ValueError: If insufficient dates exist to satisfy the splits.
        """
        dates = pd.DatetimeIndex(sorted(unique_dates))
        n = len(dates)
        w = self.test_window_dates or (n // (self.n_splits + 1))
        if w < 1:
            raise ValueError("Not enough dates for the requested number of splits.")

        folds: list[WalkForwardFold] = []
        for k in range(self.n_splits):
            test_start_idx = n - (self.n_splits - k) * w
            test_end_idx = test_start_idx + w - 1
            train_end_idx = test_start_idx - self.gap_dates - 1
            train_start_idx = (
                0
                if self.train_window_dates is None
                else max(0, train_end_idx - self.train_window_dates + 1)
            )

            n_train_dates = train_end_idx - train_start_idx + 1
            if n_train_dates < self.min_train_dates:
                raise ValueError(
                    f"Insufficient history for fold {k}: {n_train_dates} dates "
                    f"available, min_train_dates={self.min_train_dates}."
                )

            folds.append(
                WalkForwardFold(
                    fold=k,
                    train_start=dates[train_start_idx],
                    train_end=dates[train_end_idx],
                    test_start=dates[test_start_idx],
                    test_end=dates[test_end_idx],
                    n_train_dates=n_train_dates,
                    n_test_dates=test_end_idx - test_start_idx + 1,
                )
            )
        return folds

    def split(
        self, df: pd.DataFrame
    ) -> Iterator[tuple[WalkForwardFold, np.ndarray, np.ndarray]]:
        """Yield fold definitions and positional row indices for a DataFrame.

        Args:
            df: DataFrame containing at least a 'date' column.

        Yields:
            Tuples of (fold, train_positions, test_positions).

        Raises:
            ValueError: If df is empty or missing 'date'.
        """
        if df.empty:
            raise ValueError("df cannot be empty.")
        if "date" not in df.columns:
            raise ValueError("df missing required columns: {'date'}")

        folds = self.split_dates(pd.DatetimeIndex(sorted(df["date"].unique())))
        df_dates = pd.DatetimeIndex(df["date"])
        for fold in folds:
            train_mask = (df_dates >= fold.train_start) & (df_dates <= fold.train_end)
            test_mask = (df_dates >= fold.test_start) & (df_dates <= fold.test_end)
            yield fold, np.where(train_mask)[0], np.where(test_mask)[0]


@dataclass(slots=True)
class WalkForwardResult:
    """Contains results of a walk-forward validation run."""

    fold_metrics: pd.DataFrame
    summary: dict[str, float]


class WalkForwardValidator:
    """Executes walk-forward model training and out-of-sample evaluation."""

    def __init__(
        self,
        splitter: WalkForwardSplitter | None = None,
        evaluator: ModelEvaluator | None = None,
        dataset_builder: DatasetBuilder | None = None,
        val_fraction: float = 0.15,
    ) -> None:
        """Initialize WalkForwardValidator.

        Args:
            splitter: Optional WalkForwardSplitter instance.
            evaluator: Optional ModelEvaluator instance.
            dataset_builder: Optional DatasetBuilder instance.
            val_fraction: Fraction of train dates carved for early stopping val.
        """
        self.splitter = splitter
        self.evaluator = evaluator or ModelEvaluator()
        self.dataset_builder = dataset_builder or DatasetBuilder()
        self.val_fraction = val_fraction

    def run(
        self,
        df: pd.DataFrame,
        model_type: str,
        horizon: int,
        hyperparameters: dict[str, Any] | None = None,
    ) -> WalkForwardResult:
        """Execute walk-forward evaluation across all folds.

        Args:
            df: Feature DataFrame with identifiers, features, and targets.
            model_type: Model architecture key from the model factory.
            horizon: Target return prediction horizon in days.
            hyperparameters: Optional dictionary of hyperparameters for create_model.

        Returns:
            WalkForwardResult containing fold_metrics and summary.
        """
        prepared = self.dataset_builder.prepare_frame(df, horizon)
        splitter = self.splitter or WalkForwardSplitter(gap_dates=horizon)
        fold_rows: list[dict[str, Any]] = []

        for fold, train_pos, test_pos in splitter.split(prepared.frame):
            train_df = prepared.frame.iloc[train_pos]
            test_df = prepared.frame.iloc[test_pos]
            model = create_model(model_type, **(hyperparameters or {}))

            train_dates = pd.DatetimeIndex(sorted(train_df["date"].unique()))
            n_val = (
                math.ceil(self.val_fraction * len(train_dates))
                if self.val_fraction > 0
                else 0
            )

            if n_val > 0 and len(train_dates) - n_val - splitter.gap_dates > 0:
                inner_end = (
                    train_dates[-n_val - splitter.gap_dates - 1]
                    if splitter.gap_dates > 0
                    else train_dates[-n_val - 1]
                )
                inner_df = train_df[train_df["date"] <= inner_end]
                val_df = train_df[train_df["date"] >= train_dates[-n_val]]
                X_train, y_train = (
                    inner_df[prepared.feature_columns],
                    inner_df[prepared.target_column],
                )
                X_val, y_val = (
                    val_df[prepared.feature_columns],
                    val_df[prepared.target_column],
                )
            else:
                X_train, y_train = (
                    train_df[prepared.feature_columns],
                    train_df[prepared.target_column],
                )
                X_val, y_val = None, None

            model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
            y_pred = model.predict(test_df[prepared.feature_columns])
            y_test = test_df[prepared.target_column]

            base_metrics = self.evaluator.evaluate(y_test, y_pred)
            ext_metrics = self.evaluator.extended_metrics(
                y_test, y_pred, dates=test_df["date"]
            )
            all_metrics = {**base_metrics, **ext_metrics}

            logger.info(
                "Fold %s: train=[%s, %s] (%s rows), test=[%s, %s] (%s rows), IC=%.4f",
                fold.fold,
                fold.train_start.date(),
                fold.train_end.date(),
                len(train_df),
                fold.test_start.date(),
                fold.test_end.date(),
                len(test_df),
                base_metrics.get("ic", float("nan")),
            )

            row: dict[str, Any] = {
                "fold": fold.fold,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "n_train_rows": len(train_df),
                "n_test_rows": len(test_df),
            }
            row.update(all_metrics)
            fold_rows.append(row)

        fold_metrics = pd.DataFrame(fold_rows)
        summary: dict[str, float] = {}
        non_metric = {
            "fold",
            "train_start",
            "train_end",
            "test_start",
            "test_end",
            "n_train_rows",
            "n_test_rows",
        }
        for col in [c for c in fold_metrics.columns if c not in non_metric]:
            vals = fold_metrics[col].to_numpy(dtype=float)
            summary[f"mean_{col}"] = float(np.nanmean(vals))
            summary[f"std_{col}"] = float(np.nanstd(vals, ddof=0))

        return WalkForwardResult(fold_metrics=fold_metrics, summary=summary)
