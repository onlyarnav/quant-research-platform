"""Unit tests for WalkForwardSplitter and WalkForwardValidator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.walk_forward import (
    WalkForwardFold,
    WalkForwardResult,
    WalkForwardSplitter,
    WalkForwardValidator,
)

# ==============================================================================
# WalkForwardSplitter.split_dates()
# ==============================================================================


def test_splitter_golden_expanding_dates() -> None:
    """Assert golden split indices with expanding window and gap dates."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    splitter = WalkForwardSplitter(
        n_splits=3,
        train_window_dates=None,
        gap_dates=2,
        min_train_dates=10,
    )
    folds = splitter.split_dates(dates)

    assert len(folds) == 3
    # N = 100, n_splits = 3, W = 100 // 4 = 25
    # fold 0: test: [25, 49], train: [0, 25 - 2 - 1 = 22]
    assert folds[0].fold == 0
    assert folds[0].train_start == dates[0]
    assert folds[0].train_end == dates[22]
    assert folds[0].test_start == dates[25]
    assert folds[0].test_end == dates[49]
    assert folds[0].n_train_dates == 23
    assert folds[0].n_test_dates == 25

    # fold 1: test: [50, 74], train: [0, 50 - 2 - 1 = 47]
    assert folds[1].fold == 1
    assert folds[1].train_start == dates[0]
    assert folds[1].train_end == dates[47]
    assert folds[1].test_start == dates[50]
    assert folds[1].test_end == dates[74]
    assert folds[1].n_train_dates == 48
    assert folds[1].n_test_dates == 25

    # fold 2: test: [75, 99], train: [0, 75 - 2 - 1 = 72]
    assert folds[2].fold == 2
    assert folds[2].train_start == dates[0]
    assert folds[2].train_end == dates[72]
    assert folds[2].test_start == dates[75]
    assert folds[2].test_end == dates[99]
    assert folds[2].n_train_dates == 73
    assert folds[2].n_test_dates == 25


def test_splitter_rolling_window_dates() -> None:
    """Assert rolling training window size is respected."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    splitter = WalkForwardSplitter(
        n_splits=2,
        train_window_dates=15,
        gap_dates=1,
        min_train_dates=10,
    )
    folds = splitter.split_dates(dates)
    assert len(folds) == 2
    # Both folds have at most train_window_dates training dates
    assert folds[0].n_train_dates == 15
    assert folds[1].n_train_dates == 15


def test_splitter_test_windows_contiguous_and_disjoint() -> None:
    """Assert test windows are contiguous without overlapping."""
    dates = pd.date_range("2023-01-01", periods=90, freq="D")
    splitter = WalkForwardSplitter(n_splits=3, gap_dates=0, min_train_dates=5)
    folds = splitter.split_dates(dates)

    for i in range(len(folds) - 1):
        # Next test start is exactly 1 day after current test end
        assert folds[i + 1].test_start == folds[i].test_end + pd.Timedelta(days=1)


# ==============================================================================
# Input Validation & Errors
# ==============================================================================


def test_splitter_invalid_args() -> None:
    """Assert invalid constructor arguments raise ValueError."""
    with pytest.raises(ValueError, match="n_splits must be at least 1"):
        WalkForwardSplitter(n_splits=0)

    with pytest.raises(ValueError, match="gap_dates must be non-negative"):
        WalkForwardSplitter(gap_dates=-1)

    with pytest.raises(ValueError, match="train_window_dates must be at least 1"):
        WalkForwardSplitter(train_window_dates=0)

    with pytest.raises(ValueError, match="min_train_dates must be at least 1"):
        WalkForwardSplitter(min_train_dates=0)


def test_splitter_insufficient_dates() -> None:
    """Assert error when total dates cannot accommodate splits."""
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    splitter = WalkForwardSplitter(n_splits=10)
    with pytest.raises(
        ValueError, match="Not enough dates for the requested number of splits."
    ):
        splitter.split_dates(dates)


def test_splitter_insufficient_history_for_min_train() -> None:
    """Assert error when training window is smaller than min_train_dates."""
    dates = pd.date_range("2023-01-01", periods=50, freq="D")
    splitter = WalkForwardSplitter(n_splits=2, min_train_dates=40, gap_dates=5)
    with pytest.raises(ValueError, match="Insufficient history for fold 0"):
        splitter.split_dates(dates)


# ==============================================================================
# WalkForwardSplitter.split(df)
# ==============================================================================


def test_splitter_split_df_validation() -> None:
    """Assert split(df) validates empty DataFrame and missing date column."""
    splitter = WalkForwardSplitter()
    with pytest.raises(ValueError, match="df cannot be empty."):
        list(splitter.split(pd.DataFrame()))

    with pytest.raises(ValueError, match="df missing required columns: {'date'}"):
        list(splitter.split(pd.DataFrame({"x": [1, 2, 3]})))


def test_splitter_split_df_indices() -> None:
    """Assert split(df) produces correct row indices without reordering."""
    dates = pd.date_range("2023-01-01", periods=80, freq="D")
    df = pd.DataFrame(
        {
            "symbol": ["A"] * 80 + ["B"] * 80,
            "date": list(dates) + list(dates),
            "val": np.arange(160),
        }
    )
    splitter = WalkForwardSplitter(n_splits=2, gap_dates=1, min_train_dates=10)
    folds = list(splitter.split(df))

    assert len(folds) == 2
    for fold, train_pos, test_pos in folds:
        assert isinstance(fold, WalkForwardFold)
        assert len(train_pos) > 0
        assert len(test_pos) > 0
        # Disjoint row indices
        assert len(set(train_pos).intersection(set(test_pos))) == 0


# ==============================================================================
# WalkForwardValidator.run()
# ==============================================================================


def test_validator_run(feature_frame) -> None:
    """Assert validator runs across folds, returns valid result, and does not mutate input."""
    df_copy = feature_frame.copy()
    splitter = WalkForwardSplitter(n_splits=2, gap_dates=1, min_train_dates=20)
    validator = WalkForwardValidator(splitter=splitter, val_fraction=0.1)

    result = validator.run(
        df=feature_frame,
        model_type="linear",
        horizon=1,
    )

    assert isinstance(result, WalkForwardResult)
    assert len(result.fold_metrics) == 2
    assert "mean_mse" in result.summary
    assert "std_mse" in result.summary
    assert "mean_ic" in result.summary
    assert "mean_rank_ic" in result.summary

    # Ensure fold metrics contain expected columns
    for col in ["fold", "train_start", "train_end", "test_start", "test_end", "ic"]:
        assert col in result.fold_metrics.columns

    # Verify no mutation of input
    pd.testing.assert_frame_equal(feature_frame, df_copy)
