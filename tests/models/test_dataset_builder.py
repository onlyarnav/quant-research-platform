"""
Tests for DatasetBuilder.
"""

from __future__ import annotations

import pytest

import pandas as pd

from src.models.dataset_builder import DatasetBuilder, DatasetSplits


# =============================================================================
# load_all_features()
# =============================================================================

def test_load_all_features_concatenates_files(tmp_features_dir):
    """Assert combined row count equals sum of both files."""
    builder = DatasetBuilder(features_path=tmp_features_dir)
    df = builder.load_all_features()

    # 2 symbols * 20 dates = 40 rows
    assert len(df) == 40
    assert set(df["symbol"].unique()) == {"AAPL", "MSFT"}


def test_load_all_features_raises_on_empty_dir(tmp_path):
    """Point DatasetBuilder at an empty tmp dir, assert ValueError."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    builder = DatasetBuilder(features_path=empty_dir)

    with pytest.raises(ValueError, match="No feature files found"):
        builder.load_all_features()


# =============================================================================
# select_target_column()
# =============================================================================

@pytest.mark.parametrize(
    "horizon, expected",
    [
        (1, "future_return_1d"),
        (5, "future_return_5d"),
        (10, "future_return_10d"),
    ],
)
def test_select_target_column_valid_horizons(horizon, expected):
    """Parametrize over 1, 5, 10, assert correct mapping."""
    builder = DatasetBuilder()
    assert builder.select_target_column(horizon) == expected


def test_select_target_column_invalid_horizon_raises():
    """Assert ValueError for an unsupported horizon."""
    builder = DatasetBuilder()
    with pytest.raises(ValueError, match="Invalid horizon"):
        builder.select_target_column(3)


# =============================================================================
# build()
# =============================================================================

def test_build_returns_dataset_splits_instance(tmp_features_dir):
    """Confirm the return contract."""
    builder = DatasetBuilder(features_path=tmp_features_dir)
    splits = builder.build(horizon=1)
    assert isinstance(splits, DatasetSplits)


def test_build_drops_rows_with_null_target(tmp_features_dir):
    """Assert no NaN in y_train/y_val/y_test."""
    builder = DatasetBuilder(features_path=tmp_features_dir)
    splits = builder.build(horizon=1)

    assert splits.y_train.isna().sum() == 0
    assert splits.y_val.isna().sum() == 0
    assert splits.y_test.isna().sum() == 0


def test_build_drops_rows_with_null_features(tmp_features_dir):
    """Assert no NaN anywhere in X_train/X_val/X_test."""
    builder = DatasetBuilder(features_path=tmp_features_dir)
    splits = builder.build(horizon=1)

    assert splits.X_train.isna().sum().sum() == 0
    assert splits.X_val.isna().sum().sum() == 0
    assert splits.X_test.isna().sum().sum() == 0


def test_build_excludes_symbol_and_date_from_features(tmp_features_dir):
    """Assert 'symbol' and 'date' are not treated as features."""
    builder = DatasetBuilder(features_path=tmp_features_dir)
    splits = builder.build(horizon=1)

    assert "symbol" not in splits.feature_columns
    assert "date" not in splits.feature_columns


def test_build_excludes_other_targets_from_features(tmp_features_dir):
    """Assert none of ALL_TARGET_COLUMNS appear in feature_columns."""
    builder = DatasetBuilder(features_path=tmp_features_dir)
    splits = builder.build(horizon=1)

    for target in DatasetBuilder.ALL_TARGET_COLUMNS:
        assert target not in splits.feature_columns


def test_build_splits_are_chronological(tmp_features_dir):
    """
    Assert X_train, X_val, X_test occupy strictly increasing index ranges.

    Since build() performs sort_values("date").reset_index(drop=True),
    the row indices are strictly chronological, so non-overlapping,
    increasing index ranges confirm the splits are in date order.
    """
    builder = DatasetBuilder(features_path=tmp_features_dir)
    splits = builder.build(horizon=1)

    assert len(splits.X_train) > 0
    assert len(splits.X_val) > 0
    assert len(splits.X_test) > 0

    assert splits.X_train.index.max() < splits.X_val.index.min()
    assert splits.X_val.index.max() < splits.X_test.index.min()


def test_build_invalid_horizon_raises(tmp_features_dir):
    """Assert ValueError propagates from build(horizon=99)."""
    builder = DatasetBuilder(features_path=tmp_features_dir)
    with pytest.raises(ValueError, match="Invalid horizon"):
        builder.build(horizon=99)


def test_build_raises_on_insufficient_unique_dates(tmp_path):
    """
    Assert ValueError when fewer than 3 clean unique dates remain
    after dropping NaN rows.
    """
    features_dir = tmp_path / "features"
    features_dir.mkdir()

    dates = pd.date_range(start="2023-01-01", periods=3, freq="D")
    df = pd.DataFrame(
        {
            "symbol": ["AAPL"] * 3,
            "date": dates,
            "sma_5": [1.0, 2.0, 3.0],
            "future_return_1d": [0.01, None, None],
            "future_return_5d": [0.02, None, None],
            "future_return_10d": [0.03, None, None],
        }
    )
    df.to_parquet(features_dir / "AAPL.parquet", index=False)

    builder = DatasetBuilder(features_path=features_dir)

    with pytest.raises(ValueError, match="Insufficient unique dates"):
        builder.build(horizon=1)