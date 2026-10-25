"""
Extended tests for DatasetBuilder: prepare_frame, metadata, embargo, lookback, and tz.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.dataset_builder import DatasetBuilder, DatasetSplits, PreparedFrame
from src.models.split_utils import align_timestamp

# =====================================================================
# prepare_frame()
# =====================================================================


def test_prepare_frame_drops_nan_targets_and_features_and_sorts() -> None:
    """Assert prepare_frame drops rows with NaN target or features and sorts by date, symbol."""
    builder = DatasetBuilder()
    df = pd.DataFrame(
        {
            "symbol": ["MSFT", "AAPL", "AAPL", "MSFT"],
            "date": ["2023-01-02", "2023-01-02", "2023-01-01", "2023-01-01"],
            "f1": [1.0, 2.0, np.nan, 4.0],  # row 2 has NaN feature
            "future_return_1d": [0.01, np.nan, 0.02, 0.03],  # row 1 has NaN target
        }
    )
    # Rows 1 and 2 dropped -> only rows 0 and 3 remain
    prep = builder.prepare_frame(df, horizon=1)
    assert isinstance(prep, PreparedFrame)
    assert len(prep.frame) == 2
    # Sorted by date ascending: 2023-01-01 (MSFT), then 2023-01-02 (MSFT)
    assert prep.frame.loc[0, "symbol"] == "MSFT"
    assert prep.frame.loc[0, "date"] == pd.Timestamp("2023-01-01")
    assert prep.frame.loc[1, "symbol"] == "MSFT"
    assert prep.frame.loc[1, "date"] == pd.Timestamp("2023-01-02")
    assert prep.feature_columns == ["f1"]
    assert prep.target_column == "future_return_1d"


def test_prepare_frame_raises_on_missing_required_columns() -> None:
    """Assert prepare_frame raises ValueError when symbol, date, or target is missing."""
    builder = DatasetBuilder()
    df = pd.DataFrame({"symbol": ["AAPL"], "date": ["2023-01-01"]})
    with pytest.raises(ValueError, match="missing required columns"):
        builder.prepare_frame(df, horizon=1)


# =====================================================================
# build() metadata and split properties
# =====================================================================


def test_build_populates_metadata_and_aligned_meta_frames(
    feature_frame, feature_dir_factory
) -> None:
    """Assert build populates metadata and meta_* frames aligned with X_* indices."""
    feat_dir = feature_dir_factory(feature_frame)
    builder = DatasetBuilder(features_path=feat_dir)
    splits = builder.build(horizon=1)

    assert splits.metadata is not None
    assert splits.metadata.dataset_version is not None
    assert len(splits.metadata.dataset_version) == 12

    # Check index alignment of meta_* and X_*
    pd.testing.assert_index_equal(splits.X_train.index, splits.meta_train.index)
    pd.testing.assert_index_equal(splits.X_val.index, splits.meta_val.index)
    pd.testing.assert_index_equal(splits.X_test.index, splits.meta_test.index)

    # When embargo_dates=0, sum of split rows equals total rows in prepared frame
    total_split_rows = len(splits.X_train) + len(splits.X_val) + len(splits.X_test)
    raw_df = builder.load_all_features()
    prepared = builder.prepare_frame(raw_df, 1)
    assert total_split_rows == len(prepared.frame)


def test_dataset_version_determinism_and_sensitivity(
    feature_frame, feature_dir_factory
) -> None:
    """Assert dataset_version is deterministic across identical builds and changes when data changes."""
    feat_dir1 = feature_dir_factory(feature_frame)
    b1 = DatasetBuilder(features_path=feat_dir1)
    s1 = b1.build(horizon=1)
    s2 = b1.build(horizon=1)
    assert s1.metadata.dataset_version == s2.metadata.dataset_version

    # Alter data slightly
    altered_frame = feature_frame.copy()
    altered_frame.loc[0, "f1"] += 1.0
    feat_dir2 = feature_dir_factory(altered_frame)
    b2 = DatasetBuilder(features_path=feat_dir2)
    s3 = b2.build(horizon=1)
    assert s1.metadata.dataset_version != s3.metadata.dataset_version


# =====================================================================
# embargo_dates
# =====================================================================


def test_build_with_embargo_dates_purges_boundary_rows(
    feature_frame, feature_dir_factory
) -> None:
    """Assert embargo_dates=k removes k dates before val and test starts, purging rows."""
    feat_dir = feature_dir_factory(feature_frame)
    builder = DatasetBuilder(features_path=feat_dir)

    s_default = builder.build(horizon=1, embargo_dates=0)
    k = 3
    s_embargo = builder.build(horizon=1, embargo_dates=k)

    # Total rows in embargoed splits must be strictly less than un-embargoed
    total_default = (
        len(s_default.X_train) + len(s_default.X_val) + len(s_default.X_test)
    )
    total_embargo = (
        len(s_embargo.X_train) + len(s_embargo.X_val) + len(s_embargo.X_test)
    )
    assert total_embargo < total_default

    # Verify gap between train end and val start
    train_end = pd.Timestamp(s_embargo.metadata.split("train").end_date)
    val_start = pd.Timestamp(s_embargo.metadata.split("val").start_date)
    # The dates between train_end and val_start are purged
    raw_df = builder.load_all_features()
    unique_dates = sorted(raw_df["date"].unique())
    train_end_idx = unique_dates.index(train_end)
    val_start_idx = unique_dates.index(val_start)
    assert val_start_idx - train_end_idx > k


# =====================================================================
# lookback_dates
# =====================================================================


def test_build_with_lookback_dates(feature_frame, feature_dir_factory) -> None:
    """Assert lookback_dates=30 retains only the last 30 unique dates."""
    feat_dir = feature_dir_factory(feature_frame)
    builder = DatasetBuilder(features_path=feat_dir)
    splits = builder.build(horizon=1, lookback_dates=30)

    # Check total unique dates across splits
    all_dates = pd.concat(
        [splits.meta_train["date"], splits.meta_val["date"], splits.meta_test["date"]]
    ).unique()
    assert len(all_dates) == 30


def test_build_with_lookback_dates_too_small_raises(
    feature_frame, feature_dir_factory
) -> None:
    """Assert lookback_dates=2 raises ValueError with Insufficient unique dates."""
    feat_dir = feature_dir_factory(feature_frame)
    builder = DatasetBuilder(features_path=feat_dir)
    with pytest.raises(ValueError, match="Insufficient unique dates"):
        builder.build(horizon=1, lookback_dates=2)


# =====================================================================
# Timezone support & Backward Compatibility
# =====================================================================


def test_build_with_timezone_aware_data(feature_frame_tz, feature_dir_factory) -> None:
    """Assert tz-aware frame builds without error and metadata dates round-trip."""
    feat_dir = feature_dir_factory(feature_frame_tz)
    builder = DatasetBuilder(features_path=feat_dir)
    splits = builder.build(horizon=1)

    assert splits.metadata is not None
    train_end_str = splits.metadata.split("train").end_date
    assert train_end_str is not None
    ts = pd.Timestamp(train_end_str)
    aligned = align_timestamp(ts, splits.meta_train["date"])
    assert aligned.tz == splits.meta_train["date"].dt.tz


def test_build_backward_compatibility_positional_7_args() -> None:
    """Assert DatasetSplits positional construction with 7 args continues to work."""
    X = pd.DataFrame({"f1": [1.0]})
    y = pd.Series([0.1])
    splits = DatasetSplits(X, y, X, y, X, y, ["f1"])
    assert splits.meta_train is None
    assert splits.metadata is None
    assert splits.feature_columns == ["f1"]
