"""
Tests for dataset metadata, summaries, and version checksums.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.models.dataset_metadata import (
    DatasetMetadata,
    SplitSummary,
    compute_data_checksum,
    compute_dataset_version,
)


def _sample_splits() -> tuple[SplitSummary, ...]:
    """Helper to create sample SplitSummary instances."""
    return (
        SplitSummary(
            name="train",
            n_rows=100,
            n_symbols=2,
            start_date="2023-01-01T00:00:00",
            end_date="2023-06-01T00:00:00",
        ),
        SplitSummary(
            name="val",
            n_rows=20,
            n_symbols=2,
            start_date="2023-06-02T00:00:00",
            end_date="2023-07-01T00:00:00",
        ),
        SplitSummary(
            name="test",
            n_rows=20,
            n_symbols=2,
            start_date="2023-07-02T00:00:00",
            end_date="2023-08-01T00:00:00",
        ),
    )


# =====================================================================
# compute_data_checksum()
# =====================================================================


def test_compute_data_checksum_is_deterministic() -> None:
    """Assert identical DataFrames yield identical checksums."""
    df1 = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    df2 = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})

    c1 = compute_data_checksum([df1])
    c2 = compute_data_checksum([df2])
    assert c1 == c2


def test_compute_data_checksum_changes_on_value_change() -> None:
    """Assert changing a single value changes the resulting checksum."""
    df1 = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    df2 = pd.DataFrame({"a": [1.0, 2.0001], "b": [3.0, 4.0]})

    c1 = compute_data_checksum([df1])
    c2 = compute_data_checksum([df2])
    assert c1 != c2


# =====================================================================
# compute_dataset_version()
# =====================================================================


def test_compute_dataset_version_returns_12_hex_chars() -> None:
    """Assert version is exactly 12 hexadecimal characters."""
    splits = _sample_splits()
    version = compute_dataset_version(
        horizon=1,
        feature_columns=["f1", "f2"],
        splits=splits,
        data_checksum="abc123def456",
    )
    assert len(version) == 12
    assert all(c in "0123456789abcdef" for c in version)


def test_compute_dataset_version_is_stable_and_sensitive() -> None:
    """Assert stability and sensitivity to horizon, feature_columns, splits, and checksum."""
    splits = _sample_splits()
    v_base = compute_dataset_version(1, ["f1", "f2"], splits, "chk1")
    v_identical = compute_dataset_version(1, ["f1", "f2"], splits, "chk1")
    assert v_base == v_identical

    # Change horizon
    assert compute_dataset_version(5, ["f1", "f2"], splits, "chk1") != v_base
    # Change feature columns
    assert compute_dataset_version(1, ["f1", "f3"], splits, "chk1") != v_base
    # Change checksum
    assert compute_dataset_version(1, ["f1", "f2"], splits, "chk2") != v_base


# =====================================================================
# DatasetMetadata
# =====================================================================


def test_dataset_metadata_to_dict_is_json_serializable() -> None:
    """Assert to_dict() returns a valid dictionary serializable via json.dumps."""
    splits = _sample_splits()
    meta = DatasetMetadata(
        horizon=1,
        target_column="future_return_1d",
        feature_columns=("f1", "f2"),
        feature_set_version="v1",
        embargo_dates=0,
        lookback_dates=None,
        splits=splits,
        dataset_version="a1b2c3d4e5f6",
    )
    d = meta.to_dict()
    assert isinstance(d, dict)
    assert d["feature_columns"] == ["f1", "f2"]
    # Verify json.dumps succeeds
    serialized = json.dumps(d)
    assert "a1b2c3d4e5f6" in serialized


def test_dataset_metadata_split_lookup() -> None:
    """Assert split() retrieves correct split summary and raises KeyError for unknown name."""
    splits = _sample_splits()
    meta = DatasetMetadata(
        horizon=1,
        target_column="future_return_1d",
        feature_columns=("f1", "f2"),
        feature_set_version="v1",
        embargo_dates=0,
        lookback_dates=None,
        splits=splits,
        dataset_version="a1b2c3d4e5f6",
    )
    train_summary = meta.split("train")
    assert train_summary.name == "train"
    assert train_summary.n_rows == 100

    with pytest.raises(KeyError, match="Split 'oos' not found"):
        meta.split("oos")
