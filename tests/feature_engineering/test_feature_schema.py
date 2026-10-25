"""
Tests for feature schema definitions and validation helpers.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.feature_engineering.feature_schema import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    get_feature_columns,
    validate_feature_frame,
)

# =====================================================================
# Constants verification
# =====================================================================


def test_feature_columns_has_26_unique_names() -> None:
    """Assert FEATURE_COLUMNS contains exactly 26 unique technical feature names."""
    assert len(FEATURE_COLUMNS) == 26
    assert len(set(FEATURE_COLUMNS)) == 26


def test_feature_columns_matches_schema_order() -> None:
    """Assert FEATURE_COLUMNS matches the exact canonical order in Data_Schema.md."""
    expected = (
        "sma_5",
        "sma_10",
        "sma_20",
        "sma_50",
        "sma_200",
        "ema_10",
        "ema_20",
        "ema_50",
        "rsi_14",
        "macd",
        "macd_signal",
        "roc_5",
        "roc_10",
        "momentum_5",
        "momentum_10",
        "volatility_10",
        "volatility_20",
        "volatility_50",
        "atr_14",
        "bollinger_width",
        "zscore_20",
        "distance_from_sma",
        "distance_from_ema",
        "volume_ma_20",
        "volume_ratio",
        "obv",
    )
    assert FEATURE_COLUMNS == expected


def test_target_columns_has_three_horizons() -> None:
    """Assert TARGET_COLUMNS contains exactly the three forward return horizons."""
    assert TARGET_COLUMNS == (
        "future_return_1d",
        "future_return_5d",
        "future_return_10d",
    )


# =====================================================================
# get_feature_columns()
# =====================================================================


def test_get_feature_columns_drops_identifiers_and_targets() -> None:
    """Assert identifiers and targets are filtered out while preserving order."""
    cols = [
        "symbol",
        "sma_5",
        "date",
        "rsi_14",
        "future_return_1d",
        "obv",
        "future_return_10d",
    ]
    extracted = get_feature_columns(cols)
    assert extracted == ["sma_5", "rsi_14", "obv"]


def test_get_feature_columns_preserves_custom_features() -> None:
    """Assert arbitrary feature columns are preserved if not in excluded set."""
    cols = ["custom_alpha", "symbol", "feature_x"]
    assert get_feature_columns(cols) == ["custom_alpha", "feature_x"]


# =====================================================================
# validate_feature_frame()
# =====================================================================


def _make_complete_feature_frame() -> pd.DataFrame:
    """Helper to create a 1-row valid feature DataFrame."""
    data = {col: [1.0] for col in FEATURE_COLUMNS}
    data["symbol"] = ["RELIANCE"]
    data["date"] = [pd.Timestamp("2024-01-01")]
    for target in TARGET_COLUMNS:
        data[target] = [0.01]
    return pd.DataFrame(data)


def test_validate_feature_frame_passes_complete_frame() -> None:
    """Assert complete frame passes validation without error."""
    df = _make_complete_feature_frame()
    validate_feature_frame(df, require_targets=True)


def test_validate_feature_frame_raises_on_empty() -> None:
    """Assert empty DataFrame raises ValueError with expected message."""
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="Feature frame cannot be empty."):
        validate_feature_frame(empty_df)


def test_validate_feature_frame_raises_on_missing_feature() -> None:
    """Assert missing feature column raises ValueError with missing column name."""
    df = _make_complete_feature_frame().drop(columns=["sma_50"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_feature_frame(df)


def test_validate_feature_frame_raises_on_missing_target_when_required() -> None:
    """Assert missing target column raises ValueError when require_targets is True."""
    df = _make_complete_feature_frame().drop(columns=["future_return_1d"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_feature_frame(df, require_targets=True)


def test_validate_feature_frame_allows_missing_target_when_not_required() -> None:
    """Assert missing target column is accepted when require_targets is False."""
    df = _make_complete_feature_frame().drop(columns=list(TARGET_COLUMNS))
    validate_feature_frame(df, require_targets=False)
