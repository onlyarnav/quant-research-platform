"""Unit tests for RegimeClassifier protocol and VolatilityRegimeClassifier."""

from __future__ import annotations

import pandas as pd
import pytest

from src.models.regime import RegimeClassifier, VolatilityRegimeClassifier

# ==============================================================================
# Protocol & Initialization
# ==============================================================================


def test_regime_classifier_protocol() -> None:
    """Assert VolatilityRegimeClassifier satisfies RegimeClassifier protocol."""
    classifier = VolatilityRegimeClassifier()
    assert isinstance(classifier, RegimeClassifier)


def test_classifier_invalid_min_periods() -> None:
    """Assert constructor raises ValueError for invalid min_periods."""
    with pytest.raises(ValueError, match="min_periods must be at least 1"):
        VolatilityRegimeClassifier(min_periods=0)


# ==============================================================================
# Input Validation
# ==============================================================================


def test_classifier_empty_frame_raises() -> None:
    """Assert empty DataFrame raises ValueError."""
    classifier = VolatilityRegimeClassifier()
    with pytest.raises(ValueError, match="features_df cannot be empty."):
        classifier.classify(pd.DataFrame())


def test_classifier_missing_columns_raises() -> None:
    """Assert missing required columns raise ValueError."""
    classifier = VolatilityRegimeClassifier(volatility_column="volatility_20")
    df = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=5)})
    with pytest.raises(
        ValueError, match="features_df missing required columns: \\['volatility_20'\\]"
    ):
        classifier.classify(df)


# ==============================================================================
# Classification Logic & Lookahead Protection
# ==============================================================================


def test_classifier_golden_values_and_no_lookahead() -> None:
    """Assert classification matches hand-computed expanding median without lookahead."""
    dates = pd.date_range("2023-01-01", periods=6, freq="D")
    # Volatilities for 6 consecutive dates:
    # d0: 10 (past: 0 obs -> unknown)
    # d1: 20 (past: 1 obs -> unknown)
    # d2: 30 (past: 2 obs -> unknown)
    # d3: 40 (past: [10, 20, 30], median=20. 40 > 20 -> high_vol)
    # d4: 15 (past: [10, 20, 30, 40], median=25. 15 <= 25 -> low_vol)
    # d5: 25 (past: [10, 20, 30, 40, 15], median=20. 25 > 20 -> high_vol)
    df = pd.DataFrame(
        {
            "symbol": ["A"] * 6,
            "date": dates,
            "volatility_20": [10.0, 20.0, 30.0, 40.0, 15.0, 25.0],
        }
    )

    classifier = VolatilityRegimeClassifier(
        volatility_column="volatility_20", min_periods=3
    )
    result = classifier.classify(df)

    assert result.iloc[0] == "unknown"
    assert result.iloc[1] == "unknown"
    assert result.iloc[2] == "unknown"
    assert result.iloc[3] == "high_vol"
    assert result.iloc[4] == "low_vol"
    assert result.iloc[5] == "high_vol"


def test_classifier_cross_sectional_aggregation() -> None:
    """Assert multiple symbols on the same date are averaged cross-sectionally."""
    dates = pd.date_range("2023-01-01", periods=4, freq="D")
    df = pd.DataFrame(
        {
            "symbol": ["A", "B"] * 4,
            "date": list(dates) + list(dates),
            "volatility_20": [10.0, 30.0, 20.0, 40.0, 50.0, 70.0, 10.0, 10.0],
            # Mean per date:
            # d0: (10 + 50)/2 = 30
            # d1: (30 + 70)/2 = 50
            # d2: (20 + 10)/2 = 15
            # d3: (40 + 10)/2 = 25
        }
    )
    classifier = VolatilityRegimeClassifier(min_periods=2)
    result = classifier.classify(df)

    assert len(result) == 4
    assert result.iloc[0] == "unknown"
    assert result.iloc[1] == "unknown"
    # Past for d2: [30, 50], median = 40. d2 val = 15 <= 40 -> low_vol
    assert result.iloc[2] == "low_vol"
    # Past for d3: [30, 50, 15], median = 30. d3 val = 25 <= 30 -> low_vol
    assert result.iloc[3] == "low_vol"


def test_classifier_no_mutation() -> None:
    """Assert classify does not mutate input DataFrame."""
    df = pd.DataFrame(
        {
            "symbol": ["A"] * 5,
            "date": pd.date_range("2023-01-01", periods=5),
            "volatility_20": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    df_copy = df.copy()
    classifier = VolatilityRegimeClassifier(min_periods=2)
    classifier.classify(df)
    pd.testing.assert_frame_equal(df, df_copy)
