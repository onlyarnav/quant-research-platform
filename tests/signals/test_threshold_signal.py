"""
Tests for ThresholdSignalGenerator.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.signals.threshold_signal import ThresholdSignalGenerator


# =============================================================================
# __init__() validation
# =============================================================================

def test_init_rejects_negative_threshold():
    with pytest.raises(ValueError, match="non-negative"):
        ThresholdSignalGenerator(threshold=-0.01)


def test_init_accepts_zero_threshold():
    generator = ThresholdSignalGenerator(threshold=0.0)
    assert generator.threshold == 0.0


def test_init_default_threshold_is_001():
    generator = ThresholdSignalGenerator()
    assert generator.threshold == 0.01


# =============================================================================
# generate() input validation
# =============================================================================

def test_generate_raises_on_empty_df():
    generator = ThresholdSignalGenerator()
    with pytest.raises(ValueError, match="cannot be empty"):
        generator.generate(pd.DataFrame())


def test_generate_raises_on_missing_columns():
    generator = ThresholdSignalGenerator()
    predictions_df = pd.DataFrame({
        "symbol": ["AAPL"],
        "date": pd.to_datetime(["2024-01-01"]),
        # missing 'predicted_return'
    })
    with pytest.raises(ValueError, match="missing required columns"):
        generator.generate(predictions_df)


# =============================================================================
# Core signal logic
# =============================================================================

def test_generate_buy_signal_above_threshold():
    generator = ThresholdSignalGenerator(threshold=0.01)
    predictions_df = pd.DataFrame({
        "symbol": ["AAPL"],
        "date": pd.to_datetime(["2024-01-01"]),
        "predicted_return": [0.02],
    })
    output_df = generator.generate(predictions_df)
    assert output_df["signal"].iloc[0] == 1


def test_generate_sell_signal_below_negative_threshold():
    generator = ThresholdSignalGenerator(threshold=0.01)
    predictions_df = pd.DataFrame({
        "symbol": ["AAPL"],
        "date": pd.to_datetime(["2024-01-01"]),
        "predicted_return": [-0.02],
    })
    output_df = generator.generate(predictions_df)
    assert output_df["signal"].iloc[0] == -1


def test_generate_hold_signal_within_threshold():
    generator = ThresholdSignalGenerator(threshold=0.01)
    predictions_df = pd.DataFrame({
        "symbol": ["AAPL"],
        "date": pd.to_datetime(["2024-01-01"]),
        "predicted_return": [0.005],
    })
    output_df = generator.generate(predictions_df)
    assert output_df["signal"].iloc[0] == 0


def test_generate_hold_at_exact_positive_threshold_boundary():
    generator = ThresholdSignalGenerator(threshold=0.01)
    predictions_df = pd.DataFrame({
        "symbol": ["AAPL"],
        "date": pd.to_datetime(["2024-01-01"]),
        "predicted_return": [0.01],
    })
    output_df = generator.generate(predictions_df)
    assert output_df["signal"].iloc[0] == 0


def test_generate_hold_at_exact_negative_threshold_boundary():
    generator = ThresholdSignalGenerator(threshold=0.01)
    predictions_df = pd.DataFrame({
        "symbol": ["AAPL"],
        "date": pd.to_datetime(["2024-01-01"]),
        "predicted_return": [-0.01],
    })
    output_df = generator.generate(predictions_df)
    assert output_df["signal"].iloc[0] == 0


def test_generate_buy_just_above_boundary():
    generator = ThresholdSignalGenerator(threshold=0.01)
    predictions_df = pd.DataFrame({
        "symbol": ["AAPL"],
        "date": pd.to_datetime(["2024-01-01"]),
        "predicted_return": [0.0100001],
    })
    output_df = generator.generate(predictions_df)
    assert output_df["signal"].iloc[0] == 1


# =============================================================================
# Confidence and rank
# =============================================================================

def test_generate_confidence_is_absolute_value():
    generator = ThresholdSignalGenerator(threshold=0.01)
    predictions_df = pd.DataFrame({
        "symbol": ["AAPL"],
        "date": pd.to_datetime(["2024-01-01"]),
        "predicted_return": [-0.05],
    })
    output_df = generator.generate(predictions_df)
    assert output_df["confidence"].iloc[0] == 0.05


def test_generate_rank_is_always_none():
    generator = ThresholdSignalGenerator(threshold=0.01)
    predictions_df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
        "predicted_return": [0.02, -0.02],
    })
    output_df = generator.generate(predictions_df)
    assert output_df["rank"].isna().all()


# =============================================================================
# Output integrity
# =============================================================================

def test_generate_returns_correct_columns():
    generator = ThresholdSignalGenerator(threshold=0.01)
    predictions_df = pd.DataFrame({
        "symbol": ["AAPL"],
        "date": pd.to_datetime(["2024-01-01"]),
        "predicted_return": [0.02],
    })
    output_df = generator.generate(predictions_df)
    expected_cols = {"symbol", "date", "predicted_return", "signal", "confidence", "rank"}
    assert set(output_df.columns) == expected_cols


def test_generate_preserves_row_count():
    generator = ThresholdSignalGenerator(threshold=0.01)
    predictions_df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "GOOG"],
        "date": pd.to_datetime(["2024-01-01"] * 3),
        "predicted_return": [0.02, -0.02, 0.005],
    })
    output_df = generator.generate(predictions_df)
    assert len(output_df) == len(predictions_df)


def test_generate_does_not_mutate_input():
    generator = ThresholdSignalGenerator(threshold=0.01)
    predictions_df = pd.DataFrame({
        "symbol": ["AAPL"],
        "date": pd.to_datetime(["2024-01-01"]),
        "predicted_return": [0.02],
    })
    original_df = predictions_df.copy()

    generator.generate(predictions_df)

    pd.testing.assert_frame_equal(predictions_df, original_df)


def test_generate_mixed_signals_in_single_call():
    generator = ThresholdSignalGenerator(threshold=0.01)
    predictions_df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "GOOG"],
        "date": pd.to_datetime(["2024-01-01"] * 3),
        "predicted_return": [0.02, 0.005, -0.02],
    })
    output_df = generator.generate(predictions_df)
    assert list(output_df["signal"]) == [1, 0, -1]