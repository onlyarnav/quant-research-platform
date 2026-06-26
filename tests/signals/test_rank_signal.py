"""
Tests for RankSignalGenerator.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.signals.rank_signal import RankSignalGenerator


# =============================================================================
# __init__() validation
# =============================================================================

def test_init_rejects_top_n_less_than_one():
    with pytest.raises(ValueError, match="top_n must be at least 1"):
        RankSignalGenerator(top_n=0)


def test_init_rejects_negative_bottom_n():
    with pytest.raises(ValueError, match="bottom_n must be non-negative"):
        RankSignalGenerator(top_n=2, bottom_n=-1)


def test_init_default_values():
    generator = RankSignalGenerator()
    assert generator.top_n == 10
    assert generator.bottom_n == 0


# =============================================================================
# generate() input validation
# =============================================================================

def test_generate_raises_on_empty_df():
    generator = RankSignalGenerator()
    with pytest.raises(ValueError, match="cannot be empty"):
        generator.generate(pd.DataFrame())


def test_generate_raises_on_missing_columns():
    generator = RankSignalGenerator()
    predictions_df = pd.DataFrame({
        "symbol": ["AAPL"],
        "date": pd.to_datetime(["2024-01-01"]),
        # missing 'predicted_return'
    })
    with pytest.raises(ValueError, match="missing required columns"):
        generator.generate(predictions_df)


# =============================================================================
# Core ranking logic — single date
# =============================================================================

def test_generate_top_n_get_buy_signal():
    generator = RankSignalGenerator(top_n=2, bottom_n=2)
    predictions_df = pd.DataFrame({
        "symbol": ["A", "B", "C", "D", "E"],
        "date": pd.to_datetime(["2024-01-01"] * 5),
        "predicted_return": [0.05, 0.04, 0.0, -0.02, -0.05],
    })
    output_df = generator.generate(predictions_df)

    top_two = output_df[output_df["symbol"].isin(["A", "B"])]
    assert (top_two["signal"] == 1).all()


def test_generate_bottom_n_get_sell_signal():
    generator = RankSignalGenerator(top_n=2, bottom_n=2)
    predictions_df = pd.DataFrame({
        "symbol": ["A", "B", "C", "D", "E"],
        "date": pd.to_datetime(["2024-01-01"] * 5),
        "predicted_return": [0.05, 0.04, 0.0, -0.02, -0.05],
    })
    output_df = generator.generate(predictions_df)

    bottom_two = output_df[output_df["symbol"].isin(["D", "E"])]
    assert (bottom_two["signal"] == -1).all()


def test_generate_middle_gets_hold_signal():
    generator = RankSignalGenerator(top_n=2, bottom_n=2)
    predictions_df = pd.DataFrame({
        "symbol": ["A", "B", "C", "D", "E"],
        "date": pd.to_datetime(["2024-01-01"] * 5),
        "predicted_return": [0.05, 0.04, 0.0, -0.02, -0.05],
    })
    output_df = generator.generate(predictions_df)

    middle = output_df[output_df["symbol"] == "C"]
    assert middle["signal"].iloc[0] == 0


def test_generate_rank_values_are_correct():
    generator = RankSignalGenerator(top_n=2, bottom_n=2)
    predictions_df = pd.DataFrame({
        "symbol": ["A", "B", "C", "D", "E"],
        "date": pd.to_datetime(["2024-01-01"] * 5),
        "predicted_return": [0.05, 0.04, 0.0, -0.02, -0.05],
    })
    output_df = generator.generate(predictions_df)
    output_df = output_df.set_index("symbol")

    assert output_df.loc["A", "rank"] == 1
    assert output_df.loc["B", "rank"] == 2
    assert output_df.loc["C", "rank"] == 3
    assert output_df.loc["D", "rank"] == 4
    assert output_df.loc["E", "rank"] == 5


def test_generate_default_bottom_n_zero_produces_no_sell_signals():
    generator = RankSignalGenerator(top_n=2, bottom_n=0)
    predictions_df = pd.DataFrame({
        "symbol": ["A", "B", "C", "D", "E"],
        "date": pd.to_datetime(["2024-01-01"] * 5),
        "predicted_return": [0.05, 0.04, 0.0, -0.02, -0.05],
    })
    output_df = generator.generate(predictions_df)
    assert (output_df["signal"] != -1).all()


# =============================================================================
# Cross-sectional independence across multiple dates
# =============================================================================

def test_generate_ranks_independently_per_date():
    generator = RankSignalGenerator(top_n=1, bottom_n=0)
    predictions_df = pd.DataFrame({
        "symbol": ["A", "B", "A", "B"],
        "date": pd.to_datetime(
            ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]
        ),
        "predicted_return": [0.05, 0.01, 0.01, 0.05],
    })
    output_df = generator.generate(predictions_df)
    output_df = output_df.set_index(["date", "symbol"])

    # Day 1: A is top (rank 1, BUY); Day 2: B is top (rank 1, BUY)
    assert output_df.loc[(pd.Timestamp("2024-01-01"), "A"), "signal"] == 1
    assert output_df.loc[(pd.Timestamp("2024-01-01"), "B"), "signal"] == 0
    assert output_df.loc[(pd.Timestamp("2024-01-02"), "B"), "signal"] == 1
    assert output_df.loc[(pd.Timestamp("2024-01-02"), "A"), "signal"] == 0


# =============================================================================
# Confidence
# =============================================================================

def test_generate_confidence_is_absolute_value():
    generator = RankSignalGenerator(top_n=1)
    predictions_df = pd.DataFrame({
        "symbol": ["A", "B"],
        "date": pd.to_datetime(["2024-01-01"] * 2),
        "predicted_return": [-0.05, 0.02],
    })
    output_df = generator.generate(predictions_df)
    output_df = output_df.set_index("symbol")

    assert output_df.loc["A", "confidence"] == 0.05
    assert output_df.loc["B", "confidence"] == 0.02


# =============================================================================
# Output integrity
# =============================================================================

def test_generate_returns_correct_columns():
    generator = RankSignalGenerator(top_n=1)
    predictions_df = pd.DataFrame({
        "symbol": ["A", "B"],
        "date": pd.to_datetime(["2024-01-01"] * 2),
        "predicted_return": [0.05, 0.02],
    })
    output_df = generator.generate(predictions_df)
    expected_cols = {"symbol", "date", "predicted_return", "signal", "confidence", "rank"}
    assert set(output_df.columns) == expected_cols


def test_generate_preserves_row_count():
    generator = RankSignalGenerator(top_n=1)
    predictions_df = pd.DataFrame({
        "symbol": ["A", "B", "C"],
        "date": pd.to_datetime(["2024-01-01"] * 3),
        "predicted_return": [0.05, 0.02, -0.01],
    })
    output_df = generator.generate(predictions_df)
    assert len(output_df) == len(predictions_df)


def test_generate_does_not_mutate_input():
    generator = RankSignalGenerator(top_n=1)
    predictions_df = pd.DataFrame({
        "symbol": ["A", "B"],
        "date": pd.to_datetime(["2024-01-01"] * 2),
        "predicted_return": [0.05, 0.02],
    })
    original_df = predictions_df.copy()

    generator.generate(predictions_df)

    pd.testing.assert_frame_equal(predictions_df, original_df)


def test_generate_handles_ties_deterministically():
    """
    Equal predicted_return values must still receive distinct,
    deterministic ranks (method='first') rather than raising or
    producing duplicate ranks.
    """
    generator = RankSignalGenerator(top_n=1)
    predictions_df = pd.DataFrame({
        "symbol": ["A", "B"],
        "date": pd.to_datetime(["2024-01-01"] * 2),
        "predicted_return": [0.03, 0.03],
    })
    output_df = generator.generate(predictions_df)

    ranks = sorted(output_df["rank"].tolist())
    assert ranks == [1, 2]