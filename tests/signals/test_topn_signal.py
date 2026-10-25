"""Unit tests for TopNLongShortSignalGenerator."""

from __future__ import annotations

import pandas as pd
import pytest

from src.signals.topn_signal import TopNLongShortSignalGenerator

# ==============================================================================
# Initialization & Validation
# ==============================================================================


def test_init_validation() -> None:
    """Assert constructor validates top_n and bottom_n parameters."""
    with pytest.raises(ValueError, match="top_n must be at least 1"):
        TopNLongShortSignalGenerator(top_n=0)

    with pytest.raises(ValueError, match="bottom_n must be at least 1"):
        TopNLongShortSignalGenerator(top_n=5, bottom_n=0)

    # Defaults: bottom_n defaults to top_n
    gen = TopNLongShortSignalGenerator(top_n=5)
    assert gen.top_n == 5
    assert gen.bottom_n == 5


def test_generate_validation() -> None:
    """Assert generate validates empty DataFrame and required columns."""
    gen = TopNLongShortSignalGenerator(top_n=2)
    with pytest.raises(ValueError, match="predictions_df cannot be empty."):
        gen.generate(pd.DataFrame())

    df_missing = pd.DataFrame({"symbol": ["A"], "predicted_return": [0.05]})
    with pytest.raises(
        ValueError, match="predictions_df missing required columns: \\['date'\\]"
    ):
        gen.generate(df_missing)


# ==============================================================================
# Overlap Prevention & Golden Logic
# ==============================================================================


def test_overlap_prevention_and_golden_signals() -> None:
    """Assert top_n and bottom_n are capped at size // 2, preventing portfolio overlap."""
    # 5 symbols on a single date:
    # returns: A: 0.10, B: 0.05, C: 0.00, D: -0.05, E: -0.10
    # Requested top_n=3, bottom_n=3
    # size = 5 -> size // 2 = 2.
    # Long cutoff: k_long = min(3, 2) = 2 -> A (rank 1), B (rank 2) = +1
    # Short cutoff: k_short = min(3, 2) = 2 -> rank > 5 - 2 = 3 -> D (rank 4), E (rank 5) = -1
    # Middle: C (rank 3) = 0 (HOLD)
    df = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D", "E"],
            "date": [pd.Timestamp("2023-01-01")] * 5,
            "predicted_return": [0.10, 0.05, 0.00, -0.05, -0.10],
        }
    )
    gen = TopNLongShortSignalGenerator(top_n=3, bottom_n=3)
    res = gen.generate(df)

    assert len(res) == 5
    assert list(res["symbol"]) == ["A", "B", "C", "D", "E"]
    assert list(res["signal"]) == [1, 1, 0, -1, -1]
    assert list(res["rank"]) == [1, 2, 3, 4, 5]
    assert list(res["confidence"]) == [0.10, 0.05, 0.00, 0.05, 0.10]


def test_single_symbol_date_gets_hold() -> None:
    """Assert a date with a single symbol receives HOLD (0) because size // 2 == 0."""
    df = pd.DataFrame(
        {
            "symbol": ["SOLO"],
            "date": [pd.Timestamp("2023-01-01")],
            "predicted_return": [0.25],
        }
    )
    gen = TopNLongShortSignalGenerator(top_n=5, bottom_n=5)
    res = gen.generate(df)

    assert len(res) == 1
    assert res.iloc[0]["signal"] == 0
    assert res.iloc[0]["rank"] == 1


def test_independent_per_date_and_ties() -> None:
    """Assert ranking operates independently per date and breaks ties deterministically."""
    df = pd.DataFrame(
        {
            "symbol": ["A", "B", "A", "B"],
            "date": [
                pd.Timestamp("2023-01-01"),
                pd.Timestamp("2023-01-01"),
                pd.Timestamp("2023-01-02"),
                pd.Timestamp("2023-01-02"),
            ],
            "predicted_return": [0.10, -0.10, 0.05, 0.05],
        }
    )
    gen = TopNLongShortSignalGenerator(top_n=1, bottom_n=1)
    res = gen.generate(df)

    # Date 1: A=0.10 (rank 1 -> 1), B=-0.10 (rank 2 -> -1)
    d1 = res[res["date"] == pd.Timestamp("2023-01-01")].sort_values("rank")
    assert list(d1["signal"]) == [1, -1]

    # Date 2: tie at 0.05; method="first" assigns rank 1 to A (row 2), rank 2 to B (row 3)
    d2 = res[res["date"] == pd.Timestamp("2023-01-02")].sort_values("rank")
    assert list(d2["rank"]) == [1, 2]
    assert list(d2["signal"]) == [1, -1]


def test_no_mutation() -> None:
    """Assert generate does not mutate input DataFrame."""
    df = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "date": [pd.Timestamp("2023-01-01")] * 2,
            "predicted_return": [0.05, -0.05],
        }
    )
    df_copy = df.copy()
    gen = TopNLongShortSignalGenerator(top_n=1)
    gen.generate(df)
    pd.testing.assert_frame_equal(df, df_copy)
