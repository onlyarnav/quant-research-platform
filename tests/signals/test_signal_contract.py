"""Contract compliance tests for all signal generators according to Data_Schema.md."""

from __future__ import annotations

import pandas as pd
import pytest

from src.signals.rank_signal import RankSignalGenerator
from src.signals.threshold_signal import ThresholdSignalGenerator
from src.signals.topn_signal import TopNLongShortSignalGenerator

EXPECTED_SIGNAL_COLUMNS = [
    "symbol",
    "date",
    "predicted_return",
    "signal",
    "confidence",
    "rank",
]


@pytest.fixture
def sample_predictions() -> pd.DataFrame:
    """Fixture supplying multi-symbol sample predictions."""
    return pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "date": [pd.Timestamp("2023-01-01")] * 4,
            "predicted_return": [0.08, 0.02, -0.01, -0.06],
        }
    )


def test_signal_contract_threshold(sample_predictions) -> None:
    """Assert ThresholdSignalGenerator complies with Data_Schema.md schema."""
    gen = ThresholdSignalGenerator(threshold=0.03)
    res = gen.generate(sample_predictions)

    assert list(res.columns) == EXPECTED_SIGNAL_COLUMNS
    assert set(res["signal"].unique()).issubset({-1, 0, 1})
    assert (res["confidence"] >= 0.0).all()


def test_signal_contract_rank(sample_predictions) -> None:
    """Assert RankSignalGenerator complies with Data_Schema.md schema."""
    gen = RankSignalGenerator(top_n=1, bottom_n=1)
    res = gen.generate(sample_predictions)

    assert list(res.columns) == EXPECTED_SIGNAL_COLUMNS
    assert set(res["signal"].unique()).issubset({-1, 0, 1})
    assert (res["confidence"] >= 0.0).all()
    assert (res["rank"] >= 1).all()


def test_signal_contract_topn(sample_predictions) -> None:
    """Assert TopNLongShortSignalGenerator complies with Data_Schema.md schema."""
    gen = TopNLongShortSignalGenerator(top_n=1, bottom_n=1)
    res = gen.generate(sample_predictions)

    assert list(res.columns) == EXPECTED_SIGNAL_COLUMNS
    assert set(res["signal"].unique()).issubset({-1, 0, 1})
    assert (res["confidence"] >= 0.0).all()
    assert (res["rank"] >= 1).all()
