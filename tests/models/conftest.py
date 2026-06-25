"""
Pytest fixtures for ML dataset builder testing.

Provides a temporary feature directory containing synthetic parquet
files used to exercise DatasetBuilder without touching real data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tmp_features_dir(tmp_path):
    """
    Build a temporary feature directory with synthetic parquet files.

    Creates 2 symbols, each with 20 unique dates, including:
    - NaN feature rows (warm-up simulation, rows 0-2)
    - NaN target rows (end-of-series simulation, rows 17-19), applied
      partially across all three target columns so no horizon ends up
      with a fully-NaN target column.
    """
    features_dir = tmp_path / "features"
    features_dir.mkdir()

    rng = np.random.default_rng(42)
    symbols = ["AAPL", "MSFT"]
    dates = pd.date_range(start="2023-01-01", periods=20, freq="D")

    for symbol in symbols:
        df = pd.DataFrame(
            {
                "symbol": [symbol] * len(dates),
                "date": dates,
                "sma_5": rng.standard_normal(len(dates)),
                "rsi_14": rng.uniform(0, 100, len(dates)),
                "future_return_1d": rng.standard_normal(len(dates)),
                "future_return_5d": rng.standard_normal(len(dates)),
                "future_return_10d": rng.standard_normal(len(dates)),
            }
        )

        # Warm-up rows (feature NaNs)
        df.loc[0:2, ["sma_5", "rsi_14"]] = np.nan

        # End-of-series rows (target NaNs) — partial, never wipe a full column
        df.loc[17:19, ["future_return_1d", "future_return_5d", "future_return_10d"]] = np.nan

        file_path = features_dir / f"{symbol}.parquet"
        df.to_parquet(file_path, index=False)

    return features_dir