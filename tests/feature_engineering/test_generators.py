"""
Unit tests for all six feature and target generators.

Verifies:
- Schema conformance: symbol, date, and documented feature columns
- Row count preservation and index uniqueness
- NaN warm-up properties and absence of inf values
- Non-mutation of input DataFrames
- Golden mathematical calculations
- Strict no-lookahead invariance for all feature generators
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering.mean_reversion_features import (
    MeanReversionFeatureGenerator,
)
from src.feature_engineering.momentum_features import MomentumFeatureGenerator
from src.feature_engineering.target_generator import TargetGenerator
from src.feature_engineering.trend_features import TrendFeatureGenerator
from src.feature_engineering.volatility_features import VolatilityFeatureGenerator
from src.feature_engineering.volume_features import VolumeFeatureGenerator


def _make_ohlcv_frame(
    n_rows: int = 120, symbols: tuple[str, ...] = ("AAPL", "MSFT")
) -> pd.DataFrame:
    """Create a deterministic synthetic OHLCV DataFrame."""
    rng = np.random.default_rng(42)
    frames: list[pd.DataFrame] = []

    for sym in symbols:
        dates = pd.date_range("2023-01-01", periods=n_rows, freq="B")
        base_price = 100.0 + np.cumsum(rng.standard_normal(n_rows))
        base_price = np.maximum(base_price, 10.0)
        high = base_price + rng.uniform(0.5, 2.0, n_rows)
        low = base_price - rng.uniform(0.5, 2.0, n_rows)
        open_ = low + rng.uniform(0.0, 1.0, n_rows) * (high - low)
        close = low + rng.uniform(0.0, 1.0, n_rows) * (high - low)
        volume = rng.uniform(10_000, 50_000, n_rows)

        df = pd.DataFrame(
            {
                "symbol": sym,
                "date": dates,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "adj_close": close,
                "volume": volume,
            }
        )
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    return combined


# =====================================================================
# Common Contract Tests for Feature Generators
# =====================================================================


@pytest.mark.parametrize(
    "generator_cls",
    [
        TrendFeatureGenerator,
        MomentumFeatureGenerator,
        VolatilityFeatureGenerator,
        MeanReversionFeatureGenerator,
        VolumeFeatureGenerator,
        TargetGenerator,
    ],
)
def test_generator_output_schema_and_integrity(generator_cls) -> None:
    """Assert output contains required identifiers and feature columns, without infs."""
    gen = generator_cls()
    df = _make_ohlcv_frame(120)
    df_copy = df.copy(deep=True)

    result = gen.generate(df)

    # (a) output has symbol, date and every documented column
    assert "symbol" in result.columns
    assert "date" in result.columns
    expected_cols = getattr(
        gen, "FEATURE_COLUMNS", getattr(gen, "TARGET_COLUMNS", None)
    )
    for col in expected_cols:
        assert col in result.columns

    # (b) row count equals input
    assert len(result) == len(df)

    # (c) (symbol, date) is unique
    assert not result.duplicated(subset=["symbol", "date"]).any()

    # (e) no inf values
    numeric_cols = [c for c in result.columns if c not in ("symbol", "date")]
    for col in numeric_cols:
        assert not np.isinf(result[col].dropna()).any()

    # (f) input not mutated
    pd.testing.assert_frame_equal(df, df_copy)


# =====================================================================
# Golden Calculation Tests
# =====================================================================


def test_trend_generator_sma_5_golden() -> None:
    """Assert SMA_5 on closes 1..10 equals 3.0 at row index 4 (5th row)."""
    # Closes: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    # SMA_5 at index 4 = (1 + 2 + 3 + 4 + 5) / 5 = 3.0
    dates = pd.date_range("2023-01-01", periods=10, freq="B")
    df = pd.DataFrame(
        {
            "symbol": "AAPL",
            "date": dates,
            "close": np.arange(1.0, 11.0),
        }
    )
    gen = TrendFeatureGenerator()
    res = gen.generate(df)
    assert np.isnan(res.loc[0:3, "sma_5"]).all()
    assert res.loc[4, "sma_5"] == pytest.approx(3.0)


def test_momentum_generator_roc_5_golden() -> None:
    """Assert ROC_5 on a series doubling over five rows equals 1.0 (or 100 depending on scaling)."""
    # If price doubles from 10 to 20 over 5 periods, ROC = (20 - 10) / 10 = 1.0 (or 100.0)
    dates = pd.date_range("2023-01-01", periods=10, freq="B")
    closes = [10.0, 10.0, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 20.0, 20.0]
    df = pd.DataFrame({"symbol": "AAPL", "date": dates, "close": closes})
    gen = MomentumFeatureGenerator()
    res = gen.generate(df)
    # ROC_5 at index 5: compares index 5 (20.0) to index 0 (10.0) -> (20 - 10) / 10 = 1.0 (or 100%)
    roc_val = res.loc[5, "roc_5"]
    assert roc_val in (pytest.approx(1.0), pytest.approx(100.0))


def test_volume_generator_volume_ratio_golden() -> None:
    """Assert volume_ratio equals 1.0 for constant volume once window is defined."""
    dates = pd.date_range("2023-01-01", periods=30, freq="B")
    frames = []
    for sym in ("AAPL", "MSFT"):
        frames.append(
            pd.DataFrame(
                {
                    "symbol": sym,
                    "date": dates,
                    "close": np.linspace(100, 110, 30),
                    "volume": [5000.0] * 30,
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    gen = VolumeFeatureGenerator()
    res = gen.generate(df)
    # After warm-up (e.g. index 25), volume_ratio should be 1.0
    aapl_res = res[res["symbol"] == "AAPL"].reset_index(drop=True)
    assert aapl_res.loc[25, "volume_ratio"] == pytest.approx(1.0)


def test_target_generator_golden() -> None:
    """Assert future_return_1d[t] = close[t+1]/close[t] - 1, and trailing NaNs."""
    dates = pd.date_range("2023-01-01", periods=20, freq="B")
    closes = [100.0, 105.0] + [100.0] * 18
    df = pd.DataFrame({"symbol": "AAPL", "date": dates, "close": closes})
    gen = TargetGenerator()
    res = gen.generate(df)

    # future_return_1d at index 0 = 105 / 100 - 1 = 0.05
    assert res.loc[0, "future_return_1d"] == pytest.approx(0.05)
    # Last row of 1d return is NaN
    assert np.isnan(res.loc[19, "future_return_1d"])
    # 5d has 5 trailing NaNs
    assert res.loc[15:19, "future_return_5d"].isna().all()
    # 10d has 10 trailing NaNs
    assert res.loc[10:19, "future_return_10d"].isna().all()


# =====================================================================
# No-Lookahead Invariance Test (Mandatory)
# =====================================================================


@pytest.mark.parametrize(
    "generator_cls",
    [
        TrendFeatureGenerator,
        MomentumFeatureGenerator,
        VolatilityFeatureGenerator,
        MeanReversionFeatureGenerator,
        VolumeFeatureGenerator,
    ],
)
def test_no_lookahead_invariance(generator_cls) -> None:
    """
    Mandatory test: Assert that feature computation on a truncated series
    is identical to the full series up to row k (no future data leakage).
    """
    df_full = _make_ohlcv_frame(n_rows=100, symbols=("AAPL", "MSFT"))
    k = 60
    df_truncated = pd.concat(
        [df_full[df_full["symbol"] == sym].iloc[:k] for sym in ("AAPL", "MSFT")],
        ignore_index=True,
    )

    gen = generator_cls()
    res_full = gen.generate(df_full)
    res_trunc = gen.generate(df_truncated)

    for sym in ("AAPL", "MSFT"):
        sub_full = res_full[res_full["symbol"] == sym].iloc[:k].reset_index(drop=True)
        sub_trunc = res_trunc[res_trunc["symbol"] == sym].reset_index(drop=True)

        feature_cols = [c for c in sub_full.columns if c not in ("symbol", "date")]
        for col in feature_cols:
            val_full = sub_full[col].to_numpy()
            val_trunc = sub_trunc[col].to_numpy()
            np.testing.assert_allclose(
                val_full,
                val_trunc,
                equal_nan=True,
                err_msg=f"Lookahead detected in {generator_cls.__name__} for column {col}",
            )
