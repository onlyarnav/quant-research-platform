"""
Unit tests for rolling metrics and time series helpers.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.analytics.rolling_metrics import (
    drawdown_series,
    equity_series,
    exposure_series,
    monthly_returns_table,
    rolling_sharpe,
    rolling_volatility,
)


def _sample_series_df(n: int = 100) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    daily_ret = [0.01 if i % 2 == 0 else -0.005 for i in range(n)]
    pv = [100.0]
    for r in daily_ret[1:]:
        pv.append(pv[-1] * (1.0 + r))
    cummax = pd.Series(pv).cummax()
    dd = (pd.Series(pv) / cummax) - 1.0

    return pd.DataFrame(
        {
            "date": dates,
            "portfolio_value": pv,
            "daily_return": daily_ret,
            "drawdown": dd,
        }
    )


def test_rolling_window_less_than_window() -> None:
    df = _sample_series_df(n=5)
    # Default window is typically 63
    res_sharpe = rolling_sharpe(df, window=20)
    assert res_sharpe.empty
    assert list(res_sharpe.columns) == ["date", "rolling_sharpe"]

    res_vol = rolling_volatility(df, window=20)
    assert res_vol.empty
    assert list(res_vol.columns) == ["date", "rolling_volatility"]


def test_rolling_sharpe_and_volatility() -> None:
    df = _sample_series_df(n=50)
    res_sharpe = rolling_sharpe(df, window=10)
    assert not res_sharpe.empty
    assert len(res_sharpe) == 41  # 50 - 10 + 1

    res_vol = rolling_volatility(df, window=10)
    assert not res_vol.empty
    assert (res_vol["rolling_volatility"] > 0).all()


def test_drawdown_and_equity_series() -> None:
    df = _sample_series_df(n=20)
    dd_df = drawdown_series(df)
    assert "is_underwater" in dd_df.columns
    assert "drawdown" in dd_df.columns

    eq_df = equity_series(df)
    assert "cumulative_return" in eq_df.columns
    assert eq_df["cumulative_return"].iloc[0] == 0.0


def test_monthly_returns_table() -> None:
    dates = pd.date_range("2024-01-01", "2024-03-31", freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "daily_return": [0.001] * len(dates),
        }
    )
    table = monthly_returns_table(df)
    assert "Year" in table.columns
    assert "Jan" in table.columns
    assert "Total" in table.columns
    assert len(table) == 1
    assert table["Year"].iloc[0] == 2024


def test_exposure_series() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3, freq="D"),
            "long_exposure": [0.6, 0.7, 0.8],
            "short_exposure": [-0.1, -0.2, 0.0],
        }
    )
    exp = exposure_series(df)
    assert len(exp) == 3
    assert list(exp.columns) == [
        "date",
        "long_exposure",
        "short_exposure",
        "net_exposure",
        "gross_exposure",
    ]
    assert exp["net_exposure"].iloc[0] == pytest.approx(0.5)
    assert exp["gross_exposure"].iloc[0] == pytest.approx(0.7)
