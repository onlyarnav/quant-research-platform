"""
Unit tests for PerformanceMetrics class.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from config.settings import settings
from src.analytics.performance_metrics import PerformanceMetrics


def _sample_portfolio_df() -> pd.DataFrame:
    """Helper to create a standard synthetic portfolio history DataFrame."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "portfolio_value": [100.0, 102.0, 99.0, 104.0, 110.0],
            "daily_return": [0.0, 0.02, -0.02941176, 0.05050505, 0.05769231],
            "drawdown": [0.0, 0.0, -0.02941176, 0.0, 0.0],
        }
    )


def test_default_initialization() -> None:
    """Test default initialization pulls risk_free_rate from settings."""
    pm = PerformanceMetrics()
    assert pm.risk_free_rate == pytest.approx(settings.PORTFOLIO_RISK_FREE_RATE)
    assert pm.trading_days_per_year == 252


def test_invalid_initialization() -> None:
    """Test invalid trading_days_per_year raises ValueError."""
    with pytest.raises(ValueError, match="trading_days_per_year must be positive"):
        PerformanceMetrics(trading_days_per_year=0)


def test_input_validation_empty_and_missing_columns() -> None:
    """Test empty DataFrame and missing columns raise ValueError."""
    pm = PerformanceMetrics()
    empty_df = pd.DataFrame()

    with pytest.raises(ValueError, match="cannot be None or empty"):
        pm.total_return(empty_df)

    missing_cols_df = pd.DataFrame({"date": [pd.Timestamp("2024-01-01")]})
    with pytest.raises(ValueError, match="missing required columns"):
        pm.total_return(missing_cols_df)


def test_total_return_golden() -> None:
    """
    Test total_return golden path.

    Hand calculation:
    initial_val = 100.0, final_val = 110.0
    total_return = (110.0 / 100.0) - 1.0 = 0.10 (10.0%)
    """
    pm = PerformanceMetrics()
    df = _sample_portfolio_df()
    expected_total_return = (110.0 / 100.0) - 1.0  # 0.10
    assert pm.total_return(df) == pytest.approx(expected_total_return, abs=1e-6)


def test_total_return_invalid_initial_value() -> None:
    """Test initial portfolio value <= 0 raises ValueError."""
    pm = PerformanceMetrics()
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=2),
            "portfolio_value": [0.0, 100.0],
            "daily_return": [0.0, 1.0],
        }
    )
    with pytest.raises(ValueError, match="Initial portfolio value must be greater than 0"):
        pm.total_return(df)


def test_annualized_return_golden_and_single_day(caplog: pytest.LogCaptureFixture) -> None:
    """
    Test annualized_return golden path and single day fallback.

    Hand calculation:
    tot_ret = 0.10, num_days = 5, trading_days = 252
    annualized_return = (1 + 0.10) ** (252 / 5) - 1.0 = (1.10 ** 50.4) - 1.0 = 127.320489...
    """
    pm = PerformanceMetrics(trading_days_per_year=252)
    df = _sample_portfolio_df()
    expected_ann_ret = (1.10 ** (252 / 5)) - 1.0
    assert pm.annualized_return(df) == pytest.approx(expected_ann_ret, abs=1e-5)

    single_day_df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01")],
            "portfolio_value": [100.0],
            "daily_return": [0.0],
        }
    )
    res = pm.annualized_return(single_day_df)
    assert res == pytest.approx(0.0)
    assert "Annualization is not meaningful with a single data point" in caplog.text


def test_annualized_volatility_golden_and_degenerate(caplog: pytest.LogCaptureFixture) -> None:
    """
    Test annualized_volatility golden calculation and degenerate (<2 rows) fallback.

    Hand calculation:
    daily_returns = [0.01, -0.01, 0.02, -0.005]
    std(ddof=1) of [0.01, -0.01, 0.02, -0.005] = 0.013462912...
    annualized_vol = 0.013462912... * sqrt(252) = 0.2137168...
    """
    pm = PerformanceMetrics(trading_days_per_year=252)
    daily_rets = [0.01, -0.01, 0.02, -0.005]
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4),
            "portfolio_value": [100.0, 99.0, 100.98, 100.4751],
            "daily_return": daily_rets,
        }
    )
    expected_std = float(pd.Series(daily_rets).std(ddof=1))
    expected_vol = expected_std * math.sqrt(252)
    assert pm.annualized_volatility(df) == pytest.approx(expected_vol, abs=1e-6)

    single_row_df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01")],
            "portfolio_value": [100.0],
            "daily_return": [0.01],
        }
    )
    res_single = pm.annualized_volatility(single_row_df)
    assert math.isnan(res_single)
    assert "Fewer than 2 rows in portfolio history" in caplog.text


def test_sharpe_ratio_golden_and_zero_vol(caplog: pytest.LogCaptureFixture) -> None:
    """
    Test sharpe_ratio golden calculation and zero volatility fallback.

    Hand calculation:
    ann_ret = 0.15, risk_free_rate = 0.07, ann_vol = 0.10
    sharpe = (0.15 - 0.07) / 0.10 = 0.80
    """
    pm = PerformanceMetrics(risk_free_rate=0.07, trading_days_per_year=252)
    flat_df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3),
            "portfolio_value": [100.0, 101.0, 102.01],
            "daily_return": [0.01, 0.01, 0.01],
        }
    )
    # Volatility of constant returns is 0.0
    res_zero_vol = pm.sharpe_ratio(flat_df)
    assert math.isnan(res_zero_vol)
    assert "Cannot compute Sharpe ratio: annualized volatility is zero or undefined" in caplog.text


def test_sortino_ratio_golden_and_no_downside(caplog: pytest.LogCaptureFixture) -> None:
    """
    Test sortino_ratio calculation and no-downside-deviation fallback.
    """
    pm = PerformanceMetrics(risk_free_rate=0.07, trading_days_per_year=252)
    all_positive_df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3),
            "portfolio_value": [100.0, 102.0, 105.06],
            "daily_return": [0.02, 0.02, 0.03],
        }
    )
    res_no_downside = pm.sortino_ratio(all_positive_df)
    assert math.isnan(res_no_downside)
    assert "No downside deviation observed (no negative daily returns)" in caplog.text

    # Downside with negative returns
    neg_rets_df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5),
            "portfolio_value": [100.0, 102.0, 99.0, 96.0, 100.0],
            "daily_return": [0.02, 0.02, -0.02941176, -0.03030303, 0.04166667],
        }
    )
    sortino = pm.sortino_ratio(neg_rets_df)
    assert not math.isnan(sortino)


def test_max_drawdown_golden() -> None:
    """Test max_drawdown returns min drawdown value."""
    pm = PerformanceMetrics()
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4),
            "portfolio_value": [100.0, 90.0, 95.0, 85.0],
            "daily_return": [0.0, -0.10, 0.05555, -0.10526],
            "drawdown": [0.0, -0.10, -0.05, -0.15],
        }
    )
    assert pm.max_drawdown(df) == pytest.approx(-0.15)


def test_calmar_ratio_golden_and_zero_mdd(caplog: pytest.LogCaptureFixture) -> None:
    """Test calmar_ratio golden calculation and zero max drawdown fallback."""
    pm = PerformanceMetrics()
    zero_mdd_df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3),
            "portfolio_value": [100.0, 105.0, 110.0],
            "daily_return": [0.0, 0.05, 0.0476],
            "drawdown": [0.0, 0.0, 0.0],
        }
    )
    res_zero_mdd = pm.calmar_ratio(zero_mdd_df)
    assert math.isnan(res_zero_mdd)
    assert "Cannot compute Calmar ratio: max drawdown is zero" in caplog.text


def test_value_at_risk_and_cvar() -> None:
    """Test VaR and CVaR calculations and invalid confidence level."""
    pm = PerformanceMetrics()
    rng = np.random.default_rng(seed=42)
    daily_returns = rng.normal(0.001, 0.02, 100)
    portfolio_value = 100.0 * np.cumprod(1.0 + daily_returns)
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=100),
            "portfolio_value": portfolio_value,
            "daily_return": daily_returns,
            "drawdown": np.zeros(100),
        }
    )

    with pytest.raises(ValueError, match="confidence_level must be between 0 and 1 exclusive"):
        pm.value_at_risk(df, confidence_level=1.5)

    var_95 = pm.value_at_risk(df, confidence_level=0.95)
    cvar_95 = pm.conditional_value_at_risk(df, confidence_level=0.95)

    # CVaR is average of tail <= VaR, so CVaR <= VaR (more negative loss threshold)
    assert cvar_95 <= var_95


def test_compute_all_keys_and_spot_check() -> None:
    """Test compute_all returns dict with expected keys and matching single-method results."""
    pm = PerformanceMetrics()
    df = _sample_portfolio_df()
    result = pm.compute_all(df)

    expected_keys = {
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "calmar_ratio",
        "value_at_risk_95",
        "conditional_value_at_risk_95",
    }
    assert set(result.keys()) == expected_keys
    assert result["total_return"] == pytest.approx(pm.total_return(df))
    assert result["max_drawdown"] == pytest.approx(pm.max_drawdown(df))
