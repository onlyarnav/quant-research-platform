"""
Unit tests for Plotly chart builders.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from src.dashboard.charts import (
    allocation_chart,
    drawdown_chart,
    equity_curve_chart,
    monthly_returns_heatmap,
    pnl_by_symbol_chart,
    rolling_sharpe_chart,
    rolling_vol_chart,
    trade_distribution_chart,
)


def test_charts_with_empty_data() -> None:
    empty_df = pd.DataFrame()
    assert isinstance(equity_curve_chart(empty_df), go.Figure)
    assert isinstance(drawdown_chart(empty_df), go.Figure)
    assert isinstance(rolling_sharpe_chart(empty_df), go.Figure)
    assert isinstance(rolling_vol_chart(empty_df), go.Figure)
    assert isinstance(monthly_returns_heatmap(empty_df), go.Figure)
    assert isinstance(trade_distribution_chart(empty_df), go.Figure)
    assert isinstance(pnl_by_symbol_chart(empty_df), go.Figure)
    assert isinstance(allocation_chart(empty_df), go.Figure)


def test_equity_and_drawdown_charts() -> None:
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    eq_df = pd.DataFrame(
        {"date": dates, "portfolio_value": [100.0, 102.0, 99.0, 104.0, 105.0]}
    )
    bench_df = pd.DataFrame(
        {"date": dates, "portfolio_value": [100.0, 101.0, 100.0, 102.0, 103.0]}
    )
    dd_df = pd.DataFrame({"date": dates, "drawdown": [0.0, 0.0, -0.029, 0.0, 0.0]})

    fig_eq = equity_curve_chart(eq_df, bench_df)
    assert len(fig_eq.data) == 2

    fig_dd = drawdown_chart(dd_df)
    assert len(fig_dd.data) == 1


def test_allocation_and_pnl_charts() -> None:
    alloc_df = pd.DataFrame({"symbol": ["AAPL", "MSFT"], "weight": [0.6, 0.4]})
    fig_alloc = allocation_chart(alloc_df)
    assert len(fig_alloc.data) == 1

    pnl_df = pd.DataFrame({"symbol": ["AAPL", "MSFT"], "total_pnl": [500.0, -200.0]})
    fig_pnl = pnl_by_symbol_chart(pnl_df)
    assert len(fig_pnl.data) == 1
