"""
Overview dashboard view.
"""

from __future__ import annotations

import streamlit as st

from src.dashboard.charts import drawdown_chart, equity_curve_chart
from src.dashboard.client import DashboardClient
from src.dashboard.components import empty_state, error_box, kpi_row, safe_call


def render(client: DashboardClient, run_id: str | None = None) -> None:
    """Render strategy overview dashboard page."""
    st.header("Strategy Overview")

    if not run_id:
        empty_state(
            "No backtest run selected. Please select a run from the sidebar or launch a new backtest."
        )
        return

    # 1. Fetch metrics & run info
    metrics_res, err = safe_call(client.get_metrics, run_id)
    if err:
        error_box(f"Failed to load metrics for run {run_id}: {err}")
        return

    run_info, _ = safe_call(client.get_run, run_id)
    all_metrics = {
        **metrics_res.get("metrics", {}),
        **metrics_res.get("trade_stats", {}),
    }

    # 2. Render KPI Row
    specs = [
        ("total_return", "Total Return", "pct"),
        ("annualized_return", "Ann. Return", "pct"),
        ("sharpe_ratio", "Sharpe Ratio", "ratio"),
        ("sortino_ratio", "Sortino Ratio", "ratio"),
        ("max_drawdown", "Max Drawdown", "pct"),
        ("calmar_ratio", "Calmar Ratio", "ratio"),
        ("win_rate", "Win Rate", "pct"),
        ("profit_factor", "Profit Factor", "ratio"),
        ("total_trades", "Trades", None),
    ]
    kpi_row(all_metrics, specs)

    st.divider()

    # 3. Equity Curve & Benchmark
    eq_df, _ = safe_call(client.get_equity, run_id)
    bench_df = None  # Benchmark line embedded if available
    st.plotly_chart(equity_curve_chart(eq_df, bench_df), use_container_width=True)

    # 4. Underwater Drawdown
    dd_df, _ = safe_call(client.get_drawdown, run_id)
    st.plotly_chart(drawdown_chart(dd_df), use_container_width=True)

    # 5. Run Configuration Expander
    if run_info and "config" in run_info:
        with st.expander("Run Configuration Details"):
            st.json(run_info["config"])
