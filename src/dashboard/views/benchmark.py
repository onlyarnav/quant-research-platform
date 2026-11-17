"""
Benchmark comparison and active risk dashboard view.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard.client import DashboardClient
from src.dashboard.components import empty_state, error_box, kpi_row, safe_call


def render(client: DashboardClient, run_id: str | None = None) -> None:
    """Render benchmark relative performance page."""
    st.header("Benchmark Comparison & Active Risk")

    if not run_id:
        empty_state("No backtest run selected. Please select a run from the sidebar.")
        return

    bench_res, err = safe_call(client.get_benchmark, run_id)
    if err:
        error_box(f"Failed to load benchmark comparison: {err}")
        return

    if not bench_res:
        empty_state("Benchmark data was not computed for this backtest run.")
        return

    # Render Active KPIs
    specs = [
        ("alpha", "Alpha (Excess Ann.)", "pct"),
        ("beta", "Beta", "ratio"),
        ("information_ratio", "Info Ratio", "ratio"),
        ("tracking_error", "Tracking Error", "pct"),
        ("outperformance", "Outperformance", "pct"),
        ("correlation", "Correlation", "ratio"),
        ("up_capture", "Up Capture", "ratio"),
        ("down_capture", "Down Capture", "ratio"),
    ]
    kpi_row(bench_res, specs)

    st.divider()

    # Comparison metrics table
    st.subheader("Strategy vs Benchmark Metrics Comparison")
    strat_m = bench_res.get("strategy_metrics", {})
    bench_m = bench_res.get("benchmark_metrics", {})

    all_keys = sorted(set(strat_m.keys()).union(bench_m.keys()))
    rows = []
    for k in all_keys:
        rows.append(
            {
                "Metric": k,
                "Strategy": strat_m.get(k),
                "Benchmark": bench_m.get(k),
            }
        )

    comp_df = pd.DataFrame(rows)
    st.dataframe(comp_df, use_container_width=True)
