"""
Performance analytics dashboard view.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard.charts import (
    monthly_returns_heatmap,
    rolling_sharpe_chart,
    rolling_vol_chart,
)
from src.dashboard.client import DashboardClient
from src.dashboard.components import (
    download_csv_button,
    empty_state,
    error_box,
    safe_call,
)


def render(client: DashboardClient, run_id: str | None = None) -> None:
    """Render performance time-series and rolling metrics page."""
    st.header("Performance & Risk Dynamics")

    if not run_id:
        empty_state("No backtest run selected. Please select a run from the sidebar.")
        return

    # Window slider for rolling metrics
    window = st.slider(
        "Rolling Window (Trading Days)", min_value=21, max_value=252, value=63, step=21
    )

    roll_df, err = safe_call(client.get_rolling, run_id, window=window)
    if err:
        error_box(f"Failed to compute rolling metrics: {err}")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(rolling_sharpe_chart(roll_df), use_container_width=True)
        with col2:
            st.plotly_chart(rolling_vol_chart(roll_df), use_container_width=True)

    st.divider()

    # Monthly Returns Heatmap
    st.subheader("Monthly Returns Distribution")
    monthly_df, err = safe_call(client.get_monthly_returns, run_id)
    if err or monthly_df is None or monthly_df.empty:
        empty_state("Monthly returns table not available.")
    else:
        st.plotly_chart(monthly_returns_heatmap(monthly_df), use_container_width=True)
        download_csv_button(monthly_df, f"{run_id}_monthly_returns.csv")

    st.divider()

    # Comprehensive Metrics Table
    st.subheader("Performance Metric Breakdown")
    metrics_res, _ = safe_call(client.get_metrics, run_id)
    if metrics_res and "metrics" in metrics_res:
        m_items = [{"Metric": k, "Value": v} for k, v in metrics_res["metrics"].items()]
        m_table = pd.DataFrame(m_items)
        st.dataframe(m_table, use_container_width=True)
        download_csv_button(m_table, f"{run_id}_metrics.csv")
