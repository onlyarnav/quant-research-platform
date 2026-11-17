"""
Portfolio weights and asset allocation dashboard view.
"""

from __future__ import annotations

import streamlit as st

from src.dashboard.charts import allocation_chart
from src.dashboard.client import DashboardClient
from src.dashboard.components import (
    download_csv_button,
    empty_state,
    error_box,
    safe_call,
)


def render(client: DashboardClient, run_id: str | None = None) -> None:
    """Render portfolio weights allocation page."""
    st.header("Asset Allocation & Exposure")

    if not run_id:
        empty_state("No backtest run selected. Please select a run from the sidebar.")
        return

    alloc_df, err = safe_call(client.get_allocation, run_id)
    if err:
        error_box(f"Failed to load allocation data: {err}")
        return

    if alloc_df is None or alloc_df.empty:
        empty_state("No open positions or allocation history found for this run.")
        return

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Holdings Distribution")
        st.plotly_chart(allocation_chart(alloc_df), use_container_width=True)

    with c2:
        st.subheader("Position Weights")
        st.dataframe(alloc_df, use_container_width=True)
        download_csv_button(alloc_df, f"{run_id}_allocation.csv")
