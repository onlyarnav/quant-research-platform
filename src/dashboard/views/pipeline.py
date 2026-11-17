"""
Automated pipeline execution and health monitoring dashboard view.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard.client import DashboardClient
from src.dashboard.components import empty_state, error_box, safe_call


def render(client: DashboardClient, run_id: str | None = None) -> None:
    """Render pipeline run histories and system operational health checks."""
    st.header("Pipeline Operations & Health")

    # 1. System Health Checks
    st.subheader("Operational Health Status")
    health_data, err = safe_call(client.health)
    if err:
        error_box(f"Failed to query system health: {err}")
    elif health_data:
        status_color = (
            "🟢 OK"
            if health_data.get("overall") == "ok"
            else (
                "🟡 DEGRADED" if health_data.get("overall") == "degraded" else "🔴 DOWN"
            )
        )
        st.markdown(f"**Overall System Status:** `{status_color}`")

        checks = health_data.get("checks", [])
        if checks:
            st.table(pd.DataFrame(checks))

    st.divider()

    # 2. Pipeline Run History
    st.subheader("Recent Daily Orchestration Runs")
    runs, err = safe_call(client.pipeline_runs, limit=20)
    if err:
        error_box(f"Failed to query pipeline runs: {err}")
    elif not runs:
        empty_state("No pipeline execution runs recorded yet.")
    else:
        df_runs = pd.DataFrame(runs)
        cols_to_show = [
            c
            for c in [
                "run_id",
                "started_at",
                "finished_at",
                "all_succeeded",
                "total_duration_seconds",
            ]
            if c in df_runs.columns
        ]
        st.dataframe(df_runs[cols_to_show], use_container_width=True)
