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
    st.subheader("Manual Pipeline Execution")
    col_desc, col_btn = st.columns([3, 1])
    with col_desc:
        st.caption(
            "Trigger the automated daily pipeline: "
            "Ingestion ➔ Features ➔ Inference ➔ Signals ➔ Backtest ➔ Analytics."
        )
    with col_btn:
        run_pipeline = st.button(
            "🚀 Run Full Pipeline", type="primary", use_container_width=True
        )

    if run_pipeline:
        from datetime import UTC, datetime
        from src.backtesting.backtest_config import BacktestConfig
        from src.infra.pipeline_orchestrator import PipelineOrchestrator
        from src.infra.pipeline_steps import PipelineContext

        p_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_manual"
        ctx = PipelineContext(
            run_id=p_id,
            asset_class="equity",
            timeframe="daily",
            dry_run=False,
            backtest_config=BacktestConfig(strategy_name="production_daily"),
            extras={
                "signal_generator": "topn",
                "top_n": 5,
                "prediction_scope": "all",
                "backtest_start": "2026-01-01",
            },
        )
        with st.spinner("Executing end-to-end pipeline workflow..."):
            try:
                orchestrator = PipelineOrchestrator()
                report = orchestrator.run(ctx)
                if report.all_succeeded:
                    st.success(
                        f"Pipeline `{p_id}` completed successfully in {report.total_duration_seconds:.1f}s!"
                    )
                else:
                    st.warning(
                        f"Pipeline `{p_id}` completed with partial results or errors."
                    )
                step_records = [
                    {
                        "Step": s.name,
                        "Status": s.status.value.upper(),
                        "Duration (s)": f"{s.duration_seconds:.2f}",
                        "Details": s.message or "",
                    }
                    for s in report.steps
                ]
                st.table(pd.DataFrame(step_records))
            except Exception as exc:
                error_box(f"Pipeline execution encountered error: {exc}")

    st.divider()

    # 3. Pipeline Run History
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
