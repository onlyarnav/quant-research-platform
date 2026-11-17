"""
Streamlit quantitative research and production dashboard application.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Fix Streamlit path shadowing: Streamlit adds script directory (src/dashboard)
# to sys.path, which causes 'config.py' to shadow the top-level 'config' package.
_script_dir = str(Path(__file__).resolve().parent)
while _script_dir in sys.path:
    sys.path.remove(_script_dir)

_project_root = str(Path(__file__).resolve().parent.parent.parent)
while _project_root in sys.path:
    sys.path.remove(_project_root)
sys.path.insert(0, _project_root)

# If 'config' is cached in sys.modules as a module (without __path__), evict it
if "config" in sys.modules and not hasattr(sys.modules["config"], "__path__"):
    del sys.modules["config"]

import streamlit as st  # noqa: E402

from src.dashboard.client import DashboardClient, get_client  # noqa: E402
from src.dashboard.views import (  # noqa: E402
    allocation,
    benchmark,
    experiments,
    overview,
    performance,
    pipeline,
    run_backtest,
    signals,
    trades,
)

PAGE_DISPATCH = {
    "Overview": overview.render,
    "Performance": performance.render,
    "Trades": trades.render,
    "Allocation": allocation.render,
    "Benchmark": benchmark.render,
    "Signals": signals.render,
    "Experiments": experiments.render,
    "Pipeline": pipeline.render,
    "Run Backtest": run_backtest.render,
}


def main(client: DashboardClient | None = None) -> None:
    """Main Streamlit execution and routing entrypoint."""
    st.set_page_config(
        page_title="Quant Research Platform",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    cli = client or get_client()

    # Sidebar Header & Navigation
    st.sidebar.title("Quant Research")
    page_name = st.sidebar.radio("Navigation", list(PAGE_DISPATCH.keys()), index=0)

    st.sidebar.divider()

    # Backtest Run Selector
    run_id = None
    try:
        runs = cli.list_runs(limit=50)
    except Exception:
        runs = []

    if runs:
        options = [r["run_id"] for r in runs]
        labels = {
            r["run_id"]: (
                f"{r.get('strategy_name', 'strategy')} "
                f"({str(r.get('created_at', ''))[:10]})"
            )
            for r in runs
        }
        selected = st.sidebar.selectbox(
            "Selected Backtest Run",
            options=options,
            format_func=lambda x: labels.get(x, x),
            index=0,
        )
        run_id = selected
    else:
        st.sidebar.info("No saved runs found.")

    st.sidebar.divider()

    # Health Badge in Sidebar
    try:
        health_info = cli.health()
        status = health_info.get("overall", "ok").upper()
        badge = "🟢" if status == "OK" else ("🟡" if status == "DEGRADED" else "🔴")
        st.sidebar.caption(f"System Health: {badge} {status}")
    except Exception:
        st.sidebar.caption("System Health: ⚪ UNKNOWN")

    # Render Active Page
    render_fn = PAGE_DISPATCH.get(page_name, overview.render)
    render_fn(cli, run_id=run_id)


if __name__ == "__main__":
    main()
