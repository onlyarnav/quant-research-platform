"""Portfolio performance visualization page for Streamlit dashboard."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import streamlit as st

from src.analytics.performance_metrics import PerformanceMetrics
from src.dashboard import data_loader
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _format_metric_val(val: Any, is_pct: bool = False) -> str:
    """Helper to format numeric metric values for st.metric display."""
    if val is None or not isinstance(val, (int, float)) or math.isnan(val):
        return "N/A"
    return f"{val:.2%}" if is_pct else f"{val:.2f}"


def render(strategy_name: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> None:
    """
    Render the Portfolio Performance dashboard page.

    Args:
        strategy_name: Selected strategy name.
        start_date: Start date threshold for filtering.
        end_date: End date threshold for filtering.
    """
    st.header("Portfolio Performance")

    df = data_loader.load_portfolio_history(strategy_name)

    if not df.empty and "date" in df.columns and df["date"].notna().any():
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

    if df.empty:
        st.info("No data available for the selected filters.")
        return

    metrics: dict[str, float] = {}
    try:
        metrics = PerformanceMetrics().compute_all(df)
    except Exception as exc:
        st.warning(f"Could not compute full metrics: {exc}")

    cols = st.columns(6)
    cols[0].metric("Total Return", _format_metric_val(metrics.get("total_return"), is_pct=True))
    cols[1].metric("Annualized Return", _format_metric_val(metrics.get("annualized_return"), is_pct=True))
    cols[2].metric("Sharpe Ratio", _format_metric_val(metrics.get("sharpe_ratio")))
    cols[3].metric("Max Drawdown", _format_metric_val(metrics.get("max_drawdown"), is_pct=True))
    cols[4].metric("Sortino Ratio", _format_metric_val(metrics.get("sortino_ratio")))
    cols[5].metric("Calmar Ratio", _format_metric_val(metrics.get("calmar_ratio")))

    if "cumulative_return" in df.columns and "date" in df.columns and df["cumulative_return"].notna().any():
        st.subheader("Cumulative Return")
        st.line_chart(df.set_index("date")["cumulative_return"])

    if "drawdown" in df.columns and "date" in df.columns and df["drawdown"].notna().any():
        st.subheader("Drawdown")
        st.line_chart(df.set_index("date")["drawdown"])

    with st.expander("Raw Data"):
        st.dataframe(df, use_container_width=True)
