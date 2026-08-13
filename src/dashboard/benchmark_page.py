"""Benchmark comparison visualization page for Streamlit dashboard."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import streamlit as st

from src.analytics.benchmark_comparison import BenchmarkComparison
from src.dashboard import data_loader
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _format_val(val: Any, is_pct: bool = False) -> str:
    """Format numeric comparison values for metric widgets."""
    if val is None or not isinstance(val, (int, float)) or math.isnan(val):
        return "N/A"
    return f"{val:.2%}" if is_pct else f"{val:.2f}"


def render(strategy_name: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> None:
    """
    Render the Benchmark Comparison dashboard page.

    Args:
        strategy_name: Strategy identifier string.
        start_date: Start date threshold for filtering.
        end_date: End date threshold for filtering.
    """
    st.header("Benchmark Comparison")

    benchmark_choice = st.selectbox("Benchmark", ["NIFTY50", "NIFTY500"])
    st.caption("Benchmark data must be present in the database under this strategy_name convention to enable comparison.")

    strat_df = data_loader.load_portfolio_history(strategy_name)
    bench_df = data_loader.load_portfolio_history(benchmark_choice)

    if not strat_df.empty and "date" in strat_df.columns and strat_df["date"].notna().any():
        strat_df["date"] = pd.to_datetime(strat_df["date"])
        strat_df = strat_df[(strat_df["date"] >= start_date) & (strat_df["date"] <= end_date)]

    if not bench_df.empty and "date" in bench_df.columns and bench_df["date"].notna().any():
        bench_df["date"] = pd.to_datetime(bench_df["date"])
        bench_df = bench_df[(bench_df["date"] >= start_date) & (bench_df["date"] <= end_date)]

    if strat_df.empty or bench_df.empty:
        st.info("No data available for the selected filters.")
        return

    comp_results: dict[str, Any] = {}
    try:
        comp_results = BenchmarkComparison().compare(strat_df, bench_df)
    except Exception as exc:
        st.warning(f"Could not compute benchmark comparison: {exc}")

    cols = st.columns(4)
    cols[0].metric("Alpha", _format_val(comp_results.get("alpha"), is_pct=True))
    cols[1].metric("Beta", _format_val(comp_results.get("beta")))
    cols[2].metric("Information Ratio", _format_val(comp_results.get("information_ratio")))
    cols[3].metric("Outperformance", _format_val(comp_results.get("outperformance"), is_pct=True))

    if "cumulative_return" in strat_df.columns and "cumulative_return" in bench_df.columns:
        merged = pd.merge(
            strat_df[["date", "cumulative_return"]].dropna(),
            bench_df[["date", "cumulative_return"]].dropna(),
            on="date",
            suffixes=(f" ({strategy_name})", f" ({benchmark_choice})"),
        )
        if not merged.empty:
            st.subheader("Cumulative Return Comparison")
            st.line_chart(merged.set_index("date"))
