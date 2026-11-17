"""
Trading signals monitor dashboard view.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard.client import DashboardClient
from src.dashboard.components import empty_state, safe_call


def render(client: DashboardClient, run_id: str | None = None) -> None:
    """Render latest signals, signal counts, and confidence ranking."""
    st.header("Signal Generation Monitor")

    col1, col2 = st.columns(2)
    with col1:
        timeframe = st.selectbox("Timeframe", options=["daily", "hourly"], index=0)
    with col2:
        asset_class = st.selectbox(
            "Asset Class", options=["equity", "crypto", "forex"], index=0
        )

    # 1. Signal counts
    counts_res, err = safe_call(
        client.signal_counts, asset_class=asset_class, timeframe=timeframe
    )
    if not err and counts_res and counts_res[0]:
        as_of, counts = counts_res
        st.subheader(f"Signal Counts (As of {as_of})")
        c_df = pd.DataFrame(
            [{"Signal": k.upper(), "Count": v} for k, v in counts.items()]
        )
        fig = px.bar(
            c_df,
            x="Signal",
            y="Count",
            color="Signal",
            color_discrete_map={
                "BUY": "#2ca02c",
                "NEUTRAL": "#7f7f7f",
                "SELL": "#d62728",
            },
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        empty_state("No signal counts available for this asset class.")

    st.divider()

    # 2. Latest signals table
    st.subheader("Latest Generated Signals")
    sig_df, err = safe_call(
        client.latest_signals, asset_class=asset_class, timeframe=timeframe
    )
    if err or sig_df is None or sig_df.empty:
        empty_state("No signals found in the signals storage path.")
    else:
        sig_filter = st.radio(
            "Filter Signal Value",
            options=["All", "Buy Only (+1)", "Sell Only (-1)"],
            horizontal=True,
        )
        if sig_filter == "Buy Only (+1)":
            sig_df = sig_df[sig_df["signal"] > 0]
        elif sig_filter == "Sell Only (-1)":
            sig_df = sig_df[sig_df["signal"] < 0]

        st.dataframe(sig_df, use_container_width=True)
