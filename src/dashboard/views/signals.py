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
        try:
            d_obj = pd.to_datetime(as_of)
            if d_obj.tzinfo is not None:
                as_of_str = d_obj.tz_convert("Asia/Kolkata").strftime("%d %b %Y (Market Date)")
            else:
                as_of_str = d_obj.strftime("%d %b %Y")
        except Exception:
            as_of_str = str(as_of)
        st.subheader(f"Signal Counts (As of {as_of_str})")
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

        display_df = sig_df.copy()
        try:
            d_series = pd.to_datetime(display_df["date"])
            if d_series.dt.tz is not None:
                display_df["date"] = d_series.dt.tz_convert("Asia/Kolkata").dt.strftime("%Y-%m-%d")
            else:
                display_df["date"] = d_series.dt.strftime("%Y-%m-%d")
        except Exception:
            pass

        if "signal" in display_df.columns:
            display_df["action"] = display_df["signal"].map(
                {1: "🟢 BUY", 0: "🟡 HOLD", -1: "🔴 SELL"}
            ).fillna("🟡 HOLD")
        if "predicted_return" in display_df.columns:
            display_df["rank"] = display_df["predicted_return"].rank(
                ascending=False, method="dense"
            ).astype(int)
            display_df["exp_return_%"] = (display_df["predicted_return"] * 100).map("{:+.3f}%".format)

        cols_order = [
            c for c in ["rank", "symbol", "action", "exp_return_%", "confidence", "date"]
            if c in display_df.columns
        ]
        other_cols = [
            c for c in display_df.columns
            if c not in cols_order and c not in ["signal", "predicted_return"]
        ]
        st.dataframe(display_df[cols_order + other_cols], use_container_width=True)
