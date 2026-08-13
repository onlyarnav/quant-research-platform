"""Universe allocation breakdown visualization page for Streamlit dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard import data_loader
from src.utils.logger import get_logger

logger = get_logger(__name__)


def render(start_date: pd.Timestamp, end_date: pd.Timestamp) -> None:
    """
    Render the Allocation Breakdown dashboard page.

    Args:
        start_date: Start date threshold for signals.
        end_date: End date threshold for signals.
    """
    st.header("Allocation Breakdown")

    symbols = data_loader.load_available_symbols()

    recent_signals: list[dict] = []
    for symbol in symbols:
        signals_df = data_loader.load_signals(symbol, start_date, end_date)
        if not signals_df.empty:
            if "date" in signals_df.columns:
                signals_df = signals_df.sort_values(by="date", ascending=True)
            last_signal = signals_df.iloc[-1]
            if last_signal.get("signal") == 1:
                recent_signals.append(
                    {
                        "symbol": last_signal.get("symbol", symbol),
                        "predicted_return": last_signal.get("predicted_return", 0.0),
                        "model_version": last_signal.get("model_version", "N/A"),
                        "date": last_signal.get("date"),
                    }
                )

    if not recent_signals:
        st.info("No active BUY signals in the selected date range.")
        return

    alloc_df = pd.DataFrame(recent_signals)
    alloc_df.sort_values(by="predicted_return", ascending=False, inplace=True)

    st.subheader("Allocation Weighting")
    fig = px.pie(
        alloc_df,
        names="symbol",
        values="predicted_return",
        title="Relative Allocation Weight by Predicted Return",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Active BUY Signals")
    st.dataframe(alloc_df, use_container_width=True)
