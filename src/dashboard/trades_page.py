"""Trade log visualization page for Streamlit dashboard."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import streamlit as st

from src.analytics.trade_analytics import TradeAnalytics
from src.dashboard import data_loader
from src.utils.logger import get_logger

logger = get_logger(__name__)


def render(symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> None:
    """
    Render the Trade Log dashboard page.

    Args:
        symbol: Equity ticker symbol.
        start_date: Start date threshold.
        end_date: End date threshold.
    """
    st.header("Trade Log")

    trades_df = data_loader.load_trades(symbol, start_date, end_date)

    if trades_df.empty:
        st.info("No data available for the selected filters.")
        return

    stats: dict[str, Any] = {}
    try:
        stats = TradeAnalytics().compute_all(trades_df)
    except Exception as exc:
        st.warning(f"Could not compute trade analytics: {exc}")

    cols = st.columns(5)

    tot_trades = stats.get("total_trades", len(trades_df))
    cols[0].metric("Total Trades", f"{tot_trades}")

    win_rate = stats.get("win_rate")
    cols[1].metric("Win Rate", f"{win_rate:.2%}" if win_rate is not None else "N/A")

    pf = stats.get("profit_factor")
    if pf is not None and isinstance(pf, float) and math.isinf(pf):
        pf_str = "∞"
    elif pf is not None and isinstance(pf, (int, float)):
        pf_str = f"{pf:,.2f}"
    else:
        pf_str = "N/A"
    cols[2].metric("Profit Factor", pf_str)

    exp = stats.get("expectancy")
    cols[3].metric("Expectancy", f"{exp:,.2f}" if exp is not None else "N/A")

    pnl = stats.get("total_pnl")
    cols[4].metric("Total PnL", f"₹{pnl:,.2f}" if pnl is not None else "N/A")

    if "pnl" in trades_df.columns:
        st.subheader("PnL per Trade")
        x_col = "entry_date" if "entry_date" in trades_df.columns else "trade_id"
        st.bar_chart(trades_df, x=x_col, y="pnl")

    sort_col = "entry_date" if "entry_date" in trades_df.columns else trades_df.columns[0]
    sorted_df = trades_df.sort_values(by=sort_col, ascending=False)
    st.dataframe(sorted_df, use_container_width=True)
