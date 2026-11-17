"""
Trade log and trade-level analytics dashboard view.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard.charts import pnl_by_symbol_chart, trade_distribution_chart
from src.dashboard.client import DashboardClient
from src.dashboard.components import (
    download_csv_button,
    empty_state,
    error_box,
    safe_call,
)


def render(client: DashboardClient, run_id: str | None = None) -> None:
    """Render trade executions, PnL distribution, and symbol analytics."""
    st.header("Trade Execution Log & Analytics")

    if not run_id:
        empty_state("No backtest run selected. Please select a run from the sidebar.")
        return

    # 1. Distribution and PnL by Symbol Charts
    col1, col2 = st.columns(2)
    with col1:
        dist_df, _ = safe_call(client.get_trade_distribution, run_id)
        if dist_df is not None and not dist_df.empty:
            st.plotly_chart(trade_distribution_chart(dist_df), use_container_width=True)
        else:
            empty_state("Trade distribution histogram not available.")

    with col2:
        # Load trade page to compute symbol pnl
        trades_page, _ = safe_call(client.get_trades, run_id, limit=500)
        trades_list, _ = trades_page if trades_page else ([], 0)
        if trades_list:
            tdf = pd.DataFrame(trades_list)
            sym_grp = tdf.groupby("symbol")["pnl"].sum().reset_index()
            sym_grp.columns = ["symbol", "total_pnl"]
            st.plotly_chart(pnl_by_symbol_chart(sym_grp), use_container_width=True)
        else:
            empty_state("PnL by symbol chart not available.")

    st.divider()

    # 2. Paginated Trade Records Table with Filters
    st.subheader("Executed Trade Records")

    c1, c2 = st.columns(2)
    with c1:
        sym_filter = (
            st.text_input("Filter by Symbol (e.g. RELIANCE.NS)", value="").strip().upper()
            or None
        )
    with c2:
        page_size = st.selectbox("Page Size", options=[25, 50, 100], index=1)

    trades_res, err = safe_call(
        client.get_trades,
        run_id,
        symbol=sym_filter,
        limit=page_size or 50,
        offset=0,
    )

    if err:
        error_box(f"Failed to load trades: {err}")
        return

    items, total = trades_res if trades_res else ([], 0)
    st.caption(f"Showing {len(items)} of {total} total trades.")

    if items:
        df_display = pd.DataFrame(items)
        if "entry_date" in df_display.columns:
            df_display = df_display.sort_values("entry_date", ascending=False)
        st.dataframe(df_display, use_container_width=True)
        download_csv_button(df_display, f"{run_id}_trades.csv")
    else:
        empty_state("No trades matched the specified filter criteria.")
