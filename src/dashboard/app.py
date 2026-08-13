"""Main Streamlit application entrypoint for Quant Research Platform."""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from src.dashboard import (
    allocation_page,
    benchmark_page,
    data_loader,
    performance_page,
    trades_page,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

st.set_page_config(page_title="Quant Research Platform", layout="wide")


def main() -> None:
    """Streamlit sidebar navigation and page dispatch router."""
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        [
            "Portfolio Performance",
            "Trade Log",
            "Allocation Breakdown",
            "Benchmark Comparison",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Filters")

    strategy_name = st.sidebar.text_input("Strategy Name", value="NIFTY50_MOMENTUM")

    available_symbols = data_loader.load_available_symbols()
    if available_symbols:
        symbol = st.sidebar.selectbox("Symbol", available_symbols, index=0)
    else:
        symbol = st.sidebar.text_input("Symbol", value="RELIANCE.NS")

    default_start = date(2024, 1, 1)
    default_end = date(2024, 12, 31)

    date_selection = st.sidebar.date_input(
        "Date Range",
        value=(default_start, default_end),
    )

    if isinstance(date_selection, (tuple, list)) and len(date_selection) == 2:
        start_date = pd.Timestamp(date_selection[0])
        end_date = pd.Timestamp(date_selection[1])
    elif isinstance(date_selection, (tuple, list)) and len(date_selection) == 1:
        start_date = pd.Timestamp(date_selection[0])
        end_date = pd.Timestamp(date_selection[0])
    else:
        start_date = pd.Timestamp(default_start)
        end_date = pd.Timestamp(default_end)

    if page == "Portfolio Performance":
        performance_page.render(strategy_name, start_date, end_date)
    elif page == "Trade Log":
        trades_page.render(symbol, start_date, end_date)
    elif page == "Allocation Breakdown":
        allocation_page.render(start_date, end_date)
    elif page == "Benchmark Comparison":
        benchmark_page.render(strategy_name, start_date, end_date)


if __name__ == "__main__":
    main()
