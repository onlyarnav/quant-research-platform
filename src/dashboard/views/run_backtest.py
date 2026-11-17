"""
Interactive backtest submission and job tracking view.
"""

from __future__ import annotations

import time

import streamlit as st

from src.dashboard.client import DashboardClient
from src.dashboard.components import error_box, safe_call


def render(client: DashboardClient, run_id: str | None = None) -> None:
    """Render interactive backtest launcher form and job progress poller."""
    st.header("Launch Backtest Simulation")

    with st.form("backtest_submission_form"):
        st.subheader("Strategy Parameters")
        c1, c2 = st.columns(2)
        with c1:
            strategy_name = st.text_input("Strategy Name", value="momentum_daily")
            capital = st.number_input(
                "Initial Capital (₹)", min_value=1000.0, value=100_000.0, step=10_000.0
            )
            alloc_method = st.selectbox(
                "Allocation Method",
                options=[
                    "equal_weight",
                    "signal_weighted",
                    "volatility_weighted",
                    "risk_parity",
                    "fixed_notional",
                ],
                index=0,
            )
            rebalance_freq = st.selectbox(
                "Rebalance Frequency",
                options=["daily", "weekly", "monthly", "on_signal"],
                index=0,
            )

        with c2:
            timeframe = st.selectbox("Timeframe", options=["daily", "hourly"], index=0)
            asset_class = st.selectbox(
                "Asset Class", options=["equity", "crypto", "forex"], index=0
            )
            execution_lag = st.number_input(
                "Execution Lag (bars)", min_value=0, max_value=5, value=1
            )
            allow_short = st.checkbox("Allow Shorting", value=False)

        from src.data_pipeline.universe import get_nse_equity_universe

        try:
            u_syms = [a.symbol for a in get_nse_equity_universe()]
        except Exception:
            u_syms = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
        sel_symbols = st.multiselect(
            "Investment Universe (Select Stocks)", options=u_syms, default=u_syms
        )
        d1, d2 = st.columns(2)
        with d1:
            start_date_str = st.text_input(
                "Start Date (e.g. 2026-01-01, blank = full history)", value="2026-01-01"
            )
        with d2:
            end_date_str = st.text_input(
                "End Date (e.g. 2026-11-17, blank = latest)", value=""
            )

        st.subheader("Risk & Friction Constraints")
        r1, r2, r3 = st.columns(3)
        with r1:
            tx_cost = (
                st.number_input(
                    "Transaction Cost (bps)", min_value=0.0, value=10.0, step=1.0
                )
                / 10_000.0
            )
            slippage = (
                st.number_input("Slippage (bps)", min_value=0.0, value=5.0, step=1.0)
                / 10_000.0
            )
        with r2:
            max_pos = st.number_input(
                "Max Position Size",
                min_value=0.01,
                max_value=1.0,
                value=0.20,
                step=0.05,
            )
            cash_res = st.number_input(
                "Cash Reserve", min_value=0.0, max_value=0.9, value=0.05, step=0.05
            )
        with r3:
            stop_loss = st.number_input(
                "Stop Loss % (0 for none)",
                min_value=0.0,
                max_value=0.5,
                value=0.05,
                step=0.01,
            )
            take_profit = st.number_input(
                "Take Profit % (0 for none)",
                min_value=0.0,
                max_value=1.0,
                value=0.10,
                step=0.02,
            )

        submitted = st.form_submit_button("🚀 Run Backtest", use_container_width=True)

    if submitted:
        req = {
            "strategy_name": strategy_name,
            "initial_capital": capital,
            "allocation_method": alloc_method,
            "rebalance_frequency": rebalance_freq,
            "timeframe": timeframe,
            "asset_class": asset_class,
            "execution_lag": execution_lag,
            "allow_short": allow_short,
            "transaction_cost": tx_cost,
            "slippage": slippage,
            "max_position_size": max_pos,
            "cash_reserve": cash_res,
            "stop_loss_pct": stop_loss if stop_loss > 0 else None,
            "take_profit_pct": take_profit if take_profit > 0 else None,
            "symbols": sel_symbols or None,
            "start": start_date_str.strip() or None,
            "end": end_date_str.strip() or None,
        }

        job_id, err = safe_call(client.submit_backtest, req)
        if err:
            error_box(f"Failed to submit backtest: {err}")
        else:
            st.success(f"Backtest job submitted successfully! Job ID: `{job_id}`")
            st.session_state["active_job_id"] = job_id
            st.rerun()

    # Active Job Polling
    active_job = st.session_state.get("active_job_id")
    if active_job:
        st.divider()
        st.subheader("Job Status Tracker")
        status_data, err = safe_call(client.job_status, active_job)
        if err:
            error_box(f"Error fetching job status: {err}")
        elif status_data:
            current_status = status_data.get("status", "unknown").lower()
            c_info, c_action = st.columns([3, 1])
            with c_info:
                st.info(
                    f"Job `{active_job}` Status: **{current_status.upper()}**"
                )
            with c_action:
                if st.button("Dismiss", key="dismiss_job", use_container_width=True):
                    st.session_state.pop("active_job_id", None)
                    st.rerun()

            if current_status == "succeeded":
                st.success(
                    f"Backtest completed successfully! Generated Run ID: `{status_data.get('run_id')}`"
                )
            elif current_status == "failed":
                error_box(
                    f"Backtest execution failed: {status_data.get('error')}\n\n"
                    "Tip: Verify that signal parquets exist in `data/signals/` or run the pipeline first."
                )
            elif current_status in {"queued", "running"}:
                with st.spinner("Executing simulation in background..."):
                    time.sleep(1.5)
                    st.rerun()
