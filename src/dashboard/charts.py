"""
Plotly chart construction functions for Dashboard visualizations.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _empty_figure(title: str = "No Data Available") -> go.Figure:
    """Return an empty placeholder figure."""
    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": "No data to display",
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 14},
            }
        ],
        template="plotly_white",
    )
    return fig


def equity_curve_chart(
    equity_df: pd.DataFrame, benchmark_df: pd.DataFrame | None = None
) -> go.Figure:
    """Construct equity growth curve with optional benchmark overlay."""
    if equity_df is None or equity_df.empty or "date" not in equity_df.columns:
        return _empty_figure("Equity Curve")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pd.to_datetime(equity_df["date"]),
            y=equity_df["portfolio_value"],
            mode="lines",
            name="Strategy",
            line={"color": "#1f77b4", "width": 2},
        )
    )

    if (
        benchmark_df is not None
        and not benchmark_df.empty
        and "portfolio_value" in benchmark_df.columns
    ):
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(benchmark_df["date"]),
                y=benchmark_df["portfolio_value"],
                mode="lines",
                name="Benchmark",
                line={"color": "#7f7f7f", "width": 1.5, "dash": "dash"},
            )
        )

    fig.update_layout(
        title="Portfolio Equity Curve",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        template="plotly_white",
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    )
    return fig


def drawdown_chart(drawdown_df: pd.DataFrame) -> go.Figure:
    """Construct underwater drawdown chart."""
    if (
        drawdown_df is None
        or drawdown_df.empty
        or "drawdown" not in drawdown_df.columns
    ):
        return _empty_figure("Drawdown")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pd.to_datetime(drawdown_df["date"]),
            y=drawdown_df["drawdown"] * 100.0,
            fill="tozeroy",
            mode="lines",
            name="Drawdown",
            line={"color": "#d62728", "width": 1.5},
            fillcolor="rgba(214, 39, 40, 0.2)",
        )
    )
    fig.update_layout(
        title="Underwater Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def rolling_sharpe_chart(rolling_df: pd.DataFrame) -> go.Figure:
    """Construct rolling Sharpe ratio curve."""
    if (
        rolling_df is None
        or rolling_df.empty
        or "rolling_sharpe" not in rolling_df.columns
    ):
        return _empty_figure("Rolling Sharpe Ratio")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pd.to_datetime(rolling_df["date"]),
            y=rolling_df["rolling_sharpe"],
            mode="lines",
            name="Rolling Sharpe",
            line={"color": "#2ca02c", "width": 1.5},
        )
    )
    fig.add_hline(y=0.0, line_dash="dash", line_color="black", opacity=0.5)
    fig.update_layout(
        title="Rolling Sharpe Ratio",
        xaxis_title="Date",
        yaxis_title="Sharpe Ratio",
        template="plotly_white",
    )
    return fig


def rolling_vol_chart(rolling_df: pd.DataFrame) -> go.Figure:
    """Construct rolling annualized volatility curve."""
    if (
        rolling_df is None
        or rolling_df.empty
        or "rolling_volatility" not in rolling_df.columns
    ):
        return _empty_figure("Rolling Volatility")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pd.to_datetime(rolling_df["date"]),
            y=rolling_df["rolling_volatility"] * 100.0,
            mode="lines",
            name="Rolling Volatility",
            line={"color": "#ff7f0e", "width": 1.5},
        )
    )
    fig.update_layout(
        title="Rolling Annualized Volatility",
        xaxis_title="Date",
        yaxis_title="Volatility (%)",
        template="plotly_white",
    )
    return fig


def monthly_returns_heatmap(monthly_df: pd.DataFrame) -> go.Figure:
    """Construct monthly returns heatmap."""
    month_cols = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    if monthly_df is None or monthly_df.empty or "Year" not in monthly_df.columns:
        return _empty_figure("Monthly Returns Heatmap")

    years = monthly_df["Year"].astype(str).tolist()
    available_months = [m for m in month_cols if m in monthly_df.columns]
    z_vals = (monthly_df[available_months] * 100.0).values

    fig = go.Figure(
        data=go.Heatmap(
            z=z_vals,
            x=available_months,
            y=years,
            colorscale="RdYlGn",
            text=z_vals,
            texttemplate="%{text:.1f}%",
            colorbar={"title": "% Return"},
        )
    )
    fig.update_layout(
        title="Monthly Compounded Returns",
        xaxis_title="Month",
        yaxis_title="Year",
        template="plotly_white",
    )
    return fig


def trade_distribution_chart(dist_df: pd.DataFrame) -> go.Figure:
    """Construct trade returns histogram."""
    if dist_df is None or dist_df.empty or "bin_left" not in dist_df.columns:
        return _empty_figure("Trade Returns Distribution")

    labels = [
        f"[{r.bin_left*100:.1f}%, {r.bin_right*100:.1f}%)"
        for _, r in dist_df.iterrows()
    ]
    fig = go.Figure(data=[go.Bar(x=labels, y=dist_df["count"], marker_color="#1f77b4")])
    fig.update_layout(
        title="Trade Return Distribution",
        xaxis_title="Return Bin",
        yaxis_title="Count",
        template="plotly_white",
    )
    return fig


def pnl_by_symbol_chart(sym_df: pd.DataFrame) -> go.Figure:
    """Construct bar chart of PnL by symbol."""
    if sym_df is None or sym_df.empty or "symbol" not in sym_df.columns:
        return _empty_figure("PnL by Symbol")

    colors = ["#2ca02c" if p >= 0 else "#d62728" for p in sym_df["total_pnl"]]
    fig = go.Figure(
        data=[go.Bar(x=sym_df["symbol"], y=sym_df["total_pnl"], marker_color=colors)]
    )
    fig.update_layout(
        title="Realized PnL by Symbol",
        xaxis_title="Symbol",
        yaxis_title="PnL",
        template="plotly_white",
    )
    return fig


def allocation_chart(alloc_df: pd.DataFrame) -> go.Figure:
    """Construct portfolio weights donut chart."""
    if alloc_df is None or alloc_df.empty or "weight" not in alloc_df.columns:
        return _empty_figure("Portfolio Allocation")

    fig = px.pie(
        alloc_df,
        names="symbol",
        values="weight",
        title="Portfolio Allocation",
        hole=0.4,
    )
    fig.update_layout(template="plotly_white")
    return fig
