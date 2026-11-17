"""
Rolling and time-series analytics module.

Provides pure functions on portfolio history to compute rolling Sharpe,
rolling volatility, equity curves, underwater drawdown series, monthly return tables,
and exposure series.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from config.settings import settings


def rolling_sharpe(
    df: pd.DataFrame,
    window: int | None = None,
    risk_free_rate: float | None = None,
    trading_days: int | None = None,
) -> pd.DataFrame:
    """
    Compute rolling annualized Sharpe ratio series.

    Returns empty DataFrame if len(df) < window.
    """
    cols = ["date", "rolling_sharpe"]
    if (
        df is None
        or df.empty
        or "date" not in df.columns
        or "daily_return" not in df.columns
    ):
        return pd.DataFrame(columns=cols)

    win = window or settings.ROLLING_WINDOW
    if len(df) < win:
        return pd.DataFrame(columns=cols)

    rf = settings.PORTFOLIO_RISK_FREE_RATE if risk_free_rate is None else risk_free_rate
    periods = settings.TRADING_DAYS_PER_YEAR if trading_days is None else trading_days

    rets = df["daily_return"].astype(float)
    roll_mean = rets.rolling(win).mean()
    roll_std = rets.rolling(win).std(ddof=1)

    ann_ret = roll_mean * periods
    ann_vol = roll_std * math.sqrt(periods)

    sharpe = (ann_ret - rf) / ann_vol
    # Replace inf with nan
    sharpe = sharpe.replace([np.inf, -np.inf], np.nan)

    res = pd.DataFrame({"date": pd.to_datetime(df["date"]), "rolling_sharpe": sharpe})
    return res.dropna().reset_index(drop=True)


def rolling_volatility(
    df: pd.DataFrame,
    window: int | None = None,
    trading_days: int | None = None,
) -> pd.DataFrame:
    """
    Compute rolling annualized volatility series.

    Returns empty DataFrame if len(df) < window.
    """
    cols = ["date", "rolling_volatility"]
    if (
        df is None
        or df.empty
        or "date" not in df.columns
        or "daily_return" not in df.columns
    ):
        return pd.DataFrame(columns=cols)

    win = window or settings.ROLLING_WINDOW
    if len(df) < win:
        return pd.DataFrame(columns=cols)

    periods = settings.TRADING_DAYS_PER_YEAR if trading_days is None else trading_days
    rets = df["daily_return"].astype(float)
    roll_std = rets.rolling(win).std(ddof=1)
    ann_vol = roll_std * math.sqrt(periods)

    res = pd.DataFrame(
        {"date": pd.to_datetime(df["date"]), "rolling_volatility": ann_vol}
    )
    return res.dropna().reset_index(drop=True)


def drawdown_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract drawdown series with underwater flag.
    """
    cols = ["date", "drawdown", "is_underwater"]
    if df is None or df.empty or "date" not in df.columns:
        return pd.DataFrame(columns=cols)

    dd = (
        df["drawdown"]
        if "drawdown" in df.columns
        else (df["portfolio_value"] / df["portfolio_value"].cummax()) - 1.0
    )

    return pd.DataFrame(
        {
            "date": pd.to_datetime(df["date"]),
            "drawdown": dd.astype(float),
            "is_underwater": dd < -1e-9,
        }
    )


def equity_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract portfolio value equity series and cumulative return.
    """
    cols = ["date", "portfolio_value", "cumulative_return"]
    if (
        df is None
        or df.empty
        or "date" not in df.columns
        or "portfolio_value" not in df.columns
    ):
        return pd.DataFrame(columns=cols)

    pv = df["portfolio_value"].astype(float)
    cum_ret = (pv / pv.iloc[0]) - 1.0 if not pv.empty and pv.iloc[0] > 0 else 0.0

    return pd.DataFrame(
        {
            "date": pd.to_datetime(df["date"]),
            "portfolio_value": pv,
            "cumulative_return": cum_ret,
        }
    )


def monthly_returns_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute compounded monthly returns table: year x (Jan..Dec, Year).
    """
    month_names = [
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
    expected_cols = ["Year"] + month_names + ["Total"]

    if (
        df is None
        or df.empty
        or "date" not in df.columns
        or "daily_return" not in df.columns
    ):
        return pd.DataFrame(columns=expected_cols)

    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month

    years = sorted(data["year"].unique())
    records = []

    for yr in years:
        yr_data = data[data["year"] == yr]
        row: dict[str, object] = {"Year": int(yr)}

        # Compounded yearly return
        yr_ret = float(np.prod(1.0 + yr_data["daily_return"].astype(float)) - 1.0)
        row["Total"] = yr_ret

        for m_idx, m_name in enumerate(month_names, start=1):
            m_data = yr_data[yr_data["month"] == m_idx]
            if m_data.empty:
                row[m_name] = np.nan
            else:
                m_ret = float(np.prod(1.0 + m_data["daily_return"].astype(float)) - 1.0)
                row[m_name] = m_ret

        records.append(row)

    return pd.DataFrame(records)[expected_cols]


def exposure_series(daily_details_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Extract long, short, net, and gross exposure series over time.
    """
    cols = ["date", "long_exposure", "short_exposure", "net_exposure", "gross_exposure"]
    if daily_details_df is None or daily_details_df.empty:
        return pd.DataFrame(columns=cols)

    df = daily_details_df.copy()
    if "date" not in df.columns:
        return pd.DataFrame(columns=cols)

    df["date"] = pd.to_datetime(df["date"])

    long_exp = df["long_exposure"] if "long_exposure" in df.columns else 0.0
    short_exp = df["short_exposure"] if "short_exposure" in df.columns else 0.0
    net_exp = (
        df["net_exposure"] if "net_exposure" in df.columns else (long_exp + short_exp)
    )
    gross_exp = (
        df["gross_exposure"]
        if "gross_exposure" in df.columns
        else (
            long_exp.abs() + short_exp.abs()
            if hasattr(long_exp, "abs")
            else abs(long_exp) + abs(short_exp)
        )
    )

    return pd.DataFrame(
        {
            "date": df["date"],
            "long_exposure": long_exp,
            "short_exposure": short_exp,
            "net_exposure": net_exp,
            "gross_exposure": gross_exp,
        }
    )[cols]
