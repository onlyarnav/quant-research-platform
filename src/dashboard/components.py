"""
Streamlit UI layout, KPI, and formatting helper components.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import pandas as pd
import streamlit as st


def format_pct(val: float | None) -> str:
    """Format floating point number as percentage."""
    if val is None or math.isnan(val):
        return "n/a"
    if math.isinf(val):
        return "∞"
    return f"{val * 100.0:.2f}%"


def format_ratio(val: float | None) -> str:
    """Format numerical value as decimal ratio."""
    if val is None or math.isnan(val):
        return "n/a"
    if math.isinf(val):
        return "∞"
    return f"{val:.2f}"


def format_currency(val: float | None) -> str:
    """Format numerical amount as Indian Rupee with Indian digit grouping."""
    if val is None or math.isnan(val):
        return "₹0.00"
    if math.isinf(val):
        return "₹∞"

    is_neg = val < 0
    val_abs = abs(val)
    int_part = int(val_abs)
    dec_part = f"{val_abs - int_part:.2f}"[1:]  # e.g. ".50"

    s = str(int_part)
    if len(s) <= 3:
        grouped = s
    else:
        last3 = s[-3:]
        rem = s[:-3]
        groups = []
        while len(rem) > 2:
            groups.append(rem[-2:])
            rem = rem[:-2]
        if rem:
            groups.append(rem)
        groups.reverse()
        grouped = ",".join(groups) + "," + last3

    prefix = "-₹" if is_neg else "₹"
    return f"{prefix}{grouped}{dec_part}"


def empty_state(message: str = "No data available.") -> None:
    """Display an informational empty state placeholder."""
    st.info(message)


def error_box(exc: Exception | str) -> None:
    """Display a formatted error alert box."""
    st.error(f"Error: {exc}")


def download_csv_button(df: pd.DataFrame, filename: str = "data.csv") -> None:
    """Render a CSV download button for a DataFrame."""
    if df is not None and not df.empty:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=filename,
            mime="text/csv",
        )


def safe_call(
    fn: Callable[..., Any], *args: Any, **kwargs: Any
) -> tuple[Any, Exception | None]:
    """Execute function safely, capturing exceptions without UI crashes."""
    try:
        return fn(*args, **kwargs), None
    except Exception as exc:
        return None, exc


def kpi_row(
    metrics: dict[str, Any], specs: Sequence[tuple[str, str, str | None]]
) -> None:
    """
    Render a horizontal grid row of st.metric cards.

    Args:
        metrics: Key-value dictionary of computed metrics.
        specs: List of tuples (metric_key, label_display_name, format_type).
               format_type in {"pct", "ratio", "currency", None}.
    """
    cols = st.columns(len(specs))
    for i, (key, label, fmt) in enumerate(specs):
        val = metrics.get(key)
        if fmt == "pct":
            display_val = format_pct(val)
        elif fmt == "ratio":
            display_val = format_ratio(val)
        elif fmt == "currency":
            display_val = format_currency(val)
        else:
            display_val = str(val) if val is not None else "n/a"

        cols[i].metric(label=label, value=display_val)
