"""
Unit tests for dashboard components and formatters.
"""

from __future__ import annotations

from src.dashboard.components import (
    format_currency,
    format_pct,
    format_ratio,
    safe_call,
)


def test_format_pct() -> None:
    assert format_pct(0.1234) == "12.34%"
    assert format_pct(-0.052) == "-5.20%"
    assert format_pct(None) == "n/a"
    assert format_pct(float("nan")) == "n/a"
    assert format_pct(float("inf")) == "∞"


def test_format_ratio() -> None:
    assert format_ratio(1.5432) == "1.54"
    assert format_ratio(0.0) == "0.00"
    assert format_ratio(None) == "n/a"
    assert format_ratio(float("nan")) == "n/a"
    assert format_ratio(float("inf")) == "∞"


def test_format_currency_indian_grouping() -> None:
    # 1000 -> ₹1,000.00
    assert format_currency(1000.0) == "₹1,000.00"
    # 100000 -> ₹1,00,000.00 (1 Lakh)
    assert format_currency(100000.0) == "₹1,00,000.00"
    # 10000000 -> ₹1,00,00,000.00 (1 Crore)
    assert format_currency(10000000.0) == "₹1,00,00,000.00"
    # 1234567.89 -> ₹12,34,567.89
    assert format_currency(1234567.89) == "₹12,34,567.89"
    # Negative
    assert format_currency(-50000.0) == "-₹50,000.00"


def test_safe_call() -> None:
    def good_fn(x, y):
        return x + y

    def bad_fn():
        raise ValueError("Boom")

    val, err = safe_call(good_fn, 2, 3)
    assert val == 5
    assert err is None

    val, err = safe_call(bad_fn)
    assert val is None
    assert isinstance(err, ValueError)
