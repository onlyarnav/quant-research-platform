"""Tests for dashboard data_loader module."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd

from src.dashboard.data_loader import (
    _load_available_symbols_impl,
    _load_portfolio_history_impl,
    _load_signals_impl,
    _load_trades_impl,
    _orm_list_to_dataframe_impl,
    orm_list_to_dataframe,
)
from src.database.models.asset import Asset
from src.database.models.signal import Signal


def test_orm_list_to_dataframe_empty_list_returns_typed_empty_df() -> None:
    cols = ["symbol", "entry_date", "pnl"]
    result = orm_list_to_dataframe([], cols)
    assert isinstance(result, pd.DataFrame)
    assert result.empty
    assert list(result.columns) == cols


def test_orm_list_to_dataframe_extracts_correct_values() -> None:
    mock_1 = SimpleNamespace(id=1, symbol="RELIANCE.NS", pnl=100.0)
    mock_2 = SimpleNamespace(id=2, symbol="TCS.NS", pnl=-50.0)

    cols = ["trade_id", "symbol", "pnl"]
    result = _orm_list_to_dataframe_impl([mock_1, mock_2], cols)

    assert len(result) == 2
    assert list(result.columns) == cols
    assert result.iloc[0]["trade_id"] == 1
    assert result.iloc[0]["symbol"] == "RELIANCE.NS"
    assert result.iloc[0]["pnl"] == 100.0
    assert result.iloc[1]["trade_id"] == 2
    assert result.iloc[1]["symbol"] == "TCS.NS"
    assert result.iloc[1]["pnl"] == -50.0


def test_load_trades_impl_empty_db(db_session) -> None:
    start_date = pd.Timestamp("2024-01-01")
    end_date = pd.Timestamp("2024-01-31")

    df = _load_trades_impl("RELIANCE.NS", start_date, end_date)
    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert "trade_id" in df.columns
    assert "symbol" in df.columns


def test_load_trades_impl_with_data(db_session, trade_repo) -> None:
    trade_payload = {
        "symbol": "RELIANCE.NS",
        "entry_date": datetime(2024, 1, 15, tzinfo=timezone.utc),
        "exit_date": datetime(2024, 1, 20, tzinfo=timezone.utc),
        "entry_price": 2500.0,
        "exit_price": 2600.0,
        "position_size": 10.0,
        "fees": 5.0,
        "slippage_cost": 2.5,
        "pnl": 992.5,
        "return_pct": 0.0397,
    }
    trade_repo.insert(trade_payload)
    db_session.flush()

    start_date = pd.Timestamp("2024-01-01")
    end_date = pd.Timestamp("2024-01-31")

    df = _load_trades_impl("RELIANCE.NS", start_date, end_date)
    assert len(df) == 1
    assert df.iloc[0]["symbol"] == "RELIANCE.NS"
    assert df.iloc[0]["pnl"] == 992.5


def test_load_portfolio_history_impl_empty_db(db_session) -> None:
    df = _load_portfolio_history_impl("NON_EXISTENT_STRATEGY")
    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert "date" in df.columns
    assert "portfolio_value" in df.columns


def test_load_portfolio_history_impl_with_data(db_session, portfolio_repo) -> None:
    calc_at = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
    portfolio_repo.insert(
        {
            "strategy_name": "MOMENTUM_TEST",
            "metric": "portfolio_value",
            "value": 100000.0,
            "calculated_at": calc_at,
        }
    )
    portfolio_repo.insert(
        {
            "strategy_name": "MOMENTUM_TEST",
            "metric": "daily_return",
            "value": 0.015,
            "calculated_at": calc_at,
        }
    )
    db_session.flush()

    df = _load_portfolio_history_impl("MOMENTUM_TEST")
    assert len(df) == 1
    assert "date" in df.columns
    assert df.iloc[0]["portfolio_value"] == 100000.0
    assert df.iloc[0]["daily_return"] == 0.015


def test_load_signals_impl_empty_db(db_session) -> None:
    start_date = pd.Timestamp("2024-01-01")
    end_date = pd.Timestamp("2024-01-31")

    df = _load_signals_impl("RELIANCE.NS", start_date, end_date)
    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert "symbol" in df.columns
    assert "signal" in df.columns


def test_load_signals_impl_with_data(db_session) -> None:
    dt = datetime(2024, 1, 15, tzinfo=timezone.utc)
    signal = Signal(
        symbol="RELIANCE.NS",
        date=dt,
        signal=1,
        predicted_return=0.025,
        model_version="v1.0",
    )
    db_session.add(signal)
    db_session.flush()

    start_date = pd.Timestamp("2024-01-01")
    end_date = pd.Timestamp("2024-01-31")

    df = _load_signals_impl("RELIANCE.NS", start_date, end_date)
    assert len(df) == 1
    assert df.iloc[0]["symbol"] == "RELIANCE.NS"
    assert df.iloc[0]["signal"] == 1


def test_load_available_symbols_impl_empty_db(db_session) -> None:
    symbols = _load_available_symbols_impl()
    assert isinstance(symbols, list)


def test_load_available_symbols_impl_with_data(db_session) -> None:
    asset1 = Asset(
        symbol="INFY.NS",
        name="Infosys Limited",
        asset_class="equity",
        currency="INR",
        exchange="NSE",
        is_active=True,
    )
    asset2 = Asset(
        symbol="TCS.NS",
        name="Tata Consultancy Services",
        asset_class="equity",
        currency="INR",
        exchange="NSE",
        is_active=True,
    )
    db_session.add_all([asset1, asset2])
    db_session.flush()

    symbols = _load_available_symbols_impl()
    assert "INFY.NS" in symbols
    assert "TCS.NS" in symbols
