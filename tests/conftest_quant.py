"""
Shared fixtures and synthetic data generators for quantitative tests.

Provides realistic geometric random walk prices, signals, canonical history,
trade logs, and populated store fixtures for backtesting, portfolio, analytics,
and dashboard test suites.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.backtesting.backtest_config import BacktestConfig
from src.backtesting.backtest_store import BacktestStore
from src.backtesting.backtesting_engine import Trade
from src.backtesting.simulator_accounting import (
    CANONICAL_HISTORY_COLS,
    CANONICAL_TRADE_COLS,
)
from src.backtesting.simulator_models import SimulationOutput


def make_price_panel(
    n_symbols: int = 4,
    n_days: int = 100,
    seed: int = 42,
    tz: str | None = None,
) -> pd.DataFrame:
    """Generate multi-asset geometric random walk daily prices."""
    rng = np.random.default_rng(seed)
    symbols = [f"SYM_{i:02d}" for i in range(n_symbols)]
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B", tz=tz)

    records = []
    for sym in symbols:
        base = rng.uniform(50.0, 500.0)
        returns = rng.normal(0.0005, 0.015, n_days)
        prices = base * np.exp(np.cumsum(returns))
        vols = rng.integers(10_000, 1_000_000, n_days)
        for dt, px, vol in zip(dates, prices, vols):
            px_val = round(float(px), 2)
            records.append(
                {
                    "symbol": sym,
                    "date": dt,
                    "close": px_val,
                    "adj_close": px_val,
                    "volume": int(vol),
                }
            )

    df = pd.DataFrame(records)
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


def make_signal_panel(
    prices_df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate matching model signals (-1, 0, 1) with confidence and predicted return."""
    rng = np.random.default_rng(seed)
    records = []
    for row in prices_df.itertuples(index=False):
        sig = int(rng.choice([1, 0, -1], p=[0.25, 0.50, 0.25]))
        pred_ret = float(rng.normal(0.001, 0.02))
        conf = float(rng.uniform(0.5, 0.99))
        records.append(
            {
                "symbol": row.symbol,
                "date": row.date,
                "signal": sig,
                "predicted_return": round(pred_ret, 4),
                "confidence": round(conf, 4),
                "rank": int(rng.integers(1, 10)),
            }
        )
    return pd.DataFrame(records).sort_values(["symbol", "date"]).reset_index(drop=True)


def make_history_frame(
    values: list[float],
    start_date: str = "2024-01-01",
    initial_capital: float | None = None,
) -> pd.DataFrame:
    """Generate canonical 7-column portfolio history from a sequence of portfolio values."""
    if not values:
        raise ValueError("values cannot be empty")
    n = len(values)
    dates = pd.date_range(start_date, periods=n, freq="B")
    init_cap = initial_capital if initial_capital is not None else values[0]

    vals = np.array(values, dtype=float)
    daily_rets = np.zeros(n, dtype=float)
    if n > 1:
        daily_rets[1:] = vals[1:] / vals[:-1] - 1.0

    cum_rets = vals / init_cap - 1.0
    running_max = np.maximum.accumulate(vals)
    dds = (vals - running_max) / running_max

    records = []
    for i in range(n):
        pv = vals[i]
        inv = pv * 0.8
        cash = pv * 0.2
        records.append(
            {
                "date": dates[i],
                "cash": cash,
                "invested_capital": inv,
                "portfolio_value": pv,
                "daily_return": daily_rets[i],
                "cumulative_return": cum_rets[i],
                "drawdown": dds[i],
            }
        )
    return pd.DataFrame(records)[CANONICAL_HISTORY_COLS]


def make_trades_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Construct canonical 11-column trades DataFrame from trade records."""
    if not rows:
        return pd.DataFrame(columns=CANONICAL_TRADE_COLS)
    df = pd.DataFrame(rows)
    for c in CANONICAL_TRADE_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df[CANONICAL_TRADE_COLS]


@pytest.fixture
def price_panel_fixture() -> pd.DataFrame:
    """Provide a standard 4-symbol, 60-day price panel."""
    return make_price_panel(n_symbols=4, n_days=60, seed=42)


@pytest.fixture
def signal_panel_fixture(price_panel_fixture: pd.DataFrame) -> pd.DataFrame:
    """Provide a matching signal panel."""
    return make_signal_panel(price_panel_fixture, seed=42)


@pytest.fixture
def populated_store(tmp_path) -> BacktestStore:
    """Provide a BacktestStore with one pre-saved deterministic run."""
    store = BacktestStore(root=tmp_path / "backtests")
    cfg = BacktestConfig(strategy_name="baseline_test")
    hist_df = make_history_frame([100.0, 102.0, 101.0, 105.0])

    t = Trade(
        trade_id=1,
        symbol="SYM_00",
        entry_date=pd.Timestamp("2024-01-01"),
        exit_date=pd.Timestamp("2024-01-04"),
        entry_price=100.0,
        exit_price=105.0,
        position_size=10.0,
        fees=2.0,
        slippage_cost=1.0,
        pnl=47.0,
        return_pct=0.047,
    )
    trades_df = pd.DataFrame(
        [
            {
                "trade_id": 1,
                "symbol": "SYM_00",
                "entry_date": pd.Timestamp("2024-01-01"),
                "exit_date": pd.Timestamp("2024-01-04"),
                "entry_price": 100.0,
                "exit_price": 105.0,
                "position_size": 10.0,
                "fees": 2.0,
                "slippage_cost": 1.0,
                "pnl": 47.0,
                "return_pct": 0.047,
            }
        ]
    )
    details_df = pd.DataFrame(
        [
            {
                "trade_id": 1,
                "side": "long",
                "exit_reason": "signal",
                "gross_pnl": 50.0,
                "holding_days": 3,
            }
        ]
    )
    daily_df = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-01"),
                "gross_exposure": 0.8,
                "net_exposure": 0.8,
                "long_exposure": 0.8,
                "short_exposure": 0.0,
                "n_positions": 1,
                "turnover": 0.8,
                "realized_pnl_cum": 47.0,
                "unrealized_pnl": 0.0,
                "fees_cum": 2.0,
                "slippage_cum": 1.0,
            }
        ]
    )
    pos_df = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-01"),
                "symbol": "SYM_00",
                "side": 1,
                "quantity": 10.0,
                "price": 100.0,
                "market_value": 1000.0,
                "weight": 0.8,
                "unrealized_pnl": 0.0,
            }
        ]
    )

    out = SimulationOutput(
        trades=[t],
        trades_df=trades_df,
        trade_details_df=details_df,
        portfolio_history_df=hist_df,
        daily_details_df=daily_df,
        positions_history_df=pos_df,
        summary={"final_equity": 105.0, "n_trades": 1.0},
    )
    store.save_run(
        run_id="20240101T000000Z_baseline_test_12345678",
        config=cfg,
        output=out,
    )
    return store
