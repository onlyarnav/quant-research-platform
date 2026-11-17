"""
Unit tests for BacktestStore disk persistence and artifact loading.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.backtesting.backtest_config import BacktestConfig
from src.backtesting.backtest_store import BacktestStore
from src.backtesting.backtesting_engine import Trade
from src.backtesting.simulator_models import SimulationOutput


def _create_mock_output() -> SimulationOutput:
    """Helper to create a valid SimulationOutput instance."""
    t = Trade(
        trade_id=1,
        symbol="TCS",
        entry_date=pd.Timestamp("2024-01-01"),
        exit_date=pd.Timestamp("2024-01-05"),
        entry_price=3000.0,
        exit_price=3100.0,
        position_size=10.0,
        fees=6.1,
        slippage_cost=3.0,
        pnl=990.9,
        return_pct=0.033,
    )
    trades_df = pd.DataFrame(
        [
            {
                "trade_id": 1,
                "symbol": "TCS",
                "entry_date": pd.Timestamp("2024-01-01"),
                "exit_date": pd.Timestamp("2024-01-05"),
                "entry_price": 3000.0,
                "exit_price": 3100.0,
                "position_size": 10.0,
                "fees": 6.1,
                "slippage_cost": 3.0,
                "pnl": 990.9,
                "return_pct": 0.033,
            }
        ]
    )
    trade_details_df = pd.DataFrame(
        [
            {
                "trade_id": 1,
                "side": "long",
                "exit_reason": "signal",
                "gross_pnl": 1000.0,
                "holding_days": 4,
            }
        ]
    )
    hist_df = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-01"),
                "cash": 970000.0,
                "invested_capital": 30000.0,
                "portfolio_value": 1000000.0,
                "daily_return": 0.0,
                "cumulative_return": 0.0,
                "drawdown": 0.0,
            }
        ]
    )
    daily_df = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-01"),
                "gross_exposure": 0.03,
                "net_exposure": 0.03,
                "long_exposure": 0.03,
                "short_exposure": 0.0,
                "n_positions": 1,
                "turnover": 0.03,
                "realized_pnl_cum": 0.0,
                "unrealized_pnl": 0.0,
                "fees_cum": 3.0,
                "slippage_cum": 1.5,
            }
        ]
    )
    pos_df = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-01"),
                "symbol": "TCS",
                "side": 1,
                "quantity": 10.0,
                "price": 3000.0,
                "market_value": 30000.0,
                "weight": 0.03,
                "unrealized_pnl": 0.0,
            }
        ]
    )
    return SimulationOutput(
        trades=[t],
        trades_df=trades_df,
        trade_details_df=trade_details_df,
        portfolio_history_df=hist_df,
        daily_details_df=daily_df,
        positions_history_df=pos_df,
        summary={"final_equity": 1000990.9, "n_trades": 1.0},
    )


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    """Test saving and loading backtest run preserves schemas and values."""
    store = BacktestStore(root=tmp_path / "backtests")
    cfg = BacktestConfig(strategy_name="round_trip_strat")
    out = _create_mock_output()

    run_id = "20240101T000000Z_round_trip_strat_12345678"
    run_dir = store.save_run(run_id, cfg, out)
    assert run_dir.exists()

    record, loaded_out = store.load_run(run_id)
    assert record.run_id == run_id
    assert record.strategy_name == "round_trip_strat"
    assert record.n_trades == 1
    assert len(loaded_out.trades) == 1
    assert loaded_out.trades[0].symbol == "TCS"
    assert len(loaded_out.portfolio_history_df) == 1


def test_list_and_latest_run_id(tmp_path: Path) -> None:
    """Test listing runs newest first and retrieving latest run ID."""
    store = BacktestStore(root=tmp_path / "backtests")
    cfg = BacktestConfig(strategy_name="strat")
    out = _create_mock_output()

    id1 = "20240101T000000Z_strat_11111111"
    id2 = "20240102T000000Z_strat_22222222"
    store.save_run(id1, cfg, out)
    store.save_run(id2, cfg, out)

    runs = store.list_runs()
    assert len(runs) == 2
    assert store.latest_run_id() == id2


def test_security_path_traversal_prevention(tmp_path: Path) -> None:
    """Test invalid path identifiers raise ValueError."""
    store = BacktestStore(root=tmp_path / "backtests")
    cfg = BacktestConfig(strategy_name="strat")
    out = _create_mock_output()

    with pytest.raises(ValueError, match="Invalid path identifier"):
        store.save_run("../malicious_id", cfg, out)

    with pytest.raises(ValueError, match="Invalid path identifier"):
        store.load_run("../malicious_id")


def test_missing_run_raises(tmp_path: Path) -> None:
    """Test loading non-existent run raises FileNotFoundError."""
    store = BacktestStore(root=tmp_path / "backtests")
    with pytest.raises(FileNotFoundError, match="Backtest run 'ghost_run' not found"):
        store.load_run("ghost_run")


def test_save_json_with_nan_serializes_to_null(tmp_path: Path) -> None:
    """Test JSON serialization maps NaN and inf to null."""
    store = BacktestStore(root=tmp_path / "backtests")
    cfg = BacktestConfig(strategy_name="strat")
    out = _create_mock_output()
    run_id = "20240101T000000Z_strat_12345678"
    store.save_run(run_id, cfg, out)

    payload = {"val_nan": float("nan"), "val_inf": float("inf"), "num": 42.0}
    path = store.save_json(run_id, "metrics", payload)
    assert path.exists()

    loaded = store.load_json(run_id, "metrics")
    assert loaded["val_nan"] is None
    assert loaded["val_inf"] is None
    assert loaded["num"] == 42.0


def test_delete_run(tmp_path: Path) -> None:
    """Test delete_run deletes directory and artifacts."""
    store = BacktestStore(root=tmp_path / "backtests")
    cfg = BacktestConfig(strategy_name="strat")
    out = _create_mock_output()
    run_id = "20240101T000000Z_strat_12345678"
    store.save_run(run_id, cfg, out)

    assert (store.root / run_id).exists()
    store.delete_run(run_id)
    assert not (store.root / run_id).exists()
