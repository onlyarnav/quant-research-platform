"""
Integration tests for FastAPI dashboard endpoints.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.backtesting.backtest_config import BacktestConfig
from src.backtesting.backtest_store import BacktestStore
from src.backtesting.simulator_models import SimulationOutput
from src.dashboard.api.main import create_app
from src.dashboard.services import DashboardService


@pytest.fixture
def test_setup():
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = BacktestStore(root=Path(tmp_dir))
        cfg = BacktestConfig(strategy_name="api_test_strat")
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        hist = pd.DataFrame(
            {
                "date": dates,
                "portfolio_value": [100.0, 102.0, 101.0, 103.0, 105.0],
                "cash": [100.0] * 5,
                "invested_capital": [0.0] * 5,
                "daily_return": [0.0, 0.02, -0.0098, 0.0198, 0.0194],
                "drawdown": [0.0, 0.0, -0.0098, 0.0, 0.0],
                "total_return": [0.0, 0.02, 0.01, 0.03, 0.05],
            }
        )
        trades = pd.DataFrame(
            {
                "trade_id": [1],
                "symbol": ["AAPL"],
                "entry_date": [dates[0]],
                "exit_date": [dates[2]],
                "entry_price": [100.0],
                "exit_price": [105.0],
                "quantity": [10.0],
                "pnl": [50.0],
                "return_pct": [0.05],
                "fees": [1.0],
                "slippage_cost": [0.5],
            }
        )
        out = SimulationOutput(
            trades=[],
            trades_df=trades,
            trade_details_df=trades,
            portfolio_history_df=hist,
            daily_details_df=pd.DataFrame(),
            positions_history_df=pd.DataFrame(),
            summary={
                "n_trades": 1.0,
                "total_fees": 1.0,
                "total_slippage": 0.5,
                "final_equity": 105.0,
            },
        )
        store.save_run("test_run_123", cfg, out)
        store.save_json(
            "test_run_123",
            "report.json",
            {
                "metrics": {"total_return": 0.05, "sharpe_ratio": 1.5},
                "trade_stats": {"total_trades": 1, "win_rate": 1.0},
                "benchmark": None,
                "equity": hist.to_dict(orient="records"),
                "drawdown": hist[["date", "drawdown"]].to_dict(orient="records"),
                "rolling_sharpe": [],
                "rolling_volatility": [],
                "monthly_returns": [],
                "trade_distribution": [],
                "pnl_by_symbol": [],
                "exposure": [],
            },
        )

        service = DashboardService(store=store)
        app = create_app(service=service)
        client = TestClient(app)
        yield client, "test_run_123"


def test_health_endpoint(test_setup) -> None:
    client, _ = test_setup
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "checks" in data


def test_runs_endpoints(test_setup) -> None:
    client, run_id = test_setup

    # List runs
    resp = client.get("/runs")
    assert resp.status_code == 200
    runs = resp.json()
    assert len(runs) >= 1
    assert runs[0]["run_id"] == run_id

    # Get single run
    resp = client.get(f"/runs/{run_id}")
    assert resp.status_code == 200
    assert resp.json()["run_id"] == run_id

    # Metrics
    resp = client.get(f"/runs/{run_id}/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert data["metrics"]["total_return"] == 0.05

    # Equity & Drawdown
    resp_eq = client.get(f"/runs/{run_id}/equity")
    assert resp_eq.status_code == 200
    assert len(resp_eq.json()["rows"]) == 5

    resp_dd = client.get(f"/runs/{run_id}/drawdown")
    assert resp_dd.status_code == 200
    assert len(resp_dd.json()["rows"]) == 5

    # 404 for unknown run
    resp_404 = client.get("/runs/non_existent_run")
    assert resp_404.status_code == 404


def test_trades_endpoint(test_setup) -> None:
    client, run_id = test_setup
    resp = client.get(f"/runs/{run_id}/trades")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert len(data["items"]) == 1
    assert data["items"][0]["symbol"] == "AAPL"


def test_backtest_submission(test_setup) -> None:
    client, _ = test_setup
    payload = {
        "strategy_name": "test_submit",
        "initial_capital": 50000.0,
    }
    resp = client.post("/backtests", json=payload)
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    assert job_id.startswith("job_")

    # Check status
    st_resp = client.get(f"/backtests/{job_id}")
    assert st_resp.status_code == 200
    assert "status" in st_resp.json()
