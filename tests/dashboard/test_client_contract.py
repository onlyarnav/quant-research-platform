"""
Contract tests verifying parity between LocalClient and ApiClient.
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
from src.dashboard.client import ApiClient, LocalClient
from src.dashboard.services import DashboardService


@pytest.fixture
def client_pair():
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = BacktestStore(root=Path(tmp_dir))
        cfg = BacktestConfig(strategy_name="contract_test_strat")
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
        out = SimulationOutput(
            trades=[],
            trades_df=pd.DataFrame(),
            trade_details_df=pd.DataFrame(),
            portfolio_history_df=hist,
            daily_details_df=pd.DataFrame(),
            positions_history_df=pd.DataFrame(),
            summary={
                "n_trades": 0.0,
                "total_fees": 0.0,
                "total_slippage": 0.0,
                "final_equity": 105.0,
            },
        )
        store.save_run("run_contract", cfg, out)
        store.save_json(
            "run_contract",
            "report.json",
            {
                "metrics": {"total_return": 0.05, "sharpe_ratio": 1.5},
                "trade_stats": {},
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
        local_cli = LocalClient(service)

        app = create_app(service=service)
        http_client = TestClient(app)
        api_cli = ApiClient(http_client=http_client)

        yield local_cli, api_cli, "run_contract"


def test_list_runs_parity(client_pair) -> None:
    local_cli, api_cli, run_id = client_pair
    local_runs = local_cli.list_runs()
    api_runs = api_cli.list_runs()

    assert len(local_runs) == len(api_runs)
    assert local_runs[0]["run_id"] == api_runs[0]["run_id"]


def test_get_metrics_parity(client_pair) -> None:
    local_cli, api_cli, run_id = client_pair
    local_metrics = local_cli.get_metrics(run_id)
    api_metrics = api_cli.get_metrics(run_id)

    assert (
        local_metrics["metrics"]["total_return"]
        == api_metrics["metrics"]["total_return"]
    )
    assert (
        local_metrics["metrics"]["sharpe_ratio"]
        == api_metrics["metrics"]["sharpe_ratio"]
    )


def test_get_equity_parity(client_pair) -> None:
    local_cli, api_cli, run_id = client_pair
    local_eq = local_cli.get_equity(run_id)
    api_eq = api_cli.get_equity(run_id)

    assert len(local_eq) == len(api_eq)
    assert local_eq["portfolio_value"].tolist() == api_eq["portfolio_value"].tolist()
