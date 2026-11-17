"""
Unit tests for sensitivity analysis parameter sweep runner.
"""

from __future__ import annotations

import pandas as pd

from src.backtesting.backtest_config import AllocationMethod, BacktestConfig
from src.backtesting.sensitivity import run_sensitivity
from src.backtesting.simulator_accounting import empty_trades_df
from src.backtesting.simulator_models import SimulationOutput


def test_sensitivity_grid_dimensions_and_pure_execution() -> None:
    """Test sensitivity grid evaluates exact product of grid dimensions."""
    base = BacktestConfig(
        strategy_name="test_strat",
        initial_capital=100_000.0,
        allocation_method=AllocationMethod.FIXED_NOTIONAL,
    )

    cost_grid = [0.0, 0.001]
    slippage_grid = [0.0, 0.0005]
    lag_grid = [0, 1]

    def mock_run(cfg: BacktestConfig) -> SimulationOutput:
        # Mock final equity decreasing with costs
        pen = (
            (cfg.transaction_cost or 0) * 1000
            + (cfg.slippage or 0) * 1000
            + (cfg.execution_lag or 0) * 10
        )
        fin = 100_000.0 - pen
        hist = pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-01-01")],
                "cash": [fin],
                "invested_capital": [0.0],
                "portfolio_value": [fin],
                "daily_return": [0.0],
                "cumulative_return": [(fin / 100_000.0) - 1.0],
                "drawdown": [-0.02],
            }
        )
        return SimulationOutput(
            trades=[],
            trades_df=empty_trades_df(),
            trade_details_df=pd.DataFrame(),
            portfolio_history_df=hist,
            daily_details_df=pd.DataFrame(),
            positions_history_df=pd.DataFrame(),
            summary={"final_equity": fin, "n_trades": 0.0},
        )

    res_df = run_sensitivity(
        mock_run,
        base,
        cost_grid=cost_grid,
        slippage_grid=slippage_grid,
        lag_grid=lag_grid,
    )

    expected_rows = len(cost_grid) * len(slippage_grid) * len(lag_grid)  # 2 * 2 * 2 = 8
    assert len(res_df) == expected_rows
    expected_cols = {
        "transaction_cost",
        "slippage",
        "execution_lag",
        "final_equity",
        "total_return",
        "max_drawdown",
        "n_trades",
    }
    assert set(res_df.columns) == expected_cols

    # Verify cost impact is monotonic
    zero_cost_row = res_df[
        (res_df["transaction_cost"] == 0.0)
        & (res_df["slippage"] == 0.0)
        & (res_df["execution_lag"] == 0)
    ]
    high_cost_row = res_df[
        (res_df["transaction_cost"] == 0.001)
        & (res_df["slippage"] == 0.0)
        & (res_df["execution_lag"] == 0)
    ]
    assert float(zero_cost_row["final_equity"].iloc[0]) > float(
        high_cost_row["final_equity"].iloc[0]
    )
