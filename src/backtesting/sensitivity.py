"""
Sensitivity analysis runner across costs, slippage, and execution lag grids.

Evaluates strategy robustness and returns a comparison DataFrame.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pandas as pd

from src.backtesting.backtest_config import BacktestConfig
from src.backtesting.simulator_models import SimulationOutput


def run_sensitivity(
    run_fn: Callable[[BacktestConfig], SimulationOutput],
    base: BacktestConfig,
    cost_grid: Sequence[float] = (0.0, 0.0005, 0.001, 0.002),
    slippage_grid: Sequence[float] = (0.0, 0.0005, 0.001),
    lag_grid: Sequence[int] = (0, 1),
) -> pd.DataFrame:
    """
    Run parameter sensitivity grid and summarize performance stability.

    Args:
        run_fn: Callable taking BacktestConfig and returning SimulationOutput.
        base: Baseline BacktestConfig template.
        cost_grid: Sequence of transaction cost rates to test.
        slippage_grid: Sequence of execution slippage rates to test.
        lag_grid: Sequence of execution lag values in calendar bars.

    Returns:
        DataFrame containing transaction_cost, slippage, execution_lag,
        final_equity, total_return, max_drawdown, and n_trades.
    """
    resolved_base = base.resolved()
    init_cap = resolved_base.initial_capital
    records: list[dict[str, float | int]] = []

    for cost in cost_grid:
        for slip in slippage_grid:
            for lag in lag_grid:
                cfg = BacktestConfig(
                    strategy_name=resolved_base.strategy_name,
                    initial_capital=resolved_base.initial_capital,
                    transaction_cost=cost,
                    slippage=slip,
                    max_position_size=resolved_base.max_position_size,
                    cash_reserve=resolved_base.cash_reserve,
                    allocation_method=resolved_base.allocation_method,
                    rebalance_frequency=resolved_base.rebalance_frequency,
                    rebalance_tolerance=resolved_base.rebalance_tolerance,
                    allow_short=resolved_base.allow_short,
                    stop_loss_pct=resolved_base.stop_loss_pct,
                    take_profit_pct=resolved_base.take_profit_pct,
                    execution_lag=lag,
                    max_open_positions=resolved_base.max_open_positions,
                    min_trade_value=resolved_base.min_trade_value,
                    lookback_window=resolved_base.lookback_window,
                    force_close_at_end=resolved_base.force_close_at_end,
                    risk_free_rate=resolved_base.risk_free_rate,
                )

                output = run_fn(cfg)
                fin_eq = output.summary.get("final_equity", init_cap)
                tot_ret = (fin_eq / init_cap) - 1.0 if init_cap > 0 else 0.0

                if (
                    not output.portfolio_history_df.empty
                    and "drawdown" in output.portfolio_history_df.columns
                ):
                    mdd = float(output.portfolio_history_df["drawdown"].min())
                else:
                    mdd = 0.0

                records.append(
                    {
                        "transaction_cost": float(cost),
                        "slippage": float(slip),
                        "execution_lag": int(lag),
                        "final_equity": float(fin_eq),
                        "total_return": float(tot_ret),
                        "max_drawdown": float(mdd),
                        "n_trades": int(len(output.trades)),
                    }
                )

    return pd.DataFrame(records)
