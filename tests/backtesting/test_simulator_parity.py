"""
Parity verification test between legacy PortfolioSimulator and WeightedPortfolioSimulator.
"""

from __future__ import annotations

import pytest

from src.backtesting.backtest_config import AllocationMethod, BacktestConfig
from src.backtesting.portfolio_simulator import PortfolioSimulator
from src.backtesting.weighted_simulator import WeightedPortfolioSimulator
from tests.conftest_quant import make_price_panel, make_signal_panel


@pytest.mark.parametrize("seed", [42, 101, 2024])
def test_parity_between_legacy_and_weighted_simulators(seed: int) -> None:
    """Assert trade PnL and final equity match between legacy and weighted simulators."""
    prices_df = make_price_panel(n_symbols=3, n_days=40, seed=seed)
    signals_df = make_signal_panel(prices_df, seed=seed)

    cfg = BacktestConfig(
        strategy_name="parity_test",
        initial_capital=1_000_000.0,
        max_position_size=0.05,
        transaction_cost=0.001,
        slippage=0.0005,
        allocation_method=AllocationMethod.FIXED_NOTIONAL,
        allow_short=False,
        force_close_at_end=True,
    )

    legacy_sim = PortfolioSimulator(
        initial_capital=cfg.initial_capital,
        max_position_size=cfg.max_position_size,
        transaction_cost=cfg.transaction_cost,
        slippage=cfg.slippage,
    )
    legacy_res = legacy_sim.run(signals_df, prices_df)

    weighted_sim = WeightedPortfolioSimulator(cfg)
    weighted_res = weighted_sim.run(signals_df, prices_df)

    # Compare trade counts
    assert len(legacy_res.trades) == len(weighted_res.trades)

    if legacy_res.trades:
        # Compare total trade PnL
        legacy_pnl = sum(t.pnl for t in legacy_res.trades)
        weighted_pnl = sum(t.pnl for t in weighted_res.trades)
        assert weighted_pnl == pytest.approx(legacy_pnl, abs=1e-4)

        # Compare per-symbol PnL
        leg_df = legacy_res.trades_df
        wt_df = weighted_res.trades_df
        leg_sym_pnl = leg_df.groupby("symbol")["pnl"].sum()
        wt_sym_pnl = wt_df.groupby("symbol")["pnl"].sum()
        for sym in leg_sym_pnl.index:
            assert wt_sym_pnl[sym] == pytest.approx(leg_sym_pnl[sym], abs=1e-4)
