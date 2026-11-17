"""
CLI runner for the Backtesting Engine, Portfolio Allocators, and Analytics Pipeline.

Supports realistic capital allocation, rebalancing schedules, risk stop rules,
and full performance analytics.

Usage:
    python scripts/run_backtest.py --demo
    python scripts/run_backtest.py --strategy-name trend_following --allocation-method equal_weight
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.analytics.backtest_pipeline import BacktestPipeline
from src.backtesting.backtest_config import (
    AllocationMethod,
    BacktestConfig,
    RebalanceFrequency,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI configuration parameters."""
    parser = argparse.ArgumentParser(
        description="Execute quantitative backtesting pipeline."
    )
    parser.add_argument(
        "--demo", action="store_true", help="Run with synthetic demo data."
    )
    parser.add_argument(
        "--strategy-name",
        type=str,
        default="momentum_daily",
        help="Strategy identifier.",
    )
    parser.add_argument(
        "--allocation-method",
        type=str,
        default="equal_weight",
        choices=[
            "equal_weight",
            "signal_weighted",
            "volatility_weighted",
            "risk_parity",
            "fixed_notional",
        ],
        help="Capital allocation weighting scheme.",
    )
    parser.add_argument(
        "--rebalance",
        type=str,
        default="daily",
        choices=["daily", "weekly", "monthly", "on_signal"],
        help="Portfolio rebalance schedule.",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=settings.INITIAL_CAPITAL,
        help="Starting capital (INR).",
    )
    parser.add_argument(
        "--max-position-size",
        type=float,
        default=settings.MAX_POSITION_SIZE,
        help="Max position weight.",
    )
    parser.add_argument(
        "--cash-reserve",
        type=float,
        default=settings.BACKTEST_CASH_RESERVE,
        help="Unallocated cash fraction.",
    )
    parser.add_argument(
        "--stop-loss", type=float, default=None, help="Stop loss fraction (e.g. 0.05)."
    )
    parser.add_argument(
        "--take-profit",
        type=float,
        default=None,
        help="Take profit fraction (e.g. 0.10).",
    )
    parser.add_argument(
        "--execution-lag", type=int, default=1, help="Execution lag in days."
    )
    parser.add_argument(
        "--allow-short", action="store_true", help="Enable short side positions."
    )
    parser.add_argument(
        "--persist-db",
        action="store_true",
        help="Persist metrics & trades to PostgreSQL.",
    )
    parser.add_argument(
        "--symbols", nargs="*", default=None, help="Specific symbols to include."
    )
    parser.add_argument(
        "--symbol", type=str, default=None, help="Single symbol shorthand."
    )
    parser.add_argument(
        "--start", type=str, default=None, help="Start date filter (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--end", type=str, default=None, help="End date filter (YYYY-MM-DD)."
    )
    return parser.parse_args()


def generate_demo_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate multi-asset synthetic prices and signals for demonstration."""
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    rng = np.random.default_rng(42)
    symbols = ["RELIANCE", "TCS", "INFY"]

    price_records, signal_records = [], []
    for sym in symbols:
        base_px = 1000.0 if sym == "RELIANCE" else (3000.0 if sym == "TCS" else 1500.0)
        returns = rng.normal(0.001, 0.015, len(dates))
        prices = base_px * np.exp(np.cumsum(returns))
        signals = rng.choice([1, 0, -1], size=len(dates), p=[0.20, 0.60, 0.20])

        for d, px, sig in zip(dates, prices, signals):
            price_records.append(
                {"symbol": sym, "date": d, "close": round(float(px), 2)}
            )
            signal_records.append(
                {
                    "symbol": sym,
                    "date": d,
                    "signal": int(sig),
                    "confidence": round(float(rng.uniform(0.5, 0.95)), 4),
                }
            )

    return pd.DataFrame(signal_records), pd.DataFrame(price_records)


def main() -> None:
    """Run CLI backtesting."""
    args = parse_args()
    sym_list = args.symbols or ([args.symbol] if args.symbol else None)

    cfg = BacktestConfig(
        strategy_name=args.strategy_name,
        allocation_method=AllocationMethod(args.allocation_method),
        rebalance_frequency=RebalanceFrequency(args.rebalance),
        initial_capital=args.initial_capital,
        max_position_size=args.max_position_size,
        cash_reserve=args.cash_reserve,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        execution_lag=args.execution_lag,
        allow_short=args.allow_short,
    )

    pipeline = BacktestPipeline()
    signals_df, prices_df = None, None

    if args.demo:
        print("[INFO] Running backtest with synthetic multi-asset demo data...")
        signals_df, prices_df = generate_demo_data()
    else:
        print(
            f"[INFO] Running backtest '{cfg.strategy_name}' across {sym_list or 'all'} symbols..."
        )

    res = pipeline.run(
        config=cfg,
        symbols=sym_list,
        start=args.start,
        end=args.end,
        persist=True,
        persist_db=args.persist_db,
        signals_df=signals_df,
        prices_df=prices_df,
    )

    m = res.report.metrics
    t = res.report.trade_stats
    print("\n" + "=" * 65)
    print(f"             BACKTEST REPORT: {res.run_id}            ")
    print("=" * 65)
    print(f"Strategy Name      : {cfg.strategy_name}")
    print(f"Allocation Scheme  : {cfg.allocation_method.value}")
    print(f"Total Return       : {m.get('total_return', 0.0):+.2%}")
    print(f"Annualized Return  : {m.get('annualized_return', 0.0):+.2%}")
    print(f"Annualized Vol     : {m.get('annualized_volatility', 0.0):.2%}")
    print(f"Sharpe Ratio       : {m.get('sharpe_ratio', float('nan')):.2f}")
    print(f"Sortino Ratio      : {m.get('sortino_ratio', float('nan')):.2f}")
    print(f"Maximum Drawdown   : {m.get('max_drawdown', 0.0):.2%}")
    print(f"Calmar Ratio       : {m.get('calmar_ratio', float('nan')):.2f}")
    print(f"Total Trades       : {int(t.get('total_trades', 0))}")
    print(f"Win Rate           : {t.get('win_rate', 0.0):.1%}")
    print(f"Profit Factor      : {t.get('profit_factor', float('nan')):.2f}")
    print(f"Artifacts Saved To : {res.run_dir}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
