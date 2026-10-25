"""
CLI runner for the Backtesting Engine and Portfolio Simulator.

Provides both a standalone demo mode (no pre-existing data required)
and a production mode that loads signals and processed prices from Parquet.

Usage:
    python scripts/run_backtest.py --demo
    python scripts/run_backtest.py --symbol INFY
    python scripts/run_backtest.py --universe equity
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.backtesting.portfolio_simulator import PortfolioSimulator
from src.data_storage.parquet_writer import DataLayer, ParquetWriter


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run backtesting engine and portfolio simulation."
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run standalone demo with synthetic price/signal data (no files needed).",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Specific symbol to backtest (e.g., INFY).",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=settings.INITIAL_CAPITAL,
        help="Total portfolio starting capital in INR.",
    )
    parser.add_argument(
        "--max-position-size",
        type=float,
        default=settings.MAX_POSITION_SIZE,
        help="Max capital fraction allocated per position (e.g. 0.05).",
    )
    return parser.parse_args()


def generate_demo_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate realistic synthetic price and signal data for demo execution."""
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    rng = np.random.default_rng(42)

    symbols = ["RELIANCE", "TCS", "INFY"]
    price_records = []
    signal_records = []

    for sym in symbols:
        base_price = (
            1000.0 if sym == "RELIANCE" else (3000.0 if sym == "TCS" else 1500.0)
        )
        returns = rng.normal(0.001, 0.015, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))

        # Generate signals: 1 = BUY, -1 = SELL, 0 = HOLD
        raw_signals = rng.choice([1, 0, -1], size=len(dates), p=[0.15, 0.70, 0.15])

        for d, px, sig in zip(dates, prices, raw_signals):
            price_records.append(
                {
                    "symbol": sym,
                    "date": d,
                    "close": round(float(px), 2),
                    "adj_close": round(float(px), 2),
                }
            )
            signal_records.append(
                {
                    "symbol": sym,
                    "date": d,
                    "signal": int(sig),
                    "predicted_return": round(float(rng.normal(0, 0.01)), 4),
                }
            )

    return pd.DataFrame(signal_records), pd.DataFrame(price_records)


def load_parquet_data(symbol: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load signals and price parquet datasets from disk."""
    writer = ParquetWriter()
    signals_dir = Path(settings.DATA_SIGNALS_PATH) / "equity"
    prices_dir = Path(settings.DATA_PROCESSED_PATH) / "equity"

    if not signals_dir.exists() or not any(signals_dir.glob("*.parquet")):
        raise FileNotFoundError(
            f"No signal parquet files found in {signals_dir}. "
            "Run with --demo flag to test without existing files."
        )

    all_signals = []
    all_prices = []

    target_files = (
        [signals_dir / f"equity_{symbol}_daily.parquet"]
        if symbol
        else list(signals_dir.glob("equity_*_daily.parquet"))
    )

    for sig_file in target_files:
        sym = sig_file.name.replace("equity_", "").replace("_daily.parquet", "")
        price_file = prices_dir / f"equity_{sym}_daily.parquet"
        if not price_file.exists():
            continue

        sig_df = writer.read(
            layer=DataLayer.SIGNALS,
            asset_class="equity",
            symbol=sym,
            timeframe="daily",
        )
        px_df = writer.read(
            layer=DataLayer.PROCESSED,
            asset_class="equity",
            symbol=sym,
            timeframe="daily",
        )
        all_signals.append(sig_df)
        all_prices.append(px_df)

    if not all_signals:
        raise ValueError("Could not find matching signal and price data.")

    return pd.concat(all_signals, ignore_index=True), pd.concat(
        all_prices, ignore_index=True
    )


def print_backtest_report(sim_result, initial_capital: float) -> None:
    """Print formatted summary of the backtest simulation."""
    trades = sim_result.trades
    history = sim_result.portfolio_history
    ending_val = history[-1].portfolio_value if history else initial_capital
    total_return = (ending_val / initial_capital) - 1.0
    max_dd = min((s.drawdown for s in history), default=0.0)

    print("\n" + "=" * 65)
    print("                BACKTEST SIMULATION REPORT               ")
    print("=" * 65)
    print(f"Initial Capital    :  INR {initial_capital:,.2f}")
    print(f"Ending Portfolio   :  INR {ending_val:,.2f}")
    print(f"Total Net Return   :  {total_return:+.2%}")
    print(f"Maximum Drawdown   :  {max_dd:.2%}")
    print(f"Total Trades       :  {len(trades)}")

    if trades:
        df = sim_result.trades_df
        wins = df[df["pnl"] > 0]
        losses = df[df["pnl"] < 0]
        win_rate = len(wins) / len(df) if len(df) > 0 else 0.0

        print(f"Win Rate           :  {win_rate:.1%} ({len(wins)}W / {len(losses)}L)")
        print(f"Total Gross PnL    :  INR {df['pnl'].sum() + df['fees'].sum():+,.2f}")
        print(f"Total Fees Paid    :  INR {df['fees'].sum():,.2f}")
        print(f"Total Slippage Cost:  INR {df['slippage_cost'].sum():,.2f}")
        print(f"Total Net PnL      :  INR {df['pnl'].sum():+,.2f}")

        print("\n--- SAMPLE RECENT TRADES ---")
        cols = [
            "trade_id",
            "symbol",
            "entry_date",
            "exit_date",
            "entry_price",
            "exit_price",
            "pnl",
            "return_pct",
        ]
        preview = df[cols].tail(5).copy()
        preview["entry_date"] = preview["entry_date"].dt.strftime("%Y-%m-%d")
        preview["exit_date"] = preview["exit_date"].dt.strftime("%Y-%m-%d")
        preview["entry_price"] = preview["entry_price"].map(lambda x: f"{x:,.1f}")
        preview["exit_price"] = preview["exit_price"].map(lambda x: f"{x:,.1f}")
        preview["pnl"] = preview["pnl"].map(lambda x: f"{x:+,.1f}")
        preview["return_pct"] = preview["return_pct"].map(lambda x: f"{x:+.2%}")
        print(preview.to_string(index=False))

    print("=" * 65 + "\n")


def main() -> None:
    """Execute the backtest runner."""
    args = parse_args()

    if args.demo:
        print(
            "[INFO] Running standalone backtest demo with synthetic multi-asset data..."
        )
        signals_df, prices_df = generate_demo_data()
    else:
        try:
            print(f"[INFO] Loading Parquet data for symbol={args.symbol or 'all'}...")
            signals_df, prices_df = load_parquet_data(args.symbol)
        except Exception as e:
            print(f"[WARNING] {e}")
            print("[INFO] Falling back to demo mode to show backtesting engine...")
            signals_df, prices_df = generate_demo_data()

    simulator = PortfolioSimulator(
        initial_capital=args.initial_capital,
        max_position_size=args.max_position_size,
    )
    result = simulator.run(signals_df, prices_df)
    print_backtest_report(result, args.initial_capital)


if __name__ == "__main__":
    main()
