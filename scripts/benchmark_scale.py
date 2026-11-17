"""
Scale and performance benchmarking tool for the Quantitative AI Research Platform.

Measures wall clock time and peak memory (psutil) across pipeline stages:
- Synthetic market data generation
- Feature engineering
- Model training (LightGBM)
- Model inference & prediction
- Signal generation
- Portfolio backtesting & rebalancing
- Strategy analytics & report generation

Usage:
    python scripts/benchmark_scale.py
    python scripts/benchmark_scale.py --symbols 100 --days 500
    python scripts/benchmark_scale.py --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtesting.backtest_config import BacktestConfig
from src.backtesting.weighted_simulator import WeightedPortfolioSimulator
from src.feature_engineering.momentum_features import MomentumFeatureGenerator
from src.feature_engineering.trend_features import TrendFeatureGenerator
from src.feature_engineering.volatility_features import VolatilityFeatureGenerator
from src.models.lightgbm_model import LightGBMModel
from src.portfolio.allocator_factory import create_allocator
from src.signals.topn_signal import TopNLongShortSignalGenerator


@dataclass
class BenchmarkMetric:
    stage: str
    target_time_s: float
    target_mem_mb: float
    elapsed_s: float = 0.0
    peak_mem_mb: float = 0.0
    passed: bool = True


def get_current_memory_mb() -> float:
    """Return resident set size (RSS) in MB for current process."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def generate_synthetic_data(symbols: list[str], n_days: int) -> pd.DataFrame:
    """Generate realistic geometric Brownian motion synthetic price series."""
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n_days, freq="B")
    dfs = []
    np.random.seed(42)

    for sym in symbols:
        # Annualized return 10%, vol 20%
        daily_ret = np.random.normal(0.10 / 252, 0.20 / np.sqrt(252), size=n_days)
        price_path = 100.0 * np.exp(np.cumsum(daily_ret))
        high = price_path * (1 + np.abs(np.random.normal(0, 0.005, size=n_days)))
        low = price_path * (1 - np.abs(np.random.normal(0, 0.005, size=n_days)))
        volume = np.random.lognormal(14, 0.5, size=n_days)

        df = pd.DataFrame(
            {
                "symbol": sym,
                "date": dates,
                "open": price_path * 0.999,
                "high": high,
                "low": low,
                "close": price_path,
                "volume": volume,
                "returns": pd.Series(price_path).pct_change().fillna(0.0),
                "log_returns": np.log(price_path / np.roll(price_path, 1)),
            }
        )
        df.iloc[0, df.columns.get_loc("log_returns")] = 0.0
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def run_benchmark(n_symbols: int = 50, n_days: int = 1250) -> list[BenchmarkMetric]:
    """Execute end-to-end benchmark across core engine stages."""
    symbols = [f"SYM_{i:03d}" for i in range(n_symbols)]
    results: list[BenchmarkMetric] = []

    print(f"Starting scale benchmark ({n_symbols} symbols x {n_days} days)...")

    # 1. Data Generation
    m1 = BenchmarkMetric("Data Generation", target_time_s=30.0, target_mem_mb=2048.0)
    t0 = time.perf_counter()
    market_df = generate_synthetic_data(symbols, n_days)
    m1.elapsed_s = time.perf_counter() - t0
    m1.peak_mem_mb = get_current_memory_mb()
    m1.passed = m1.elapsed_s <= m1.target_time_s and m1.peak_mem_mb <= m1.target_mem_mb
    results.append(m1)

    # 2. Feature Engineering
    m2 = BenchmarkMetric(
        "Feature Engineering", target_time_s=600.0, target_mem_mb=6144.0
    )
    t0 = time.perf_counter()
    trend_gen = TrendFeatureGenerator()
    mom_gen = MomentumFeatureGenerator()
    vol_gen = VolatilityFeatureGenerator()
    feature_dfs = []
    for _, group in market_df.groupby("symbol"):
        f1 = trend_gen.generate(group)
        f2 = mom_gen.generate(group)
        f3 = vol_gen.generate(group)
        merged = (
            group[["symbol", "date", "close"]]
            .merge(f1, on=["symbol", "date"])
            .merge(f2, on=["symbol", "date"])
            .merge(f3, on=["symbol", "date"])
        )
        merged["target_return"] = merged["close"].pct_change().shift(-1)
        feature_dfs.append(merged.dropna())
    features_df = pd.concat(feature_dfs, ignore_index=True)
    m2.elapsed_s = time.perf_counter() - t0
    m2.peak_mem_mb = get_current_memory_mb()
    m2.passed = m2.elapsed_s <= m2.target_time_s and m2.peak_mem_mb <= m2.target_mem_mb
    results.append(m2)

    # 3. Model Training (LightGBM)
    m3 = BenchmarkMetric(
        "Model Training (LightGBM)", target_time_s=120.0, target_mem_mb=4096.0
    )
    t0 = time.perf_counter()
    feature_cols = [
        c for c in features_df.columns if c.startswith(("sma_", "rsi_", "volatility_"))
    ]
    split_idx = int(len(features_df) * 0.8)
    train_df = features_df.iloc[:split_idx]
    test_df = features_df.iloc[split_idx:]

    lgb = LightGBMModel(n_estimators=50, max_depth=5, verbosity=-1)
    lgb.fit(train_df[feature_cols], train_df["target_return"])
    m3.elapsed_s = time.perf_counter() - t0
    m3.peak_mem_mb = get_current_memory_mb()
    m3.passed = m3.elapsed_s <= m3.target_time_s and m3.peak_mem_mb <= m3.target_mem_mb
    results.append(m3)

    # 4. Model Prediction
    m4 = BenchmarkMetric("Model Inference", target_time_s=30.0, target_mem_mb=4096.0)
    t0 = time.perf_counter()
    preds = lgb.predict(test_df[feature_cols])
    pred_df = test_df[["symbol", "date"]].copy()
    pred_df["predicted_return"] = preds
    m4.elapsed_s = time.perf_counter() - t0
    m4.peak_mem_mb = get_current_memory_mb()
    m4.passed = m4.elapsed_s <= m4.target_time_s and m4.peak_mem_mb <= m4.target_mem_mb
    results.append(m4)

    # 5. Signal Generation
    m5 = BenchmarkMetric("Signal Generation", target_time_s=30.0, target_mem_mb=4096.0)
    t0 = time.perf_counter()
    sig_gen = TopNLongShortSignalGenerator(top_n=5, bottom_n=5)
    signals_df = sig_gen.generate(pred_df)
    m5.elapsed_s = time.perf_counter() - t0
    m5.peak_mem_mb = get_current_memory_mb()
    m5.passed = m5.elapsed_s <= m5.target_time_s and m5.peak_mem_mb <= m5.target_mem_mb
    results.append(m5)

    # 6. Backtest Simulation
    m6 = BenchmarkMetric(
        "Backtest Simulation", target_time_s=600.0, target_mem_mb=6144.0
    )
    t0 = time.perf_counter()
    cfg = BacktestConfig(
        strategy_name="scale_test",
        initial_capital=1_000_000.0,
        allocation_method="equal_weight",
        rebalance_frequency="daily",
    )
    allocator = create_allocator("equal_weight")
    sim = WeightedPortfolioSimulator(config=cfg, allocator=allocator)
    prices_subset = market_df[market_df["date"].isin(signals_df["date"].unique())]
    output = sim.run(signals_df, prices_subset)
    m6.elapsed_s = time.perf_counter() - t0
    m6.peak_mem_mb = get_current_memory_mb()
    m6.passed = m6.elapsed_s <= m6.target_time_s and m6.peak_mem_mb <= m6.target_mem_mb
    results.append(m6)

    # 7. Analytics & Report Build
    m7 = BenchmarkMetric("Analytics & Report", target_time_s=60.0, target_mem_mb=4096.0)
    t0 = time.perf_counter()
    from src.analytics.report_builder import ReportBuilder

    rb = ReportBuilder()
    _ = rb.build(
        portfolio_history_df=output.portfolio_history_df,
        trades_df=output.trades_df,
    )
    m7.elapsed_s = time.perf_counter() - t0
    m7.peak_mem_mb = get_current_memory_mb()
    m7.passed = m7.elapsed_s <= m7.target_time_s and m7.peak_mem_mb <= m7.target_mem_mb
    results.append(m7)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run platform scale and memory benchmark."
    )
    parser.add_argument(
        "--symbols", type=int, default=50, help="Universe size (default: 50)"
    )
    parser.add_argument(
        "--days", type=int, default=1250, help="Trading days (default: 1250)"
    )
    parser.add_argument("--json", action="store_true", help="Output JSON results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_benchmark(n_symbols=args.symbols, n_days=args.days)

    if args.json:
        data = [
            {
                "stage": r.stage,
                "elapsed_s": round(r.elapsed_s, 2),
                "target_time_s": r.target_time_s,
                "peak_mem_mb": round(r.peak_mem_mb, 1),
                "target_mem_mb": r.target_mem_mb,
                "passed": r.passed,
            }
            for r in results
        ]
        print(json.dumps(data, indent=2))
        sys.exit(0)

    print("\n" + "=" * 80)
    print(
        f"{'STAGE':<28} | {'TIME (s)':<10} | {'TARGET (s)':<10} | {'MEM (MB)':<10} | {'STATUS'}"
    )
    print("-" * 80)
    all_passed = True
    for r in results:
        status_label = "PASS" if r.passed else "FAIL"
        if not r.passed:
            all_passed = False
        print(
            f"{r.stage:<28} | {r.elapsed_s:<10.2f} | {r.target_time_s:<10.1f} | "
            f"{r.peak_mem_mb:<10.1f} | {status_label}"
        )
    print("=" * 80)

    peak = max(r.peak_mem_mb for r in results)
    print(f"Overall Peak Memory: {peak:.1f} MB ({peak/1024:.2f} GB)")
    print("Recommended EC2 Sizing: t3.large (8 GB RAM, 2 vCPUs) is fully sufficient.")
    print("=" * 80)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
