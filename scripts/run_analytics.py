"""
CLI tool to inspect, recalculate, and export analytics for stored backtests.

Usage:
    python scripts/run_analytics.py --latest
    python scripts/run_analytics.py --run-id <run_id>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analytics.backtest_pipeline import BacktestPipeline
from src.backtesting.backtest_store import BacktestStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recalculate and display backtest analytics."
    )
    parser.add_argument(
        "--run-id", type=str, default=None, help="Target backtest run identifier."
    )
    parser.add_argument(
        "--latest", action="store_true", help="Inspect the most recent backtest."
    )
    parser.add_argument(
        "--export-json", action="store_true", help="Print report JSON to stdout."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = BacktestStore()

    target_run = args.run_id
    if args.latest or not target_run:
        target_run = store.latest_run_id()
        if not target_run:
            print("[ERROR] No stored backtest runs found on disk.")
            sys.exit(1)

    print(f"[INFO] Loading artifacts and computing analytics for run: {target_run}...")
    pipeline = BacktestPipeline(store=store)

    try:
        report = pipeline.rebuild_report(target_run)
    except Exception as exc:
        print(f"[ERROR] Failed to rebuild report for {target_run}: {exc}")
        sys.exit(1)

    if args.export_json:
        print(json.dumps(report.to_dict(), indent=2))
        return

    m = report.metrics
    t = report.trade_stats
    print("\n" + "=" * 65)
    print(f"             ANALYTICS SUMMARY: {target_run}            ")
    print("=" * 65)
    for k, v in m.items():
        if isinstance(v, float):
            print(f"  {k:<30} : {v:.4f}")
        else:
            print(f"  {k:<30} : {v}")

    print("\n--- TRADE STATISTICS ---")
    for k, v in t.items():
        if isinstance(v, float):
            print(f"  {k:<30} : {v:.4f}")
        else:
            print(f"  {k:<30} : {v}")

    if report.benchmark:
        print("\n--- BENCHMARK COMPARISON ---")
        for k in [
            "alpha",
            "beta",
            "information_ratio",
            "tracking_error",
            "outperformance",
        ]:
            if k in report.benchmark:
                print(f"  {k:<30} : {report.benchmark[k]:.4f}")

    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
