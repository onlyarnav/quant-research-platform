"""
CLI tool to execute the automated daily quantitative pipeline.

Executes ingestion, feature generation, ML inference, signal generation,
backtesting, and analytics persistence sequentially.

Usage:
    python scripts/run_daily_pipeline.py --dry-run
    python scripts/run_daily_pipeline.py --strategy-name daily_prod
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtesting.backtest_config import BacktestConfig
from src.infra.pipeline_orchestrator import PipelineOrchestrator
from src.infra.pipeline_steps import PipelineContext


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run daily production pipeline.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate pipeline steps without disk writes.",
    )
    parser.add_argument(
        "--strategy-name",
        type=str,
        default="daily_production",
        help="Backtest strategy name.",
    )
    parser.add_argument(
        "--asset-class", type=str, default="equity", help="Target asset class."
    )
    parser.add_argument(
        "--timeframe", type=str, default="daily", help="Target timeframe."
    )
    parser.add_argument(
        "--symbols", nargs="*", default=None, help="Subset of symbols to process."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_pipeline"

    context = PipelineContext(
        run_id=run_id,
        asset_class=args.asset_class,
        timeframe=args.timeframe,
        symbols=args.symbols,
        dry_run=args.dry_run,
        backtest_config=BacktestConfig(strategy_name=args.strategy_name),
    )

    print(
        f"[INFO] Launching daily pipeline run_id={run_id} (dry_run={args.dry_run})..."
    )
    orchestrator = PipelineOrchestrator()
    report = orchestrator.run(context)

    print("\n" + "=" * 65)
    print(f"             PIPELINE RUN REPORT: {report.run_id}            ")
    print("=" * 65)
    print(
        f"Overall Result     : {'SUCCESS' if report.all_succeeded else 'FAILED / PARTIAL'}"
    )
    print(f"Total Duration     : {report.total_duration_seconds:.2f}s")
    print("\nStep Summary:")
    for step in report.steps:
        status_icon = (
            "+"
            if step.status.value == "success"
            else ("x" if step.status.value == "failed" else "-")
        )
        print(
            f"  [{status_icon}] {step.name:<25} : {step.status.value.upper():<8} ({step.duration_seconds:.2f}s)"
        )
        if step.message:
            print(f"      Details: {step.message}")

    print("=" * 65 + "\n")
    sys.exit(0 if report.all_succeeded else 1)


if __name__ == "__main__":
    main()
