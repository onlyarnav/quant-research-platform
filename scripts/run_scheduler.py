"""
CLI daemon runner for the background pipeline scheduler.

Usage:
    python scripts/run_scheduler.py
    python scripts/run_scheduler.py --time 08:30 --timezone Asia/Kolkata
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.infra.scheduler import start_background_scheduler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start background pipeline scheduler.")
    parser.add_argument(
        "--time",
        type=str,
        default=settings.PIPELINE_RUN_TIME,
        help="Execution time (HH:MM).",
    )
    parser.add_argument(
        "--timezone",
        type=str,
        default=settings.PIPELINE_TIMEZONE,
        help="Timezone string.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[INFO] Initializing scheduler: trigger at {args.time} ({args.timezone})...")
    start_background_scheduler(run_time=args.time, timezone=args.timezone)


if __name__ == "__main__":
    main()
