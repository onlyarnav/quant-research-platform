"""
CLI system health verification tool.

Usage:
    python scripts/health_check.py
    python scripts/health_check.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.infra.health import HealthChecker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run platform health inspection.")
    parser.add_argument(
        "--json", action="store_true", help="Output health report as JSON."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checker = HealthChecker()
    report = checker.run_all()

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print("\n" + "=" * 65)
        print("                SYSTEM OPERATIONAL HEALTH REPORT         ")
        print("=" * 65)
        print(f"Overall Status : {report.overall.upper()}")
        print(f"Generated At   : {report.generated_at}")
        print("\nComponent Checks:")
        for c in report.checks:
            icon = "+" if c.status == "ok" else ("!" if c.status == "warn" else "x")
            print(f"  [{icon}] {c.name:<18} : {c.status.upper():<8} - {c.message}")
            if c.details:
                print(f"      Details: {c.details}")
        print("=" * 65 + "\n")

    sys.exit(0 if report.overall != "down" else 1)


if __name__ == "__main__":
    main()
