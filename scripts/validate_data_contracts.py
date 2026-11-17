"""
CLI tool to validate Parquet datasets against canonical schemas (docs/Data Schema.md).

Usage:
    python scripts/validate_data_contracts.py
    python scripts/validate_data_contracts.py --data-dir data/
    python scripts/validate_data_contracts.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.infra.contract_validator import validate_data_lake


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Parquet files against canonical data schemas."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Custom root data directory to validate.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output report in JSON format.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_data_lake(data_dir=args.data_dir)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
        sys.exit(0 if report.is_clean else 1)

    print("=" * 60)
    print("DATA CONTRACT VALIDATION REPORT")
    print("=" * 60)
    print(f"Files Checked: {report.files_checked}")
    print(f"Files Passed:  {report.files_passed}")
    print(f"Violations:    {len(report.violations)}")
    print("-" * 60)

    if report.is_clean:
        print("Status: ALL DATA CONTRACTS SATISFIED (PASS)")
        print("=" * 60)
        sys.exit(0)

    print("Status: CONTRACT VIOLATIONS DETECTED (FAIL)")
    for v in report.violations:
        print(f"[{v.tier.upper()}] {v.violation_type}: {v.file_path}")
        print(f"  Details: {v.details}")
    print("=" * 60)
    sys.exit(1)


if __name__ == "__main__":
    main()
