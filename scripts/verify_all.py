"""
Unified verification tool for the Quantitative AI Research Platform.

Runs static analysis, unit/db/integration tests, data contracts, and e2e smoke checks.

Modes:
    python scripts/verify_all.py --fast    # Fast static checks & quick unit tests (<60s)
    python scripts/verify_all.py           # Full test suite + data contracts + linting
    python scripts/verify_all.py --docker  # Validate Docker & Compose production specs
    python scripts/verify_all.py --e2e     # End-to-end pipeline & backtest dry runs
    python scripts/verify_all.py --json    # Output aggregated JSON summary
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


@dataclass
class StepResult:
    name: str
    command: str
    duration_s: float
    passed: bool
    output: str


@dataclass
class VerificationSummary:
    mode: str
    total_steps: int = 0
    passed_steps: int = 0
    failed_steps: int = 0
    total_duration_s: float = 0.0
    steps: list[StepResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return self.failed_steps == 0

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "all_passed": self.all_passed,
            "total_steps": self.total_steps,
            "passed_steps": self.passed_steps,
            "failed_steps": self.failed_steps,
            "total_duration_s": round(self.total_duration_s, 2),
            "steps": [
                {
                    "name": s.name,
                    "command": s.command,
                    "passed": s.passed,
                    "duration_s": round(s.duration_s, 2),
                    "output_tail": s.output[-500:] if s.output else "",
                }
                for s in self.steps
            ],
        }


def run_command_step(name: str, cmd: list[str]) -> StepResult:
    """Execute a single shell command step and measure wall time."""
    start = time.perf_counter()
    cmd_str = " ".join(cmd)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=300,
            check=False,
        )
        duration = time.perf_counter() - start
        return StepResult(
            name=name,
            command=cmd_str,
            duration_s=duration,
            passed=(proc.returncode == 0),
            output=proc.stdout or "",
        )
    except subprocess.TimeoutExpired:
        duration = time.perf_counter() - start
        return StepResult(
            name=name,
            command=cmd_str,
            duration_s=duration,
            passed=False,
            output="Execution timed out after 300 seconds",
        )
    except Exception as exc:
        duration = time.perf_counter() - start
        return StepResult(
            name=name,
            command=cmd_str,
            duration_s=duration,
            passed=False,
            output=f"Failed to launch command: {exc}",
        )


def get_steps_for_mode(mode: str) -> list[tuple[str, list[str]]]:
    """Define execution steps based on requested mode."""
    steps: list[tuple[str, list[str]]] = []

    if mode == "docker":
        steps.append(("docker_version", ["docker", "--version"]))
        steps.append(
            (
                "compose_config_validation",
                ["docker", "compose", "-f", "docker-compose.prod.yml", "config"],
            )
        )
        return steps

    # Fast / Static steps
    steps.append(
        (
            "black_check",
            [
                PYTHON,
                "-m",
                "black",
                "--check",
                "--line-length",
                "88",
                "src",
                "tests",
                "config",
                "scripts",
            ],
        )
    )
    steps.append(
        (
            "isort_check",
            [
                PYTHON,
                "-m",
                "isort",
                "--check-only",
                "src",
                "tests",
                "config",
                "scripts",
            ],
        )
    )
    steps.append(
        ("flake8_lint", [PYTHON, "-m", "flake8", "src", "tests", "config", "scripts"])
    )
    steps.append(("mypy_typecheck", [PYTHON, "-m", "mypy", "src", "config"]))
    steps.append(
        ("bandit_security", [PYTHON, "-m", "bandit", "-r", "src", "-ll", "-ii", "-q"])
    )
    steps.append(
        (
            "docstring_coverage",
            [PYTHON, "-m", "interrogate", "-v", "--fail-under=80", "src"],
        )
    )
    steps.append(("data_contracts", [PYTHON, "scripts/validate_data_contracts.py"]))

    if mode == "fast":
        steps.append(
            (
                "fast_unit_tests",
                [PYTHON, "-m", "pytest", "tests", "-m", "not slow and not db", "-q"],
            )
        )
        return steps

    # Full suite steps
    steps.append(("full_pytest_suite", [PYTHON, "-m", "pytest", "tests", "-q"]))

    if mode == "e2e":
        steps.append(
            ("pipeline_dry_run", [PYTHON, "scripts/run_daily_pipeline.py", "--dry-run"])
        )
        steps.append(("backtest_demo", [PYTHON, "scripts/run_backtest.py", "--demo"]))
        steps.append(("analytics_run", [PYTHON, "scripts/run_analytics.py"]))
        steps.append(("health_check", [PYTHON, "scripts/health_check.py"]))

    return steps


def main() -> None:
    parser = argparse.ArgumentParser(description="One-command verification runner.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--fast", action="store_true", help="Run fast static & unit checks (<60s)"
    )
    group.add_argument(
        "--docker", action="store_true", help="Validate Docker and prod compose"
    )
    group.add_argument(
        "--e2e", action="store_true", help="Run full pipeline & backtest dry runs"
    )
    parser.add_argument(
        "--json", action="store_true", help="Output summary in JSON format"
    )

    args = parser.parse_args()
    mode = (
        "docker"
        if args.docker
        else ("fast" if args.fast else ("e2e" if args.e2e else "full"))
    )

    steps_to_run = get_steps_for_mode(mode)
    summary = VerificationSummary(mode=mode, total_steps=len(steps_to_run))

    if not args.json:
        print("=" * 70)
        print(f"QUANT PLATFORM VERIFICATION RUNNER (Mode: {mode.upper()})")
        print("=" * 70)

    start_total = time.perf_counter()
    for name, cmd in steps_to_run:
        if not args.json:
            print(f"-> Running: {name} ... ", end="", flush=True)
        res = run_command_step(name, cmd)
        summary.steps.append(res)
        if res.passed:
            summary.passed_steps += 1
            if not args.json:
                print(f"PASS ({res.duration_s:.1f}s)")
        else:
            summary.failed_steps += 1
            if not args.json:
                print(f"FAIL ({res.duration_s:.1f}s)")
                print("-" * 50)
                print(res.output[:800])
                print("-" * 50)

    summary.total_duration_s = time.perf_counter() - start_total

    if args.json:
        print(json.dumps(summary.to_dict(), indent=2))
        sys.exit(0 if summary.all_passed else 1)

    print("=" * 70)
    print(
        f"VERIFICATION SUMMARY: {summary.passed_steps}/{summary.total_steps} Passed ({summary.total_duration_s:.1f}s)"
    )
    if summary.all_passed:
        print("OVERALL STATUS: ALL GATES PASSED (PASS)")
        print("=" * 70)
        sys.exit(0)
    else:
        print(f"OVERALL STATUS: {summary.failed_steps} GATE(S) FAILED (FAIL)")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
