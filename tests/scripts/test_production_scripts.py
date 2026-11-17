"""
Unit tests for production CLI runner scripts.
"""

from __future__ import annotations

import subprocess
import sys


def test_run_backtest_demo_script() -> None:
    cmd = [
        sys.executable,
        "scripts/run_backtest.py",
        "--demo",
        "--strategy-name",
        "cli_demo_test",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0
    assert "BACKTEST REPORT" in proc.stdout
    assert "Total Return" in proc.stdout


def test_health_check_script_json() -> None:
    cmd = [sys.executable, "scripts/health_check.py", "--json"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    # Output should contain valid JSON structure
    assert '"overall":' in proc.stdout
    assert '"checks":' in proc.stdout


def test_daily_pipeline_dry_run() -> None:
    cmd = [
        sys.executable,
        "scripts/run_daily_pipeline.py",
        "--dry-run",
        "--strategy-name",
        "cli_dry_run",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0
    assert "PIPELINE RUN REPORT" in proc.stdout
    assert "SUCCESS" in proc.stdout
