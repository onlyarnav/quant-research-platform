"""
CLI runner for the Streamlit research dashboard.

Usage:
    python scripts/run_dashboard.py --port 8501
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start Streamlit research dashboard.")
    parser.add_argument(
        "--port", type=int, default=8501, help="Port to bind Streamlit app."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app_path = Path(__file__).parent.parent / "src" / "dashboard" / "app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(args.port),
        "--server.headless",
        "true",
    ]
    print(
        f"[INFO] Launching Streamlit dashboard from {app_path} on port {args.port}..."
    )
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("[INFO] Streamlit server terminated by user.")


if __name__ == "__main__":
    main()
