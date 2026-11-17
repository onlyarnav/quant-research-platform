"""
CLI runner for the FastAPI dashboard and analytics backend.

Usage:
    python scripts/run_api.py --port 8000 --reload
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start FastAPI backend server.")
    parser.add_argument(
        "--host", type=str, default=settings.API_HOST, help="Host interface to bind."
    )
    parser.add_argument(
        "--port", type=int, default=settings.API_PORT, help="Port to listen on."
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload on code changes."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[INFO] Starting FastAPI application on http://{args.host}:{args.port}...")
    uvicorn.run(
        "src.dashboard.api.main:app", host=args.host, port=args.port, reload=args.reload
    )


if __name__ == "__main__":
    main()
