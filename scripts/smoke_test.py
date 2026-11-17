"""
Post-deployment smoke test script.

Validates that an API instance (local or remote/AWS behind Caddy) is healthy,
returns expected status codes, has required security headers, and responds to probes.

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --url http://127.0.0.1:8000
    python scripts/smoke_test.py --url https://your-domain.com/api --api-key secret
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import httpx

REQUIRED_SECURITY_HEADERS = {
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
}


def run_smoke_test(
    base_url: str,
    api_key: str | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Execute smoke test against base_url."""
    base = base_url.rstrip("/")
    headers = {"Accept": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    results: dict[str, Any] = {"base_url": base, "tests": {}, "all_passed": True}

    with httpx.Client(base_url=base, timeout=timeout, headers=headers) as client:
        # 1. Health Probe
        try:
            r = client.get("/health")
            passed = r.status_code == 200 and r.json().get("status") == "ok"
            results["tests"]["health"] = {
                "passed": passed,
                "status_code": r.status_code,
                "data": r.json() if r.status_code == 200 else r.text[:200],
            }
            if not passed:
                results["all_passed"] = False
        except Exception as exc:
            results["tests"]["health"] = {"passed": False, "error": str(exc)}
            results["all_passed"] = False

        # 2. Readiness Probe
        try:
            r = client.get("/ready")
            passed = r.status_code == 200
            results["tests"]["ready"] = {
                "passed": passed,
                "status_code": r.status_code,
                "data": r.json() if r.status_code == 200 else r.text[:200],
            }
            if not passed:
                results["all_passed"] = False
        except Exception as exc:
            results["tests"]["ready"] = {"passed": False, "error": str(exc)}
            results["all_passed"] = False

        # 3. Disclaimer Endpoint
        try:
            r = client.get("/disclaimer")
            passed = r.status_code == 200 and "disclaimer" in r.json()
            results["tests"]["disclaimer"] = {
                "passed": passed,
                "status_code": r.status_code,
            }
            if not passed:
                results["all_passed"] = False
        except Exception as exc:
            results["tests"]["disclaimer"] = {"passed": False, "error": str(exc)}
            results["all_passed"] = False

        # 4. Backtest Runs List
        try:
            r = client.get("/runs")
            passed = r.status_code == 200 and isinstance(r.json(), list)
            results["tests"]["runs"] = {
                "passed": passed,
                "status_code": r.status_code,
                "run_count": len(r.json()) if passed else None,
            }
            if not passed:
                results["all_passed"] = False
        except Exception as exc:
            results["tests"]["runs"] = {"passed": False, "error": str(exc)}
            results["all_passed"] = False

        # 5. Security Headers Check
        try:
            r = client.get("/health")
            header_failures: list[str] = []
            for h_key, h_val in REQUIRED_SECURITY_HEADERS.items():
                actual = r.headers.get(h_key)
                if not actual or actual.lower() != h_val.lower():
                    header_failures.append(f"{h_key} (expected {h_val}, got {actual})")
            sec_passed = len(header_failures) == 0
            results["tests"]["security_headers"] = {
                "passed": sec_passed,
                "failures": header_failures,
            }
            if not sec_passed:
                results["all_passed"] = False
        except Exception as exc:
            results["tests"]["security_headers"] = {"passed": False, "error": str(exc)}
            results["all_passed"] = False

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-deployment HTTP smoke test.")
    parser.add_argument(
        "--url",
        type=str,
        default="http://127.0.0.1:8000",
        help="Target base URL of the API.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional API key for authenticated endpoints.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Request timeout in seconds.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_smoke_test(
        base_url=args.url,
        api_key=args.api_key,
        timeout=args.timeout,
    )

    if args.json:
        print(json.dumps(results, indent=2))
        sys.exit(0 if results["all_passed"] else 1)

    print("=" * 60)
    print(f"SMOKE TEST: {results['base_url']}")
    print("=" * 60)
    for test_name, res in results["tests"].items():
        status_label = "PASS" if res.get("passed") else "FAIL"
        print(f"[{status_label}] {test_name}: {res}")
    print("-" * 60)
    if results["all_passed"]:
        print("RESULT: ALL SMOKE CHECKS PASSED")
        print("=" * 60)
        sys.exit(0)
    else:
        print("RESULT: ONE OR MORE SMOKE CHECKS FAILED")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
