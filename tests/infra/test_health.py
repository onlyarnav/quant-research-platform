"""
Unit tests for HealthChecker and HealthReport.
"""

from __future__ import annotations

from contextlib import contextmanager

from src.infra.health import HealthCheck, HealthChecker, HealthReport


def test_health_checker_mock_db() -> None:
    @contextmanager
    def mock_good_session():
        class MockSession:
            def execute(self, stmt):
                pass

        yield MockSession()

    checker = HealthChecker(session_factory=mock_good_session)
    res = checker.check_database()
    assert res.status == "ok"
    assert "operational" in res.message


def test_health_checker_db_failure() -> None:
    @contextmanager
    def mock_bad_session():
        raise ConnectionError("DB down")
        yield

    checker = HealthChecker(session_factory=mock_bad_session)
    res = checker.check_database()
    assert res.status == "fail"
    assert "unreachable" in res.message


def test_health_report_synthesis() -> None:
    chk1 = HealthCheck(name="c1", status="ok", message="all good")
    chk2 = HealthCheck(name="c2", status="warn", message="aging")
    rep = HealthReport(
        checks=[chk1, chk2], overall="degraded", generated_at="2024-01-01T00:00:00"
    )

    d = rep.to_dict()
    assert d["overall"] == "degraded"
    assert len(d["checks"]) == 2
