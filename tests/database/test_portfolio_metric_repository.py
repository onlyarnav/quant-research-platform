"""
Tests for PortfolioMetricRepository.
"""

from datetime import datetime, timezone

from src.database.models.portfolio_metric import PortfolioMetric


def make_metric(**overrides) -> dict:
    """
    Create a valid portfolio metric payload for testing.
    """

    base = {
        "strategy_name": "momentum_v1",
        "metric": "sharpe_ratio",
        "value": 1.85,
    }

    base.update(overrides)
    return base


# ============================================================
# insert()
# ============================================================

def test_insert_creates_new_metric(portfolio_repo):
    metric = portfolio_repo.insert(make_metric())

    assert isinstance(metric, PortfolioMetric)
    assert metric.strategy_name == "momentum_v1"
    assert metric.metric == "sharpe_ratio"
    assert metric.value == 1.85


def test_insert_returns_metric_with_id(portfolio_repo):
    metric = portfolio_repo.insert(make_metric())

    assert metric.id is not None


def test_insert_allows_duplicate_strategy_and_metric(
    portfolio_repo,
):
    portfolio_repo.insert(make_metric())

    portfolio_repo.insert(make_metric())

    metrics = portfolio_repo.get_by_strategy(
        "momentum_v1"
    )

    assert len(metrics) == 2


# ============================================================
# get_by_strategy()
# ============================================================

def test_get_by_strategy_returns_inserted_metric(
    portfolio_repo,
):
    portfolio_repo.insert(make_metric())

    metrics = portfolio_repo.get_by_strategy(
        "momentum_v1"
    )

    assert len(metrics) == 1


def test_get_by_strategy_returns_empty_for_unknown(
    portfolio_repo,
):
    metrics = portfolio_repo.get_by_strategy(
        "unknown_strategy"
    )

    assert metrics == []


def test_get_by_strategy_returns_multiple_metrics(
    portfolio_repo,
):
    portfolio_repo.insert(
        make_metric(metric="sharpe_ratio")
    )

    portfolio_repo.insert(
        make_metric(metric="max_drawdown")
    )

    portfolio_repo.insert(
        make_metric(metric="cagr")
    )

    metrics = portfolio_repo.get_by_strategy(
        "momentum_v1"
    )

    assert len(metrics) == 3


# ============================================================
# get_by_date_range()
# ============================================================

def test_get_by_date_range_returns_correct_records(
    portfolio_repo,
):
    portfolio_repo.insert(
        make_metric(
            calculated_at=datetime(
                2024,
                1,
                1,
                tzinfo=timezone.utc,
            )
        )
    )

    portfolio_repo.insert(
        make_metric(
            calculated_at=datetime(
                2024,
                1,
                2,
                tzinfo=timezone.utc,
            )
        )
    )

    portfolio_repo.insert(
        make_metric(
            calculated_at=datetime(
                2024,
                1,
                3,
                tzinfo=timezone.utc,
            )
        )
    )

    metrics = portfolio_repo.get_by_date_range(
        strategy_name="momentum_v1",
        start_date=datetime(
            2024,
            1,
            2,
            tzinfo=timezone.utc,
        ),
        end_date=datetime(
            2024,
            1,
            2,
            tzinfo=timezone.utc,
        ),
    )

    assert len(metrics) == 1
    assert metrics[0].calculated_at == datetime(
        2024,
        1,
        2,
        tzinfo=timezone.utc,
    )


def test_get_by_date_range_returns_empty(
    portfolio_repo,
):
    metrics = portfolio_repo.get_by_date_range(
        strategy_name="momentum_v1",
        start_date=datetime(
            2025,
            1,
            1,
            tzinfo=timezone.utc,
        ),
        end_date=datetime(
            2025,
            1,
            31,
            tzinfo=timezone.utc,
        ),
    )

    assert metrics == []


# ============================================================
# get_by_metric()
# ============================================================

def test_get_by_metric_returns_matching(
    portfolio_repo,
):
    portfolio_repo.insert(
        make_metric(
            strategy_name="momentum_v1",
            metric="sharpe_ratio",
        )
    )

    portfolio_repo.insert(
        make_metric(
            strategy_name="mean_reversion_v1",
            metric="sharpe_ratio",
        )
    )

    portfolio_repo.insert(
        make_metric(
            strategy_name="momentum_v1",
            metric="max_drawdown",
        )
    )

    metrics = portfolio_repo.get_by_metric(
        "sharpe_ratio"
    )

    assert len(metrics) == 2


def test_get_by_metric_returns_empty(
    portfolio_repo,
):
    metrics = portfolio_repo.get_by_metric(
        "unknown_metric"
    )

    assert metrics == []


def test_get_by_metric_ordered_by_strategy_then_calculated_at(
    portfolio_repo,
):
    portfolio_repo.insert(
        make_metric(
            strategy_name="z_strategy",
            metric="sharpe_ratio",
            calculated_at=datetime(
                2024,
                1,
                2,
                tzinfo=timezone.utc,
            ),
        )
    )

    portfolio_repo.insert(
        make_metric(
            strategy_name="a_strategy",
            metric="sharpe_ratio",
            calculated_at=datetime(
                2024,
                1,
                1,
                tzinfo=timezone.utc,
            ),
        )
    )

    metrics = portfolio_repo.get_by_metric(
        "sharpe_ratio"
    )

    assert (
        metrics[0].strategy_name,
        metrics[0].calculated_at,
    ) < (
        metrics[1].strategy_name,
        metrics[1].calculated_at,
    )


# ============================================================
# bulk_insert()
# ============================================================

def test_bulk_insert_returns_count(
    portfolio_repo,
):
    records = [
        make_metric(strategy_name="strategy_1"),
        make_metric(strategy_name="strategy_2"),
        make_metric(strategy_name="strategy_3"),
    ]

    count = portfolio_repo.bulk_insert(records)

    assert count == 3


def test_bulk_insert_empty_returns_zero(
    portfolio_repo,
):
    count = portfolio_repo.bulk_insert([])

    assert count == 0


# ============================================================
# delete_by_strategy()
# ============================================================

def test_delete_by_strategy_removes_records(
    portfolio_repo,
):
    portfolio_repo.insert(make_metric())

    portfolio_repo.delete_by_strategy(
        "momentum_v1"
    )

    metrics = portfolio_repo.get_by_strategy(
        "momentum_v1"
    )

    assert metrics == []


def test_delete_by_strategy_returns_count(
    portfolio_repo,
):
    portfolio_repo.insert(
        make_metric(metric="sharpe_ratio")
    )

    portfolio_repo.insert(
        make_metric(metric="max_drawdown")
    )

    deleted = portfolio_repo.delete_by_strategy(
        "momentum_v1"
    )

    assert deleted == 2


def test_delete_by_strategy_returns_zero_for_unknown(
    portfolio_repo,
):
    deleted = portfolio_repo.delete_by_strategy(
        "unknown_strategy"
    )

    assert deleted == 0