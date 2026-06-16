"""
Tests for SignalRepository.
"""

from datetime import datetime, timezone

from src.database.models.signal import Signal


def make_signal(**overrides) -> dict:
    """
    Create a valid signal payload for testing.
    """

    base = {
        "symbol": "RELIANCE.NS",
        "date": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "signal": 1,
        "predicted_return": 0.025,
        "model_version": "v1.0",
    }

    base.update(overrides)
    return base


# ============================================================
# upsert()
# ============================================================

def test_upsert_inserts_new_signal(signal_repo):
    signal = signal_repo.upsert(make_signal())

    assert isinstance(signal, Signal)
    assert signal.symbol == "RELIANCE.NS"
    assert signal.signal == 1
    assert signal.predicted_return == 0.025


def test_upsert_returns_signal_with_id(signal_repo):
    signal = signal_repo.upsert(make_signal())

    assert signal.id is not None


def test_upsert_updates_on_conflict(signal_repo):
    signal_repo.upsert(
        make_signal(predicted_return=0.025)
    )

    updated = signal_repo.upsert(
        make_signal(predicted_return=0.050)
    )

    assert updated.predicted_return == 0.050


# ============================================================
# get_by_symbol()
# ============================================================

def test_get_by_symbol_returns_inserted_signal(signal_repo):
    signal_repo.upsert(make_signal())

    signals = signal_repo.get_by_symbol("RELIANCE.NS")

    assert len(signals) == 1


def test_get_by_symbol_returns_empty_for_unknown(signal_repo):
    signals = signal_repo.get_by_symbol("UNKNOWN")

    assert signals == []


def test_get_by_symbol_ordered_by_date_asc(signal_repo):
    signal_repo.upsert(
        make_signal(
            date=datetime(
                2024,
                1,
                3,
                tzinfo=timezone.utc,
            )
        )
    )

    signal_repo.upsert(
        make_signal(
            date=datetime(
                2024,
                1,
                1,
                tzinfo=timezone.utc,
            )
        )
    )

    signals = signal_repo.get_by_symbol(
        "RELIANCE.NS"
    )

    assert signals[0].date < signals[1].date


# ============================================================
# get_by_date_range()
# ============================================================

def test_get_by_date_range_returns_correct_records(signal_repo):
    signal_repo.upsert(
        make_signal(
            date=datetime(
                2024,
                1,
                1,
                tzinfo=timezone.utc,
            )
        )
    )

    signal_repo.upsert(
        make_signal(
            date=datetime(
                2024,
                1,
                2,
                tzinfo=timezone.utc,
            )
        )
    )

    signal_repo.upsert(
        make_signal(
            date=datetime(
                2024,
                1,
                3,
                tzinfo=timezone.utc,
            )
        )
    )

    signals = signal_repo.get_by_date_range(
        symbol="RELIANCE.NS",
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

    assert len(signals) == 1
    assert signals[0].date == datetime(
        2024,
        1,
        2,
        tzinfo=timezone.utc,
    )


def test_get_by_date_range_returns_empty(signal_repo):
    signals = signal_repo.get_by_date_range(
        symbol="RELIANCE.NS",
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

    assert signals == []


# ============================================================
# get_by_model_version()
# ============================================================

def test_get_by_model_version_returns_matching(signal_repo):
    signal_repo.upsert(
        make_signal(
            symbol="A.NS",
            model_version="v1.0",
        )
    )

    signal_repo.upsert(
        make_signal(
            symbol="B.NS",
            model_version="v1.0",
        )
    )

    signal_repo.upsert(
        make_signal(
            symbol="C.NS",
            model_version="v2.0",
        )
    )

    signals = signal_repo.get_by_model_version(
        "v1.0"
    )

    assert len(signals) == 2


def test_get_by_model_version_returns_empty(signal_repo):
    signals = signal_repo.get_by_model_version(
        "v999"
    )

    assert signals == []


def test_get_by_model_version_ordered_by_symbol_then_date(
    signal_repo,
):
    signal_repo.upsert(
        make_signal(
            symbol="Z.NS",
            date=datetime(
                2024,
                1,
                2,
                tzinfo=timezone.utc,
            ),
            model_version="v1.0",
        )
    )

    signal_repo.upsert(
        make_signal(
            symbol="A.NS",
            date=datetime(
                2024,
                1,
                1,
                tzinfo=timezone.utc,
            ),
            model_version="v1.0",
        )
    )

    signals = signal_repo.get_by_model_version(
        "v1.0"
    )

    assert (
        signals[0].symbol,
        signals[0].date,
    ) < (
        signals[1].symbol,
        signals[1].date,
    )


# ============================================================
# bulk_insert()
# ============================================================

def test_bulk_insert_returns_count(signal_repo):
    records = [
        make_signal(symbol="A.NS"),
        make_signal(symbol="B.NS"),
        make_signal(symbol="C.NS"),
    ]

    count = signal_repo.bulk_insert(records)

    assert count == 3


def test_bulk_insert_empty_returns_zero(signal_repo):
    count = signal_repo.bulk_insert([])

    assert count == 0


def test_bulk_insert_ignores_duplicates(signal_repo):
    signal_repo.upsert(make_signal())

    count = signal_repo.bulk_insert(
        [make_signal()]
    )

    assert count == 1


# ============================================================
# delete_by_symbol()
# ============================================================

def test_delete_by_symbol_removes_records(signal_repo):
    signal_repo.upsert(make_signal())

    signal_repo.delete_by_symbol(
        "RELIANCE.NS"
    )

    signals = signal_repo.get_by_symbol(
        "RELIANCE.NS"
    )

    assert signals == []


def test_delete_by_symbol_returns_count(signal_repo):
    signal_repo.upsert(make_signal())

    deleted = signal_repo.delete_by_symbol(
        "RELIANCE.NS"
    )

    assert deleted == 1


def test_delete_by_symbol_returns_zero_for_unknown(
    signal_repo,
):
    deleted = signal_repo.delete_by_symbol(
        "UNKNOWN"
    )

    assert deleted == 0