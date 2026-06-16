"""
Tests for FeatureRepository.
"""

from datetime import datetime, timezone

from src.database.models.feature import Feature


def make_feature(**overrides) -> dict:
    """
    Create a valid feature metadata payload for testing.
    """

    base = {
        "symbol": "RELIANCE.NS",
        "date": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "feature_set_version": "v1.0",
    }

    base.update(overrides)
    return base


# ============================================================
# upsert()
# ============================================================

def test_upsert_inserts_new_feature(feature_repo):
    feature = feature_repo.upsert(make_feature())

    assert isinstance(feature, Feature)
    assert feature.symbol == "RELIANCE.NS"
    assert feature.feature_set_version == "v1.0"


def test_upsert_returns_feature_with_id(feature_repo):
    feature = feature_repo.upsert(make_feature())

    assert feature.id is not None


def test_upsert_updates_on_conflict(feature_repo):
    feature_repo.upsert(
        make_feature(feature_set_version="v1.0")
    )

    updated = feature_repo.upsert(
        make_feature(feature_set_version="v2.0")
    )

    assert updated.feature_set_version == "v2.0"


# ============================================================
# get_by_symbol()
# ============================================================

def test_get_by_symbol_returns_inserted_feature(feature_repo):
    feature_repo.upsert(make_feature())

    features = feature_repo.get_by_symbol("RELIANCE.NS")

    assert len(features) == 1


def test_get_by_symbol_returns_empty_for_unknown(feature_repo):
    features = feature_repo.get_by_symbol("UNKNOWN")

    assert features == []


def test_get_by_symbol_ordered_by_date_asc(feature_repo):
    feature_repo.upsert(
        make_feature(
            date=datetime(
                2024,
                1,
                3,
                tzinfo=timezone.utc,
            )
        )
    )

    feature_repo.upsert(
        make_feature(
            date=datetime(
                2024,
                1,
                1,
                tzinfo=timezone.utc,
            )
        )
    )

    features = feature_repo.get_by_symbol(
        "RELIANCE.NS"
    )

    assert features[0].date < features[1].date


# ============================================================
# get_by_date_range()
# ============================================================

def test_get_by_date_range_returns_correct_records(feature_repo):
    feature_repo.upsert(
        make_feature(
            date=datetime(
                2024,
                1,
                1,
                tzinfo=timezone.utc,
            )
        )
    )

    feature_repo.upsert(
        make_feature(
            date=datetime(
                2024,
                1,
                2,
                tzinfo=timezone.utc,
            )
        )
    )

    feature_repo.upsert(
        make_feature(
            date=datetime(
                2024,
                1,
                3,
                tzinfo=timezone.utc,
            )
        )
    )

    features = feature_repo.get_by_date_range(
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

    assert len(features) == 1
    assert features[0].date == datetime(
        2024,
        1,
        2,
        tzinfo=timezone.utc,
    )


def test_get_by_date_range_returns_empty(feature_repo):
    features = feature_repo.get_by_date_range(
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

    assert features == []


# ============================================================
# get_by_feature_set_version()
# ============================================================

def test_get_by_feature_set_version_returns_matching(
    feature_repo,
):
    feature_repo.upsert(
        make_feature(
            symbol="A.NS",
            feature_set_version="v1.0",
        )
    )

    feature_repo.upsert(
        make_feature(
            symbol="B.NS",
            feature_set_version="v1.0",
        )
    )

    feature_repo.upsert(
        make_feature(
            symbol="C.NS",
            feature_set_version="v2.0",
        )
    )

    features = feature_repo.get_by_feature_set_version(
        "v1.0"
    )

    assert len(features) == 2


def test_get_by_feature_set_version_returns_empty(
    feature_repo,
):
    features = feature_repo.get_by_feature_set_version(
        "v999"
    )

    assert features == []


def test_get_by_feature_set_version_ordered_by_symbol_then_date(
    feature_repo,
):
    feature_repo.upsert(
        make_feature(
            symbol="Z.NS",
            date=datetime(
                2024,
                1,
                2,
                tzinfo=timezone.utc,
            ),
            feature_set_version="v1.0",
        )
    )

    feature_repo.upsert(
        make_feature(
            symbol="A.NS",
            date=datetime(
                2024,
                1,
                1,
                tzinfo=timezone.utc,
            ),
            feature_set_version="v1.0",
        )
    )

    features = feature_repo.get_by_feature_set_version(
        "v1.0"
    )

    assert (
        features[0].symbol,
        features[0].date,
    ) < (
        features[1].symbol,
        features[1].date,
    )


# ============================================================
# bulk_insert()
# ============================================================

def test_bulk_insert_returns_count(feature_repo):
    records = [
        make_feature(symbol="A.NS"),
        make_feature(symbol="B.NS"),
        make_feature(symbol="C.NS"),
    ]

    count = feature_repo.bulk_insert(records)

    assert count == 3


def test_bulk_insert_empty_returns_zero(feature_repo):
    count = feature_repo.bulk_insert([])

    assert count == 0


def test_bulk_insert_ignores_duplicates(feature_repo):
    feature_repo.upsert(make_feature())

    count = feature_repo.bulk_insert(
        [make_feature()]
    )

    assert count == 1


# ============================================================
# delete_by_symbol()
# ============================================================

def test_delete_by_symbol_removes_records(feature_repo):
    feature_repo.upsert(make_feature())

    feature_repo.delete_by_symbol(
        "RELIANCE.NS"
    )

    features = feature_repo.get_by_symbol(
        "RELIANCE.NS"
    )

    assert features == []


def test_delete_by_symbol_returns_count(feature_repo):
    feature_repo.upsert(make_feature())

    deleted = feature_repo.delete_by_symbol(
        "RELIANCE.NS"
    )

    assert deleted == 1


def test_delete_by_symbol_returns_zero_for_unknown(
    feature_repo,
):
    deleted = feature_repo.delete_by_symbol(
        "UNKNOWN"
    )

    assert deleted == 0