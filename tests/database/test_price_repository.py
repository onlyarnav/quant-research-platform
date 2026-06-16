"""
Tests for PriceRepository.
"""

from datetime import datetime, timezone

from src.database.models.price import Price


def make_price(**overrides) -> dict:
    """
    Create a valid price payload for testing.
    """

    base = {
        "symbol": "RELIANCE.NS",
        "date": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "open": 2500.0,
        "high": 2550.0,
        "low": 2480.0,
        "close": 2530.0,
        "adj_close": 2530.0,
        "volume": 1_000_000.0,
        "source": "yfinance",
    }

    base.update(overrides)
    return base


# ============================================================
# upsert()
# ============================================================

def test_upsert_inserts_new_price(price_repo):
    price = price_repo.upsert(make_price())

    assert isinstance(price, Price)
    assert price.symbol == "RELIANCE.NS"
    assert price.close == 2530.0
    assert price.source == "yfinance"


def test_upsert_returns_price_with_id(price_repo):
    price = price_repo.upsert(make_price())

    assert price.id is not None


def test_upsert_updates_on_conflict(price_repo):
    price_repo.upsert(
        make_price(close=2530.0)
    )

    updated = price_repo.upsert(
        make_price(close=2600.0)
    )

    assert updated.close == 2600.0


# ============================================================
# get_by_symbol()
# ============================================================

def test_get_by_symbol_returns_inserted_price(price_repo):
    price_repo.upsert(make_price())

    prices = price_repo.get_by_symbol("RELIANCE.NS")

    assert len(prices) == 1


def test_get_by_symbol_returns_empty_for_unknown(price_repo):
    prices = price_repo.get_by_symbol("UNKNOWN")

    assert prices == []


def test_get_by_symbol_order_asc(price_repo):
    price_repo.upsert(
        make_price(
            date=datetime(
                2024,
                1,
                3,
                tzinfo=timezone.utc,
            )
        )
    )

    price_repo.upsert(
        make_price(
            date=datetime(
                2024,
                1,
                1,
                tzinfo=timezone.utc,
            )
        )
    )

    prices = price_repo.get_by_symbol(
        "RELIANCE.NS"
    )

    assert prices[0].date < prices[1].date


# ============================================================
# get_by_date_range()
# ============================================================

def test_get_by_date_range_returns_correct_records(price_repo):
    price_repo.upsert(
        make_price(
            date=datetime(
                2024,
                1,
                1,
                tzinfo=timezone.utc,
            )
        )
    )

    price_repo.upsert(
        make_price(
            date=datetime(
                2024,
                1,
                2,
                tzinfo=timezone.utc,
            )
        )
    )

    price_repo.upsert(
        make_price(
            date=datetime(
                2024,
                1,
                3,
                tzinfo=timezone.utc,
            )
        )
    )

    prices = price_repo.get_by_date_range(
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

    assert len(prices) == 1
    assert prices[0].date == datetime(
        2024,
        1,
        2,
        tzinfo=timezone.utc,
    )


def test_get_by_date_range_returns_empty(price_repo):
    prices = price_repo.get_by_date_range(
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

    assert prices == []


# ============================================================
# bulk_insert()
# ============================================================

def test_bulk_insert_returns_count(price_repo):
    records = [
        make_price(symbol="A.NS"),
        make_price(symbol="B.NS"),
        make_price(symbol="C.NS"),
    ]

    count = price_repo.bulk_insert(records)

    assert count == 3


def test_bulk_insert_empty_returns_zero(price_repo):
    count = price_repo.bulk_insert([])

    assert count == 0


def test_bulk_insert_ignores_duplicates(price_repo):
    price_repo.upsert(make_price())

    count = price_repo.bulk_insert(
        [make_price()]
    )

    assert count == 1


# ============================================================
# delete_by_symbol()
# ============================================================

def test_delete_by_symbol_removes_records(price_repo):
    price_repo.upsert(make_price())

    price_repo.delete_by_symbol(
        "RELIANCE.NS"
    )

    prices = price_repo.get_by_symbol(
        "RELIANCE.NS"
    )

    assert prices == []


def test_delete_by_symbol_returns_count(price_repo):
    price_repo.upsert(make_price())

    deleted = price_repo.delete_by_symbol(
        "RELIANCE.NS"
    )

    assert deleted == 1


def test_delete_by_symbol_returns_zero_for_unknown(price_repo):
    deleted = price_repo.delete_by_symbol(
        "UNKNOWN"
    )

    assert deleted == 0